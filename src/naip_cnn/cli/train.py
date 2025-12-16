from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import naip_cnn.config as proj_config
import wandb
from naip_cnn import models
from naip_cnn.acquisitions import Acquisition
from naip_cnn.data import NAIPDatasetWrapper
from naip_cnn.utils.training import R2Score2D, SaveBestWeights
from naip_cnn.utils.wandb import initialize_wandb_run

from . import config


@dataclass
class TrainingResult:
    model_run: models.ModelRun
    best_epoch: int
    stopped_epoch: int
    interrupted: bool = False


def load_data() -> tuple[tf.data.Dataset, tf.data.Dataset, NAIPDatasetWrapper]:
    acquisitions = [Acquisition.from_name(name) for name in config.DATASETS]
    wrapper = NAIPDatasetWrapper(
        acquisitions,
        naip_res=config.NAIP_RES,
        lidar_res=config.LIDAR_RES,
        footprint=config.FOOTPRINT,
        spacing=config.SPACING,
    )

    train = (
        wrapper.load_train(
            label=config.LABEL,
            bands=config.BANDS,
            veg_indices=config.VEG_INDICES,
            augmenter=config.AUGMENT,
        )
        .cache()
        .shuffle(buffer_size=1_000)
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val = (
        wrapper.load_val(
            label=config.LABEL, bands=config.BANDS, veg_indices=config.VEG_INDICES
        )
        .cache()
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train, val, wrapper


def load_model_run(wrapper) -> models.ModelRun:
    model = config.MODEL_CLS(
        shape=(*wrapper.naip_shape, len(config.ALL_BANDS)),
        out_shape=wrapper.lidar_shape,
        **config.MODEL_PARAMS,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARN_RATE),
        loss=config.LOSS_FUNCTION,
        metrics=["mae", "mse", R2Score2D()],
        run_eagerly=False,
    )

    return models.ModelRun(
        model=model,
        model_params=config.MODEL_PARAMS,
        dataset=wrapper,
        label=config.LABEL,
        bands=config.ALL_BANDS,
    )


def train_model(
    model_run: models.ModelRun, train: tf.data.Dataset, val: tf.data.Dataset
) -> TrainingResult:
    checkpoint = SaveBestWeights(model_directory="./models", verbose=False)
    early_stopping = EarlyStopping(verbose=1, patience=config.PATIENCE)
    wandb_logger = wandb.keras.WandbMetricsLogger()

    try:
        model_run.model.fit(
            train,
            verbose=1,
            validation_data=val,
            epochs=config.EPOCHS,
            callbacks=[
                checkpoint,
                early_stopping,
                wandb_logger,
            ],
        )
    # Allow manual early stopping
    except KeyboardInterrupt:
        model_run.model.stop_training = True
        interrupted = True
    else:
        interrupted = False

    best_epoch, stopped_epoch = checkpoint.best_epoch, checkpoint.last_epoch

    checkpoint.load_best_weights()

    return TrainingResult(model_run, best_epoch, stopped_epoch, interrupted)


def evaluate_model(training_result: TrainingResult, val: tf.data.Dataset) -> dict:
    y_pred = training_result.model_run.model.predict(val)
    y_true = np.concatenate([data[1] for data in val.as_numpy_iterator()])
    metric_vals = training_result.model_run.model.evaluate(val)

    metrics = {
        "best_epoch": training_result.best_epoch,
        "stopped_epoch": training_result.stopped_epoch,
    }

    for metric, value in zip(
        training_result.model_run.model.metrics_names, metric_vals
    ):
        # Prefix all metrics with "final/" to differentiate them from epoch metrics
        metrics[f"final/{metric}"] = value

    # Create evaluation figures
    log_correlation_scatterplot(y_true, y_pred)
    log_distribution_histogram(y_true, y_pred)

    return metrics


def log_distribution_histogram(y_true, y_pred):
    """Log a histogram of the true and predicted values to W&B."""
    fig, ax = plt.subplots(figsize=(6, 4))

    _, bins, _ = ax.hist(y_true.ravel(), bins=100, alpha=0.5, label="y_true")
    ax.hist(y_pred.ravel(), bins=bins, alpha=0.5, label="y_pred")
    ax.legend()
    ax.set_yticks([])
    plt.tight_layout()

    wandb.log({"hist": wandb.Image(fig)})


def log_correlation_scatterplot(y_true, y_pred):
    """Log a scatterplot of the true and predicted values to W&B."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.1, marker=".")
    ax.plot(
        (y_true.min(), y_true.max()), (y_true.min(), y_true.max()), "k-.", alpha=0.75
    )
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    plt.tight_layout()

    wandb.log({"scatter": wandb.Image(fig)})


def train(
    allow_duplicate: bool = False,
    allow_cpu: bool = False,
    dry_run: bool = False,
    debug: bool = False,
):
    """Train a new model and log it to W&B."""
    if not allow_cpu:
        msg = "No GPU detected. Use --allow-cpu to train anyways."
        assert tf.config.list_physical_devices("GPU"), msg
    test_run = dry_run or debug

    # Load the data
    train, val, wrapper = load_data()

    # Build the model
    model_run = load_model_run(wrapper)

    # Initialize the tracking experiment
    run = initialize_wandb_run(
        dataset=wrapper,
        model_run=model_run,
        bands=config.ALL_BANDS,
        label=config.LABEL,
        batch_size=config.BATCH_SIZE,
        learn_rate=config.LEARN_RATE,
        epochs=config.EPOCHS,
        n_train=len(train) * config.BATCH_SIZE,
        n_val=len(val) * config.BATCH_SIZE,
        augmenter=config.AUGMENT,
        allow_duplicate=True if test_run else allow_duplicate,
        mode="disabled" if test_run else "online",
    )

    if dry_run:
        import json

        print(json.dumps(dict(run.config), indent=2))
        print(model_run.model.summary())
        return

    if debug:
        tf.debugging.disable_traceback_filtering()
        train_model(model_run, train.take(1), val.take(1))
        return

    # Train and save the model
    training_result = train_model(model_run, train, val)

    if training_result.interrupted:
        prompt = "\n\nTraining interrupted. Continue evaluating model? [Y/n]: "
        if input(prompt).lower() == "n":
            wandb.run.finish()

            # There's a subtle difference between the run type returned by `wandb.init`
            # and `wandb.Api().run`. Only the latter can be deleted programatically.
            api_run = wandb.Api().run(run.path)
            api_run.delete(delete_artifacts=True)

            print("Run deleted.")
            return

    # Save the repository state and model as artifacts
    wandb.run.log_code()
    wandb.run.log_model(
        training_result.model_run.save_model(),
        name=f"{wandb.run.name}_model",
        aliases=[proj_config.MODEL_VERSION],
    )

    # Evaluate the model
    summary = evaluate_model(training_result, val)
    summary["interrupted"] = training_result.interrupted
    wandb.run.summary.update(summary)

    # Notify on run completion, unless the user ended the run manually
    if not training_result.interrupted:
        run_summary = (
            f"R^2: {summary['final/r2_score']:.4f}, MAE: {summary['final/mae']:.4f}  "
        )
        wandb.alert(title="Run Complete", text=run_summary)

    # Mark the run as complete
    wandb.run.finish()
