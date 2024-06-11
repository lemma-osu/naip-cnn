from __future__ import annotations

import json
import tempfile
from pathlib import Path

import tensorflow as tf
import wandb

from naip_cnn.augment import Augment
from naip_cnn.config import WANDB_PROJECT
from naip_cnn.data import NAIPDatasetWrapper
from naip_cnn.models import ModelRun


def load_wandb_model(run_path: str) -> tf.keras.Model:
    """Load a model logged with a W&B run."""
    run = wandb.Api().run(run_path)

    model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if len(model_artifacts) != 1:
        raise ValueError(f"Expected one model artifact, found {len(model_artifacts)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(model_artifacts[0].download(root=tmpdir))
        # Download returns a directory with one model file
        model_path = next(model_dir.glob("*.keras"))
        return tf.keras.models.load_model(model_path)


def load_wandb_model_run(run_path: str) -> ModelRun:
    """Load a model run from a W&B run."""
    run = wandb.Api().run(run_path)
    cfg = run.config
    model = load_wandb_model(run_path)

    model_params = cfg["model"]["params"]
    bands = tuple(cfg["data"]["imagery"]["bands"].split("-"))
    label = cfg["data"]["lidar"]["label"]
    dataset_name = Path(cfg["data"]["train"]["path"]).stem.replace("_train", "")

    dataset = NAIPDatasetWrapper.from_filename(dataset_name)

    return ModelRun(model, model_params, dataset, label, bands)


def initialize_wandb_run(
    *,
    dataset: NAIPDatasetWrapper,
    model_run: ModelRun,
    bands: tuple[str],
    label: str,
    batch_size: int,
    learn_rate: float,
    epochs: int,
    n_train: int,
    n_val: int,
    augmenter: Augment | None = None,
    allow_duplicate: bool = False,
) -> wandb.apis.public.runs.Run:
    """Initialize a W&B run for tracking an experiment."""
    group = f"{dataset.lidar_res:n}m_{label}"

    config = _build_wandb_config(
        dataset=dataset,
        model_run=model_run,
        bands=bands,
        label=label,
        batch_size=batch_size,
        learn_rate=learn_rate,
        epochs=epochs,
        n_train=n_train,
        n_val=n_val,
        augmenter=augmenter,
    )

    if not allow_duplicate:
        prev_runs = wandb.Api().runs(WANDB_PROJECT)

        for prev_run in prev_runs:
            if _configs_are_equal(prev_run.config, config):
                raise ValueError(
                    f"Configuration matches an existing run ({prev_run.url}). "
                    "To allow duplicate configurations, set `allow_duplicate=True`."
                )

    return wandb.init(project=WANDB_PROJECT, config=config, group=group, save_code=True)


def _build_wandb_config(
    *,
    dataset: NAIPDatasetWrapper,
    model_run: ModelRun,
    bands: tuple[str],
    label: str,
    batch_size: int,
    learn_rate: float,
    epochs: int,
    n_train: int,
    n_val: int,
    augmenter: Augment | None = None,
) -> dict:
    """Build a configuration dictionary for tracking an experiment with W&B."""
    return {
        "training": {
            "batch_size": batch_size,
            "learning_rate": learn_rate,
            "epochs": epochs,
        },
        "model": {
            "architecture": model_run.model.name,
            "path": model_run.model_path.as_posix(),
            "params": model_run.model_params,
        },
        "data": {
            "train": {
                "path": dataset._train.path.as_posix(),
                "n_samples": n_train,
                "augmentation": augmenter.to_json() if augmenter is not None else None,
            },
            "val": {
                "path": dataset._val.path.as_posix(),
                "n_samples": n_val,
            },
            "date": {
                "start": dataset.acquisition.start_date,
                "end": dataset.acquisition.end_date,
            },
            "footprint": {
                "shape": dataset.footprint,
                "spacing": dataset.spacing,
                "units": "meters",
            },
            "imagery": {
                "bands": "-".join(bands),
                "resolution": dataset.naip_res,
                "acquisition": dataset.acquisition.name,
            },
            "lidar": {
                "label": label,
                "resolution": dataset.lidar_res,
                "asset": dataset.acquisition.lidar_asset,
            },
        },
    }


def _configs_are_equal(config1, config2):
    """
    Compare two configuration dictionaries for equality.

    Note that we implement this from scratch to rather than a simple equality check
    because values may be modified by W&B, e.g. converting tuples to lists and floats
    to ints.
    """
    # Normalize to JSON to, e.g. convert tuples to lists
    config1 = json.loads(json.dumps(config1))
    config2 = json.loads(json.dumps(config2))

    # Check if the comparison is None, which may occur for nested checks
    if config2 is None:
        return False

    for key, value in config1.items():
        # Keys are mismatched
        if key not in config2:
            return False

        # Compare nested dictionaries
        if isinstance(value, dict) and not _configs_are_equal(value, config2[key]):
            return False

        # Values are mismatched
        if value != config2[key]:
            return False

    return True
