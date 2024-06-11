from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from naip_cnn.config import MODEL_DIR
from naip_cnn.data import NAIPDatasetWrapper
from naip_cnn.utils.wandb import load_wandb_model_run


@dataclass
class ModelRun:
    """A model run with associated model, dataset, and parameters."""

    model: tf.keras.Model
    model_params: dict
    dataset: NAIPDatasetWrapper
    label: str
    bands: tuple[str]
    suffix: str = None

    def __post_init__(self):
        band_str = "".join(self.bands)
        parts = [self.model.name, self.dataset.name, self.label, band_str]
        if self.suffix:
            parts.append(self.suffix)

        self.name = "-".join(parts)
        self.model_path = MODEL_DIR / f"{self.name}.keras"

        if "regularization" in self.model_params:
            # Serialize the regularization function
            self.model_params["regularization"] = tf.keras.regularizers.serialize(
                self.model_params["regularization"]
            )

    def __repr__(self) -> str:
        return f"<ModelRun name={self.name}>"

    def load_best_checkpoint(self, delete_checkpoints=True) -> int:
        """Load the best weights from a checkpoint and return the associated epoch.

        This assumes that checkpoints are named ".checkpoint_{run.name}_{epoch}.h5" and
        were generated with `save_best_only=True`, so that the last checkpoint saved is
        the best. All checkpoints are optionally deleted after the best one is loaded.
        """
        checkpoints = list(MODEL_DIR.glob(f".checkpoint_{self.name}_*.h5"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found.")

        checkpoints.sort()
        checkpoint_path = checkpoints[-1]
        epoch = int(checkpoint_path.stem.split("_")[-1])

        self.model.load_weights(checkpoint_path)

        if delete_checkpoints:
            for file in checkpoints:
                file.unlink()

        return epoch

    def save_model(self) -> Path:
        """Save the model to disk."""
        self.model.save(self.model_path)
        return self.model_path

    @staticmethod
    def from_filename(filename: str) -> ModelRun:
        """Load a model run from a filename."""
        basename = Path(filename).stem
        parts = basename.split("-")
        bands = tuple(parts[-1])
        label = parts[-2]
        dataset_name = "-".join(parts[1:-2])

        model = tf.keras.models.load_model(MODEL_DIR / f"{basename}.keras")
        bands = tuple(bands)
        dataset = NAIPDatasetWrapper.from_filename(dataset_name)

        return ModelRun(
            model, model_params={}, dataset=dataset, label=label, bands=bands
        )

    @staticmethod
    def from_wandb_run(run_path: str) -> ModelRun:
        """Load a model run from a W&B run."""
        return load_wandb_model_run(run_path)


def CNN_v1(
    kernel_dim=5,
    filter_no=32,
    Dense1_no=256,
    Dense2_no=32,
    shape: tuple[int, int, int] = (30, 30, 4),
):
    """Original model by Adam Sibley."""
    kernel_size = (kernel_dim, kernel_dim, 4)

    return tf.keras.models.Sequential(
        [
            layers.Conv3D(
                filters=filter_no,
                kernel_size=kernel_size,
                input_shape=(*shape, 1),
                padding="valid",
                activation="relu",
                use_bias=True,
            ),
            layers.Flatten(),
            layers.Dense(Dense1_no, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(Dense2_no, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(units=1, activation="linear"),
        ],
        name="CNN_v1",
    )


def _encoder_block(
    x,
    n_filters,
    filter_size,
    pool_size,
    convolutions_per_block,
    dropout,
    regularization=None,
):
    for _ in range(convolutions_per_block):
        x = layers.Conv2D(
            n_filters,
            filter_size,
            padding="same",
            activation="relu",
            kernel_regularizer=regularization,
        )(x)

    p = layers.MaxPool2D(pool_size)(x)

    if dropout is not None:
        p = layers.Dropout(dropout)(p)

    return x, p


def _decoder_block(x, conv_features, n_filters, regularization=None):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Conv2D(
        n_filters,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularization,
    )(x)
    return layers.Conv2D(
        n_filters,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularization,
    )(x)


def CNN_v2(
    shape: tuple[int, int, int] = (150, 150, 4),
    out_shape=(5, 5),
    encoder_blocks=4,
    initial_filters=16,
    filter_size=(3, 3),
    pool_size=(2, 2),
    convolutions_per_block=2,
    dropout=0.3,
    regularization=None,
    name="CNN",
):
    """A simple, configurable CNN model."""
    inputs = x = tf.keras.layers.Input(shape=shape)

    # Build the encoder blocks
    for i in range(encoder_blocks):
        filters = initial_filters * 2**i

        _, x = _encoder_block(
            x,
            filters,
            filter_size=filter_size,
            pool_size=pool_size,
            convolutions_per_block=convolutions_per_block,
            dropout=dropout,
            regularization=regularization,
        )

    # Build the flatten and dense output layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=out_shape[0] * out_shape[1], activation="linear")(x)

    outputs = layers.Reshape(out_shape)(x)

    return tf.keras.Model(inputs, outputs, name=name)


def UNet_v1(
    in_shape,
    out_shape,
    encoder_blocks=4,
    max_filters=512,
    dropout=0.3,
    regularization=None,
    name="UNet",
):
    """A modified UNet with a configurable number of encoder blocks and a calculated
    number of decoder blocks."""
    if in_shape[0] < out_shape[0]:
        raise ValueError("The output shape must be <= the input shape.")
    # Calculate the image size after encoding
    encoded_size = in_shape[0] / 2**encoder_blocks
    if not encoded_size.is_integer():
        raise ValueError("The input shape must be divisible by 2 ** encoder_blocks.")

    # Calculate the number of decoder blocks to achieve the output shape
    decoder_blocks = np.log2(out_shape[0] / encoded_size)
    if not decoder_blocks.is_integer():
        raise ValueError(
            f"The output shape must be divisible by the encoded size ({encoded_size})."
        )

    # Calcuate the exponential of the first filter size
    start_filter_exp = np.log2(max_filters) - encoder_blocks

    inputs = x = tf.keras.layers.Input(shape=in_shape)

    # Build the encoder blocks
    encoders = []
    for i in range(encoder_blocks):
        filters = 2 ** (start_filter_exp + i)

        encoder, pooling = _encoder_block(
            x, filters, dropout=dropout, regularization=regularization
        )
        encoders.append(encoder)

        x = pooling

    # Bottleneck
    bottleneck_filters = 2 ** (start_filter_exp + encoder_blocks)
    x = layers.Conv2D(bottleneck_filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(bottleneck_filters, 3, padding="same", activation="relu")(x)

    # Build the decoder blocks
    for i in range(int(decoder_blocks)):
        filters = 2 ** (start_filter_exp + encoder_blocks - i)
        x = _decoder_block(
            x, encoders[-(i + 1)], filters, regularization=regularization
        )

    outputs = layers.Conv2D(
        1, 1, activation="linear", kernel_regularizer=regularization
    )(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model
