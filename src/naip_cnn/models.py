from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras import layers

from naip_cnn.config import MODEL_DIR, PRED_DIR
from naip_cnn.data import NAIPDatasetWrapper, NAIPTFRecord


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

    def predict(
        self,
        dataset_id: str,
        year: int,
        batch_size: int = 256,
        filename: str | None = None,
        apply_mask: bool = False,
        **kwargs,
    ):
        """Predict a label for each pixel in a NAIP image from a given year."""
        tfrecord = NAIPTFRecord(
            id=dataset_id,
            year=year,
            footprint=self.dataset.footprint,
            res=self.dataset.naip_res,
        )

        # Ensure the mask file exists BEFORE running prediction
        if apply_mask and not tfrecord.mask_path.exists():
            msg = f"Expected a mask GeoTIFF at {tfrecord.mask_path} but found none."
            raise ValueError(msg)

        image = tfrecord.load_dataset(bands=self.bands).batch(batch_size)
        raw_pred = self.model.predict(
            image, steps=tfrecord.n_batches(batch_size), **kwargs
        )
        h, w = tfrecord.profile["height"], tfrecord.profile["width"]
        pred = (
            raw_pred.reshape(h, w, *self.dataset.lidar_shape)
            .swapaxes(1, 2)
            .reshape(h * self.dataset.lidar_shape[0], w * self.dataset.lidar_shape[1])
            .clip(min=0)
        )

        if apply_mask:
            with rasterio.open(tfrecord.mask_path) as src:
                mask = src.read(1)

                assert mask.shape == pred.shape, (
                    f"Mask shape {mask.shape} does not match prediction shape"
                    f" {pred.shape}."
                )
                pred[mask == 0] = 255

        # Optionally write out to geotiff
        if filename:
            pred_path = PRED_DIR / f"{filename}.tif"
            pred_profile = tfrecord.profile.copy()

            pred_profile["transform"][0] = self.dataset.lidar_res
            pred_profile["transform"][4] = self.dataset.lidar_res
            pred_profile.update(
                {
                    "width": pred.shape[1],
                    "height": pred.shape[0],
                    "dtype": "uint8",
                    "count": 1,
                    "compress": "DEFLATE",
                    "nodata": 255,
                }
            )

            with rasterio.open(pred_path, "w", **pred_profile) as dst:
                dst.write(pred.astype(np.uint8), 1)

            return pred_path

        return pred

    def save_model(self) -> Path:
        """Save the model to disk."""
        self.model.save(self.model_path)
        return self.model_path

    @staticmethod
    def from_wandb_run(run_path: str) -> ModelRun:
        """Load a model run from a W&B run."""
        from naip_cnn.utils.wandb import load_wandb_model_run

        return load_wandb_model_run(run_path)


def CNN_original(
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
        name="CNN_original",
    )


def _encoder_block(
    x,
    n_filters,
    filter_size,
    pool_size,
    convolutions_per_block,
    dropout,
    regularization=None,
    activation="relu",
    **kwargs,
):
    for _ in range(convolutions_per_block):
        x = layers.Conv2D(
            n_filters,
            filter_size,
            padding="same",
            activation=activation,
            kernel_regularizer=regularization,
            **kwargs,
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


def CNN_base(
    shape: tuple[int, int, int] = (150, 150, 4),
    out_shape=(5, 5),
    encoder_blocks=4,
    initial_filters=16,
    filter_size=(3, 3),
    pool_size=(2, 2),
    convolutions_per_block=2,
    dropout=0.3,
    regularization=None,
    activation="relu",
    **kwargs,
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
            activation=activation,
            **kwargs,
        )

    # Build the flatten and dense output layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=out_shape[0] * out_shape[1], activation="linear")(x)

    outputs = layers.Reshape(out_shape)(x)

    return tf.keras.Model(inputs, outputs, name="CNN_base")


def CNN_resized(
    shape: tuple[int, int, int] = (30, 30, 4),
    resize_shape: tuple[int, int] = (64, 64),
    out_shape=(1, 1),
    encoder_blocks=4,
    initial_filters=16,
    filter_size=(3, 3),
    pool_size=(2, 2),
    convolutions_per_block=2,
    dropout=0.3,
    regularization=None,
    activation="relu",
    **kwargs,
):
    """
    A simple, configurable CNN model with a resizing layer.

    This addition was based on the fact that the 1m NAIP resampled to 0.6m seemed to
    perform better.
    """
    inputs = x = tf.keras.layers.Input(shape=shape)

    x = layers.Resizing(*resize_shape)(x)

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
            activation=activation,
            **kwargs,
        )

    # Build the flatten and dense output layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=out_shape[0] * out_shape[1], activation="linear")(x)

    outputs = layers.Reshape(out_shape)(x)

    return tf.keras.Model(inputs, outputs, name="CNN_resized")


def UNet_base(
    in_shape,
    out_shape,
    encoder_blocks=4,
    max_filters=512,
    dropout=0.3,
    regularization=None,
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

    return tf.keras.Model(inputs, outputs, name=UNet_base)
