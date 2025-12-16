from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import rasterio

from naip_cnn.config import MODEL_DIR, PRED_DIR
from naip_cnn.data import NAIPDatasetWrapper, NAIPTFRecord


@dataclass
class ModelRun:
    """A model run with associated model, dataset, and parameters."""

    model: keras.Model
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
            self.model_params["regularization"] = keras.regularizers.serialize(
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

                # TODO: Fix. Masks always seem to export with an extra pixel. Since
                # we're currently only masking with admin boundaries, this is
                # non-critical.
                mask = mask[: pred.shape[0], : pred.shape[1]]
                # assert mask.shape == pred.shape, (
                #     f"Mask shape {mask.shape} does not match prediction shape"
                #     f" {pred.shape}."
                # )
                pred[mask == 0] = 255

        # Optionally write out to geotiff
        if filename:
            pred_path = PRED_DIR / f"{filename}.tif"
            pred_profile = copy.deepcopy(tfrecord.profile)

            # Match the sign of the pixel scale from the mixer
            pred_profile["transform"][0] = math.copysign(
                self.dataset.lidar_res, pred_profile["transform"][0]
            )
            pred_profile["transform"][4] = math.copysign(
                self.dataset.lidar_res, pred_profile["transform"][4]
            )

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

    return keras.models.Sequential(
        [
            keras.layers.Conv3D(
                filters=filter_no,
                kernel_size=kernel_size,
                input_shape=(*shape, 1),
                padding="valid",
                activation="relu",
                use_bias=True,
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(Dense1_no, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(Dense2_no, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(units=1, activation="linear"),
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
        x = keras.layers.Conv2D(
            n_filters,
            filter_size,
            padding="same",
            activation=activation,
            kernel_regularizer=regularization,
            **kwargs,
        )(x)

    p = keras.layers.MaxPool2D(pool_size)(x)

    if dropout is not None:
        p = keras.layers.Dropout(dropout)(p)

    return x, p


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
    inputs = x = keras.layers.Input(shape=shape)

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
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=out_shape[0] * out_shape[1], activation="linear")(x)

    outputs = keras.layers.Reshape(out_shape)(x)

    return keras.Model(inputs, outputs, name="CNN_base")


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
    inputs = x = keras.layers.Input(shape=shape)

    x = keras.layers.Resizing(*resize_shape)(x)

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
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=out_shape[0] * out_shape[1], activation="linear")(x)

    outputs = keras.layers.Reshape(out_shape)(x)

    return keras.Model(inputs, outputs, name="CNN_resized")
