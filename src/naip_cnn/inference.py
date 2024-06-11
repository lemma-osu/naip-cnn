from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import ee
import numpy as np
import rasterio
import rioxarray
import tensorflow as tf
import xarray as xr
from numpy.typing import NDArray
from tiler import Merger, Tiler

from naip_cnn.config import BANDS, CRS, NAIP_RES, PRED_DIR, TFRECORD_DIR
from naip_cnn.models import ModelRun
from naip_cnn.utils.parsing import float_to_str


def predict_image(
    *,
    img_path: str,
    model: tf.keras.Model,
    in_tile_shape: tuple[int, int, int],
    out_tile_shape: tuple[int, ...],
    in_res: float,
    out_res: float,
    tile_overlap=0.0,
    batch_size=8,
) -> xr.DataArray:
    """Generate a prediction from a GeoTIFF image using a model.

    This handles tiling, batching, resampling, predicting, and merging.

    Parameters
    ----------
    img_path : str
        The path to the image to predict. Must be openable with `rasterio`.
    model : tf.keras.Model
        The model to use for prediction.
    in_tile_shape : tuple[int, int, int]
        The shape of the input tiles. Should be (height, width, channels).
    out_tile_shape : tuple[int, ...]
        The shape of the output tiles. Should be (height, width, <channels?>). The
        channel dimension is optional, depending on the model.
    in_res : float
        The resolution of the input image in its native projection.
    out_res : float
        The resolution of the output image in its native projection.
    tile_overlap : float, optional
        The amount of overlap between tiles, as a fraction of the tile size. Note that
        overlap is not supported when the input and output resolutions are different.
        Defaults to 0.0.
    batch_size : int, optional
        The batch size to use for prediction. Defaults to 8. Tune this to optimize
        performance and memory usage.

    Returns
    -------
    xr.DataArray
        The predicted image, with the same CRS and coordinates as the input image.
    """
    in_da = rioxarray.open_rasterio(img_path)
    in_da = in_da.rio.reproject(dst_crs=in_da.rio.crs, resolution=in_res)

    img = in_da.values.astype(np.float32) / 255.0
    img = np.moveaxis(img, 0, -1)

    resample_factor = out_res / in_res
    in_shape = img.shape
    out_shape = (
        int(img.shape[0] // resample_factor),
        int(img.shape[1] // resample_factor),
    )
    out_channel_dim = None

    if tile_overlap > 0.0 and resample_factor != 1.0:
        raise ValueError("Overlap is not supported when resampling!")

    # The model may or may not predict a channel dimension, so set accordingly
    if len(out_tile_shape) == 3:
        out_shape = (*out_shape, out_tile_shape[-1])
        out_channel_dim = -1

    input_tiler = Tiler(
        data_shape=in_shape,
        tile_shape=in_tile_shape,
        channel_dimension=-1,
        overlap=tile_overlap,
        mode="drop",
    )
    output_tiler = Tiler(
        data_shape=out_shape,
        tile_shape=out_tile_shape,
        channel_dimension=out_channel_dim,
        overlap=tile_overlap,
        mode="drop",
    )
    merger = Merger(output_tiler)

    for batch_id, batch in input_tiler(img, batch_size=batch_size, progress_bar=True):
        pred = model.predict(batch, verbose=0)

        merger.add_batch(batch_id, batch_size=batch_size, data=pred)

    pred = merger.merge()

    # Use the resampled coordinates of the input array to georeference the output array
    out_da = in_da.rio.reproject(dst_crs=in_da.rio.crs, shape=out_shape[:2])
    # Crop to the predicted output, as edge pixels may have been lost
    out_da = out_da.isel(x=slice(0, pred.shape[1]), y=slice(0, pred.shape[0]))

    out_da["pred"] = xr.DataArray(pred.squeeze(), dims=("y", "x"))

    return out_da.pred


@dataclass
class NAIPTFRecord:
    """A TFRecord dataset of NAIP tiles for inference.

    This class is used both for exporting NAIP imagery to TFRecords and for
    loading, parsing, and predicting on those TFRecords.
    """

    id: str
    footprint: tuple[float, float]
    year: int
    bounds: ee.Geometry = None
    res: float = NAIP_RES

    def __post_init__(self) -> None:
        footprint_str = f"{self.footprint[0]}x{self.footprint[1]}"
        res_str = float_to_str(self.res)

        self.name = "-".join(
            [
                self.id,
                str(self.year),
                res_str,
                footprint_str,
            ]
        )

        self.mixer_path = TFRECORD_DIR / f"{self.name}-mixer.json"

    @property
    def naip_shape(self):
        return int(self.footprint[0] // self.res), int(self.footprint[1] // self.res)

    def load_naip(self) -> ee.Image:
        if self.bounds is None:
            raise ValueError("Bounds must be set before loading NAIP imagery.")

        return (
            ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(self.bounds)
            .filterDate(str(self.year), str(self.year + 1))
            .mosaic()
            .select(BANDS)
            .reproject(CRS)
            .uint8()
        )

    def export_to_drive(self, **kwargs):
        """Export the NAIP image to Google Drive."""
        task = ee.batch.Export.image.toDrive(
            image=self.load_naip(),
            description=self.name,
            region=self.bounds,
            scale=self.res,
            fileFormat="TFRecord",
            maxPixels=1e13,
            formatOptions={
                "patchDimensions": self.naip_shape,
                "compressed": True,
                "maxFileSize": 1e9,
            },
            **kwargs,
        )
        task.start()
        return task

    def __repr__(self) -> str:
        return f"<NAIPTFRecord name={self.name}>"

    @cached_property
    def profile(self) -> dict:
        with open(self.mixer_path) as f:
            mixer = json.load(f)

        return {
            "width": mixer["patchesPerRow"],
            "height": mixer["totalPatches"] // mixer["patchesPerRow"],
            "crs": mixer["projection"]["crs"],
            "transform": mixer["projection"]["affine"]["doubleMatrix"],
        }

    def n_batches(self, batch_size: int):
        return np.ceil((self.profile["width"] * self.profile["height"]) / batch_size)

    def load_dataset(self, bands: tuple[str] = BANDS) -> tf.data.Dataset:
        """Load and parse the TFRecord dataset."""
        tfrecords = list(TFRECORD_DIR.glob(f"{self.name}*.tfrecord.gz"))

        data = tf.data.TFRecordDataset(tfrecords, compression_type="GZIP")
        return data.map(lambda x: self._parse(x, bands=bands))

    def _parse(
        self,
        serialized_example: bytes,
        bands: tuple[str],
    ) -> tf.Tensor:
        """Parse a single example from a TFRecord file containing NAIP imagery.

        Note that this currently only supports encoded byte images.
        """

        def decode_band(band: tf.Tensor) -> tf.Tensor:
            """Decode a byte string into a 2D uint8 Tensor."""
            data = tf.io.decode_raw(band, out_type=tf.uint8)
            return tf.cast(tf.reshape(data, self.naip_shape), tf.float32) / 255.0

        # uint8 data types must be parsed as strings and then decoded to uint8.
        # See https://issuetracker.google.com/issues/296941927
        features = {b: tf.io.FixedLenFeature(shape=[], dtype=tf.string) for b in bands}
        example = tf.io.parse_single_example(serialized_example, features)
        image = tf.stack([decode_band(example[b]) for b in bands], axis=-1)

        return tf.expand_dims(image, axis=-1)

    def predict(self, run: ModelRun, batch_size: int = 256, **kwargs) -> NDArray:
        """Predict a label for each pixel in the NAIP image using a given model run."""
        image = self.load_dataset(bands=run.bands).batch(batch_size)
        raw_pred = run.model.predict(image, steps=self.n_batches(batch_size), **kwargs)
        h, w = self.profile["height"], self.profile["width"]
        return (
            raw_pred.reshape(h, w, *run.dataset.lidar_shape)
            .swapaxes(1, 2)
            .reshape(h * run.dataset.lidar_shape[0], w * run.dataset.lidar_shape[1])
            .clip(min=0)
        )

    def export_prediction(self, pred: NDArray, *, run: ModelRun) -> Path:
        """Export the prediction to a GeoTIFF."""
        pred_path = PRED_DIR / f"{self.name}-{run.name}.tif"
        pred_profile = self.profile.copy()

        pred_profile["transform"][0] = run.dataset.lidar_res
        pred_profile["transform"][4] = run.dataset.lidar_res
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
