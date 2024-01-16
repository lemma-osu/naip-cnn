from __future__ import annotations

import numpy as np
import rioxarray
import tensorflow as t
import xarray as xr
from tiler import Merger, Tiler


def predict_image(
    *,
    img_path: str,
    model: t.keras.Model,
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

    for batch_id, batch in input_tiler(img, batch_size=batch_size, progress_bar=False):
        pred = model.predict(batch, verbose=0)

        merger.add_batch(batch_id, batch_size=batch_size, data=pred)

    pred = merger.merge()

    # Use the resampled coordinates of the input array to georeference the output array
    out_da = in_da.rio.reproject(dst_crs=in_da.rio.crs, shape=out_shape[:2])
    # Crop to the predicted output, as edge pixels may have been lost
    out_da = out_da.isel(x=slice(0, pred.shape[1]), y=slice(0, pred.shape[0]))

    out_da["pred"] = xr.DataArray(pred.squeeze(), dims=("y", "x"))

    return out_da.pred
