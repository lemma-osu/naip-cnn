"""
This module contains functions for sampling data from local rasters and Earth Engine 
images.
"""

from __future__ import annotations

import math

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed

from naip_cnn.config import BANDS


def _create_point_grid(*, bounds, spacing: float, crs: str) -> gpd.GeoDataFrame:
    """Create a regularly spaced grid covering bounds.

    Parameters
    ----------
    bounds : tuple
        The rectangular area to cover (xmin, ymin, xmax, ymax).
    spacing : float
        The spacing between points in the grid, in the units of the bounds and CRS.
    crs : str
        The CRS of the bounds.

    Returns
    -------
    gpd.GeoDataFrame
        A dataframe of points covering the bounds in the specified projection.
    """
    xmin, ymin, xmax, ymax = bounds
    x = np.arange(xmin + spacing, xmax - spacing, spacing)
    y = np.arange(ymin + spacing, ymax - spacing, spacing)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.ravel(), yy.ravel())).T
    return gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(points[:, 0], points[:, 1]), crs=crs
    )


def extract_raster_at_points(
    raster: rasterio.io.DatasetReader, *, points: gpd.GeoDataFrame, band: int = 1
) -> gpd.GeoDataFrame:
    """Extract pixel values from a raster at a set of points.

    Parameters
    ----------
    raster : rasterio.io.DatasetReader
        An open raster to sample from.
    points : gpd.GeoDataFrame
        A collection of point geometries to sample values at.
    band : int
        The raster band index to sample from.

    Returns
    -------
    gpd.GeoDataFrame
        The original points dataframe with a column added for extracted values. Column
        names are parsed from the raster band descriptions.

    Raises
    ------
    ValueError
        If the raster CRS does not match the points CRS.
    """
    if raster.crs != points.crs:
        msg = f"The raster CRS ({raster.crs}) and point CRS ({points.crs}) must match."
        raise ValueError(msg)

    data = raster.read(band)
    band_name = raster.descriptions[band - 1]
    band_name = band_name if band_name is not None else f"b{band}"

    points["idx"] = list(zip(*raster.index(points.geometry.x, points.geometry.y)))
    points[band_name] = points.idx.apply(lambda idx: data[idx])

    return points.drop(columns=["idx"])


def generate_spaced_samples(
    raster: rasterio.io.DatasetReader,
    *,
    min_spacing: float,
    n: int | None = None,
    band: int = 1,
    seed: int = 0,
) -> gpd.GeoDataFrame:
    """Generate a systematic random sample of points at a minimum spacing over a raster.


    Parameters
    ----------
    raster : rasterio.io.DatasetReader
        An open raster to sample from.
    n : int | None
        The number of points to sample. If None, all unmasked points are returned.
    min_spacing : float
        The minimum spacing between points.
    band : int
        The raster band index to sample from.
    seed : int
        The random seed to use when sampling, for reproducibility.

    Returns
    -------
    gpd.GeoDataFrame
        A dataframe of points from unmasked areas of the raster, in the raster CRS.

    Raises
    ------
    ValueError
        If there are < n unmasked pixels at the requested spacing.
    """
    grid = _create_point_grid(bounds=raster.bounds, spacing=min_spacing, crs=raster.crs)
    samples = extract_raster_at_points(raster=raster, points=grid, band=band).dropna()

    if n is None:
        return samples

    if n > len(samples):
        raise ValueError(
            f"`n` ({n}) must be less than or equal to the number of unmasked pixels"
            f" ({len(samples)}). Try reducing `n` or `min_spacing`."
        )

    return samples.sample(n=n, random_state=seed)[["geometry"]].reset_index(drop=True)


def _create_footprint_at_xy(
    xy: tuple[float, float], *, dims: tuple[int, int], proj: ee.Projection
) -> ee.Geometry:
    """Create a rectangular footprint centered on an xy coordinate."""
    x, y = xy
    xmin = x - dims[0] // 2
    ymin = y - dims[1] // 2
    xmax = x + dims[0] // 2
    ymax = y + dims[1] // 2
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj=proj, geodesic=False)


def _extract_values_at_footprint(
    *, img: ee.Image, geom: ee.Geometry, proj: ee.Projection
) -> ee.Dictionary:
    """Extract a footprint of pixel values from an image over a geometry."""
    return img.reduceRegion(
        reducer=ee.Reducer.toList(), geometry=geom, scale=1, crs=proj
    )


def extract_footprints_from_dataframe(
    *,
    df: gpd.GeoDataFrame,
    img: ee.Image,
    dims: tuple[int, int],
    proj: ee.Projection,
    chunk_size: int = 5_000,
) -> pd.DataFrame:
    """Extract a dataframe of pixel values from an image over a dataframe of geometries.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of Point geometries to sample values around.
    img : ee.Image
        The image to sample from.
    dims : tuple[int, int]
        The dimensions of the footprint to sample at each geometry, in the proj units.
    proj : ee.Projection
        The projection to sample in.
    chunk_size : int
        The number of geometries to sample at a time. This is used to avoid exceeding
        Earth Engine limits. Because samples are downloaded in parallel, experimenting
        with this value may improve performance.

    Returns
    -------
    pd.DataFrame
        A dataframe of pixel values from the image. Each row is a point from the
        original dataframe, with one column added per image band. Pixel values are
        stored as flat lists of values.
    """
    n_chunks = math.ceil(len(df) / chunk_size)

    def _extract_footprints_from_chunk(chunk: pd.DataFrame) -> gpd.GeoDataFrame:
        features = []
        for _, f in chunk.iterrows():
            footprint = _create_footprint_at_xy(
                xy=(f.geometry.x, f.geometry.y), dims=dims, proj=proj
            )
            values = _extract_values_at_footprint(img=img, geom=footprint, proj=proj)
            features.append(ee.Feature(None, values.set("id", f.id)))

        features = ee.FeatureCollection(features)
        info = features.getInfo()
        return gpd.GeoDataFrame.from_features(info["features"])

    chunks = (
        df.iloc[chunk_size * chunk : chunk_size * (chunk + 1)]
        for chunk in range(n_chunks)
    )
    gdfs = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_extract_footprints_from_chunk)(chunk) for chunk in chunks
    )

    return pd.concat(gdfs).drop(columns="geometry").reset_index(drop=True)


def parse_pixel_array(
    footprint: pd.Series,
    *,
    shape: tuple[int] = (30, 30),
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Parse a footprint of pixel values into a numpy array"""
    features = np.empty((*shape, len(BANDS)), dtype=dtype)

    for i, band in enumerate(BANDS):
        values = footprint[band]
        features[..., i] = np.array(values).reshape(shape)

    return features
