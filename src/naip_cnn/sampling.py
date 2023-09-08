"""
This module contains functions for sampling data from Earth Engine images.
"""

from __future__ import annotations

import json

import ee
import numpy as np
import pandas as pd

from naip_cnn.config import BANDS


def point_to_footprint(
    point: ee.Feature, *, dims: tuple[int, int], proj: ee.Projection
) -> ee.Geometry:
    """Create a rectangular footprint centered on a Point feature."""
    geom = point.geometry()
    x = geom.transform(proj).coordinates().getNumber(0)
    y = geom.transform(proj).coordinates().getNumber(1)

    xmin = x.subtract(dims[0] // 2)
    ymin = y.subtract(dims[1] // 2)
    xmax = x.add(dims[0] // 2)
    ymax = y.add(dims[1] // 2)

    footprint = ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax], proj=proj, geodesic=False
    )
    # TODO: add id to footprint
    return ee.Feature(footprint, {"height": dims[0], "width": dims[1]})


def extract_values_at_footprint(
    footprint: ee.Feature,
    *,
    img: ee.Image,
    proj: ee.Projection,
    scale: int = 1,
    drop_if_null: bool = True,
) -> ee.Feature:
    """Extract a footprint of pixel values from an image over a geometry."""
    values = img.reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=footprint.geometry(),
        scale=scale,
        crs=proj,
    )

    if drop_if_null:
        # If masked values are sampled, there will be fewer pixels than expected
        footprint_area = footprint.getNumber("height").multiply(
            footprint.getNumber("width")
        )
        n_pixels = footprint_area.divide(scale * scale)
        has_nulls = (
            values.values().map(lambda v: ee.List(v).size().lt(n_pixels)).contains(1)
        )
        return ee.Algorithms.If(has_nulls, None, footprint.set(values))

    return footprint.set(values)


def parse_pixel_array(
    footprint: pd.Series,
    *,
    shape: tuple[int] = (30, 30),
    dtype: np.dtype = np.uint8,
    col: tuple[str] | str = BANDS,
) -> np.ndarray:
    """Parse a footprint of 1D pixel values into an ND numpy array"""

    def parse_array(array: str) -> np.ndarray:
        """Parse a 1D string array to a 2D array."""
        values = json.loads(array)
        return np.array(values).reshape(shape)

    if isinstance(col, str):
        return parse_array(footprint[col])

    features = np.empty((*shape, len(col)), dtype=dtype)
    for i, band in enumerate(col):
        features[..., i] = parse_array(footprint[band])

    return features
