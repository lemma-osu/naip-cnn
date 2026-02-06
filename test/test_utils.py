import ee

from naip_cnn.config import CRS
from naip_cnn.utils.transform import (
    compose_transform,
    compute_dimensions,
    compute_snapped_origin,
)


def test_compute_snapped_origin():
    # Use an arbitrary rectangle as a pseudo regression test
    geometry = ee.Geometry.Rectangle([-119.497, 44.749, -118.299, 43.924])
    assert compute_snapped_origin(
        geometry,
        snap_size=30,
        proj=ee.Projection(CRS),
    ) == (639780, 1188420)


def test_compose_transform():
    base_transform = [30, 0, 0, 0, -30, 0]
    origin = (603930, 1272330)
    scale = 1.0
    expected_transform = [1.0, 0, 603930, 0, -1.0, 1272330]
    assert compose_transform(base_transform, origin, scale) == expected_transform


def test_compute_dimensions():
    # Use an arbitrary rectangle as a pseudo regression test
    geometry = ee.Geometry.Rectangle([-119.497, 44.749, -118.299, 43.924])
    origin = (603930, 1272330)
    scale = 1.0
    expected_dimensions = (132600, 176970)
    assert (
        compute_dimensions(
            geometry,
            origin,
            scale,
            snap_size=30,
            proj=ee.Projection(CRS),
        )
        == expected_dimensions
    )
