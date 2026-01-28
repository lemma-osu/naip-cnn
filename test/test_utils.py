import ee

from naip_cnn.config import CRS
from naip_cnn.utils.transform import compose_transform, compute_snapped_origin


def test_compute_snapped_origin():
    geometry = ee.FeatureCollection(
        "projects/ee-maptheforests/assets/USFS_RD_CNN"
    ).geometry()
    assert compute_snapped_origin(
        geometry,
        snap_size=30,
        proj=ee.Projection(CRS),
    ) == (603930, 1272330)


def test_compose_transform():
    base_transform = [30, 0, 0, 0, -30, 0]
    origin = (603930, 1272330)
    scale = 1.0
    expected_transform = [1.0, 0, 603930, 0, -1.0, 1272330]
    assert compose_transform(base_transform, origin, scale) == expected_transform
