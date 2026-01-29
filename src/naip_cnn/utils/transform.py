import math

import ee
import numpy as np


def compute_snapped_origin(
    region: ee.Geometry,
    snap_size: int,
    proj: ee.Projection,
    max_error: float = 1.0,
) -> tuple[float, float]:
    """
    Compute an origin point (top-left corner) that encompasses the given region
    and is snapped to the specified pixel size.
    """
    bbox = region.bounds(max_error, proj=proj)
    coords = np.asarray(bbox.coordinates().get(0).getInfo())
    minx = coords[:, 0].min()
    maxy = coords[:, 1].max()
    origin_x = np.floor(minx / snap_size) * snap_size
    origin_y = np.ceil(maxy / snap_size) * snap_size
    return origin_x, origin_y


def compute_dimensions(
    region: ee.Geometry,
    origin: tuple[float, float],
    scale: float,
    snap_size: int,
    proj: ee.Projection,
    max_error: float = 1.0,
) -> tuple[int, int]:
    """
    Compute the dimensions (width, height) in pixels required to cover the given
    region from the specified origin at the given scale.
    """
    bbox = region.bounds(max_error, proj=proj)
    coords = np.asarray(bbox.coordinates().get(0).getInfo())
    maxx = coords[:, 0].max()
    miny = coords[:, 1].min()

    width = int(math.ceil((maxx - origin[0]) / scale))
    height = int(math.ceil((origin[1] - miny) / scale))

    # Snap dimensions up to the nearest multiple of snap_size
    width = int(math.ceil(width / snap_size) * snap_size)
    height = int(math.ceil(height / snap_size) * snap_size)

    return width, height


def compose_transform(
    base_transform: list[float],
    origin: tuple[float, float],
    scale: float,
) -> list[float]:
    """
    Compose an affine transform given a base transform, origin, and scale.
    """
    return [
        math.copysign(scale, base_transform[0]),
        base_transform[1],
        origin[0],
        base_transform[3],
        math.copysign(scale, base_transform[4]),
        origin[1],
    ]
