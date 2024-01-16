"""
This script exports NAIP GeoTIFFs for a set of test areas used to evaluate models.
"""

from io import BytesIO
from pathlib import Path

import ee
import rasterio
from tqdm.auto import tqdm

from naip_cnn.config import CRS

ee.Initialize()


# Test areas with a range of forest structures. These are all outside of the
# training acqusitions, allowing for independent assessments.
TEST_AOIS = {
    # NW of the Malheur. Mixed density with lots of open areas and regen.
    "LADD_CANYON_RD": ee.Geometry.Point([-118.24133223116047, 45.03989434530832]),
    # NW of Waldo Lake. Mixed age managed stands.
    "FURNISH_CRK": ee.Geometry.Point([-122.15174567491412, 43.7998276876186]),
    # In the Elliott. Mixed age managed stands.
    "WHISKEY_CAMP_CRK": ee.Geometry.Point([-123.6530102470513, 43.46198195227025]),
    # W of the Malheur. Mixed density with standing snags.
    "MILK_SPRING": ee.Geometry.Point([-120.15718198236159, 44.463409883421264]),
}


def compute_naip_chip(
    center: ee.Geometry, year: int = 2020, buffer: int = 1200, scale: float = 1.0
) -> bytes:
    """Compute pixels for a NAIP chip centered on the given point.
    Return encoded GeoTIFF bytes."""
    aoi = center.buffer(buffer).bounds()

    naip = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterDate(str(year), str(year + 1))
        .filterBounds(aoi)
        .mosaic()
        .reproject(CRS)
        .clip(aoi)
    )

    info = naip.getInfo()
    dims = info["bands"][0]["dimensions"]
    origin = info["bands"][0]["origin"]
    affine_transform = {
        "scaleX": scale,
        "scaleY": -scale,
        "shearX": 0,
        "shearY": 0,
        "translateX": origin[0],
        "translateY": origin[1] + dims[1] * scale,
    }

    return ee.data.computePixels(
        dict(
            expression=naip,
            fileFormat="GeoTIFF",
            grid={
                "crsWkt": CRS,
                "dimensions": {"width": dims[0], "height": dims[1]},
                "affineTransform": affine_transform,
            },
        )
    )


if __name__ == "__main__":
    out_dir = Path("./data/naip/")
    out_dir.mkdir(exist_ok=True)
    evaluation_year = 2020

    for name, aoi in tqdm(TEST_AOIS.items()):
        img = compute_naip_chip(aoi, year=evaluation_year)

        with rasterio.open(BytesIO(img)) as src:
            array = src.read()
            out_path = f"./data/naip/{name}_{evaluation_year}.tif"
            meta = src.meta.copy()
            del meta["nodata"]

        with rasterio.open(out_path, "w", compress="LZW", **meta) as dst:
            dst.write(array)

            for i, band in enumerate(["R", "G", "B", "N"], start=1):
                dst.set_band_description(i, band)
