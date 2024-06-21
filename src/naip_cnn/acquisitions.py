from __future__ import annotations

import ee

from naip_cnn import config


class Acquisition:
    """A LiDAR aquisition with associated NAIP imagery in Earth Engine."""

    def __init__(
        self, name: str, start_date: str, end_date: str, collection: str = None
    ):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date

        asset_path = f"{collection}/{name}" if collection is not None else name
        self.lidar_asset = f"{config.EE_ASSET_DIR}/{asset_path}"

    def __repr__(self) -> str:
        return f"<Acquisition name={self.name}>"

    @property
    def proj(self) -> ee.Projection:
        return ee.Projection(config.CRS)

    @property
    def geometry(self) -> ee.Geometry:
        return ee.Image(self.lidar_asset).geometry()

    @property
    def mask(self) -> ee.Image:
        """A mask of fast loss over the acqusition period."""
        return (
            ee.ImageCollection("USFS/GTAC/LCMS/v2022-8")
            .filter(ee.Filter.eq("study_area", "CONUS"))
            .filterDate(self.start_date, self.end_date)
            .select("Change")
            .map(lambda img: img.eq(3))
            .max()
            .eq(0)
            .reproject(self.proj.atScale(30))
        )

    def load_naip(self, scale: float = config.NAIP_RES) -> ee.Image:
        return (
            ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.geometry.bounds())
            .mosaic()
            .updateMask(self.mask)
            .reproject(self.proj.atScale(scale))
        )

    def load_lidar(self, scale: float = config.LIDAR_RES) -> ee.Image:
        return (
            ee.Image(self.lidar_asset)
            .updateMask(self.mask)
            .reproject(self.proj.atScale(scale))
        )

    @staticmethod
    def from_name(name) -> Acquisition:
        """Return an acquisition from its name."""
        return eval(name)


# OR NAIP years: 2004 (RGB), 2005 (RGB), 2009, 2011, 2012, 2014, 2016, 2020, 2022

HJA2008_5m = Acquisition(
    name="HJA2008_5m", start_date="2008-01-01", end_date="2009-12-31"
)
HJA2011_5m = Acquisition(
    name="HJA2011_5m", start_date="2011-01-01", end_date="2011-12-31"
)
HJA2016_5m = Acquisition(
    name="HJA2016_5m", start_date="2016-01-01", end_date="2016-12-31"
)
HJA2020_5m = Acquisition(
    name="HJA2020_5m", start_date="2020-01-01", end_date="2020-12-31"
)

HJA2020_1m = Acquisition(
    name="HJA2020_1m", start_date="2020-01-01", end_date="2020-12-31"
)

MAL2007 = Acquisition(
    name="MAL2007",
    start_date="2007-01-01",
    end_date="2009-12-31",
    collection="malheur_lidar",
)
MAL2008_CampCreek = Acquisition(
    name="MAL2008_CampCreek",
    start_date="2008-01-01",
    end_date="2009-12-31",
    collection="malheur_lidar",
)
MAL2008_2009_MalheurRiver = Acquisition(
    name="MAL2008_2009_MalheurRiver",
    start_date="2008-01-01",
    end_date="2009-12-31",
    collection="malheur_lidar",
)
MAL2010 = Acquisition(
    name="MAL2010",
    start_date="2010-01-01",
    end_date="2011-12-31",
    collection="malheur_lidar",
)
MAL2014 = Acquisition(
    name="MAL2014",
    start_date="2014-01-01",
    end_date="2014-12-31",
    collection="malheur_lidar",
)
MAL2016_CanyonCreek = Acquisition(
    name="MAL2016_CanyonCreek",
    start_date="2016-01-01",
    end_date="2016-12-31",
    collection="malheur_lidar",
)
MAL2017_Crow = Acquisition(
    name="MAL2017_Crow",
    start_date="2016-01-01",
    end_date="2017-12-31",
    collection="malheur_lidar",
)
MAL2017_JohnDay = Acquisition(
    name="MAL2017_JohnDay",
    start_date="2016-01-01",
    end_date="2017-12-31",
    collection="malheur_lidar",
)
MAL2018_Aldrich_UpperBear = Acquisition(
    name="MAL2018_Aldrich_UpperBear",
    start_date="2018-01-01",
    end_date="2020-12-31",
    collection="malheur_lidar",
)
MAL2018_Rattlesnake = Acquisition(
    name="MAL2018_Rattlesnake",
    start_date="2018-01-01",
    end_date="2020-12-31",
    collection="malheur_lidar",
)
MAL2019 = Acquisition(
    name="MAL2019",
    start_date="2019-01-01",
    end_date="2020-12-31",
    collection="malheur_lidar",
)
MAL2020_UpperJohnDay = Acquisition(
    name="MAL2020_UpperJohnDay",
    start_date="2020-01-01",
    end_date="2020-12-31",
    collection="malheur_lidar",
)
