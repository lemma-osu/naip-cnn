"""
This script converts single-band LiDAR rasters to multiband rasters which can be 
ingested into Earth Engine prior to sampling. After running this script, the generated 
rasters should be uploaded manually to Earth Engine, and 
`scripts/collect_lidar_assets.js` should be run in the Code Editor.
"""

from pathlib import Path

import rasterio

from naip_cnn.config import CRS

LIDAR_DIR = Path("./data/lidar/raw/")
OUT_DIR = Path("./data/lidar/multiband/")
DATASETS = {
    "MAL2007": {
        "cover": "MAL2007_FIRST_RETURNS_1st_cover_above1p5_30METERS.tif",
        "p25": "MAL2007_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2007_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2007_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2007_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2008_CampCreek": {
        "cover": "MAL2008_CampCreek_FIRST_RETURNS_1st_cover_above1p5_30METERS.tif",
        "p25": "MAL2008_CampCreek_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2008_CampCreek_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2008_CampCreek_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2008_CampCreek_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2008_2009_MalheurRiver": {
        "cover": (
            "MAL2008_2009_MalheurRiver_FIRST_RETURNS_1st_cover_above1p5_30METERS.tif"
        ),
        "p25": "MAL2008_2009_MalheurRiver_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2008_2009_MalheurRiver_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2008_2009_MalheurRiver_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2008_2009_MalheurRiver_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2010": {
        "cover": "MAL2010_FIRST_RETURNS_1st_cover_above1p5_30METERS.tif",
        "p25": "MAL2010_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2010_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2010_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2010_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2014": {
        "cover": "MAL2014_FIRST_RETURNS_1st_cover_above1p5_30METERS.tif",
        "p25": "MAL2014_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2014_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2014_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2014_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2016_CanyonCreek": {
        # Note this uses all_cover instead of 1st_cover
        "cover": "MAL2016_CanyonCreek_FIRST_RETURNS_all_cover_above1p5_30METERS.tif",
        "p25": "MAL2016_CanyonCreek_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2016_CanyonCreek_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2016_CanyonCreek_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2016_CanyonCreek_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2017_Crow": {
        "cover": "MAL2017_Crow_FIRST_RETURNS_1st_cover_above1p5_30METERS.tif",
        "p25": "MAL2017_Crow_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2017_Crow_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2017_Crow_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2017_Crow_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2017_JohnDay": {
        "cover": "MAL2017_JohnDay_FIRST_RETURNS_1st_cover_1p5plus_30METERS.tif",
        "p25": "MAL2017_JohnDay_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2017_JohnDay_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2017_JohnDay_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2017_JohnDay_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2018_Aldrich_UpperBear": {
        # Note this uses all_cover instead of 1st_cover
        "cover": (
            "MAL2018_Aldrich_UpperBear_FIRST_RETURNS_all_cover_above1p5_30METERS.tif"
        ),
        "p25": "MAL2018_Aldrich_UpperBear_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2018_Aldrich_UpperBear_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2018_Aldrich_UpperBear_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2018_Aldrich_UpperBear_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2018_Rattlesnake": {
        "cover": "MAL2018_Rattlesnake_FIRST_RETURNS_1st_cover_above1p5_30METERS.tif",
        "p25": "MAL2018_Rattlesnake_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2018_Rattlesnake_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2018_Rattlesnake_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2018_Rattlesnake_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2019": {
        # Note this uses all_1st_cover instead of 1st_cover
        "cover": "MAL2019_FIRST_RETURNS_all_1st_cover_1p5plus_30METERS.tif",
        "p25": "MAL2019_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2019_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2019_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2019_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
    "MAL2020_UpperJohnDay": {
        "cover": "MAL2020_UpperJohnDay_FIRST_RETURNS_1st_cover_1p5plus_30METERS.tif",
        "p25": "MAL2020_UpperJohnDay_FIRST_RETURNS_elev_P25_1p5plus_30METERS.tif",
        "p50": "MAL2020_UpperJohnDay_FIRST_RETURNS_elev_ave_1p5plus_30METERS.tif",
        "p95": "MAL2020_UpperJohnDay_FIRST_RETURNS_elev_P95_1p5plus_30METERS.tif",
        "p100": "MAL2020_UpperJohnDay_FIRST_RETURNS_elev_max_1p5plus_30METERS.tif",
    },
}


if __name__ == "__main__":
    for name, dataset in DATASETS.items():
        # Use the first raster as a template for the multiband raster
        ref_path = LIDAR_DIR / list(dataset.values())[0]
        with rasterio.open(ref_path) as ref:
            profile = ref.profile

        profile.update(
            count=len(dataset),
            crs=CRS,
            compress="DEFLATE",
        )

        dataset_path = OUT_DIR / f"{name}.tif"
        with rasterio.open(dataset_path, "w", **profile) as dst:
            for band_name, filename in dataset.items():
                path = LIDAR_DIR / filename
                band_idx = list(dataset.keys()).index(band_name) + 1
                with rasterio.open(path) as src:
                    dst.write(src.read(1), band_idx)
                    dst.set_band_description(band_idx, band_name)
