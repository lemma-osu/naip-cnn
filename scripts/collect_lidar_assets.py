"""
This script takes LiDAR Image assets previously uploaded to Earth Engine and collects
them into an ImageCollection while setting band names.
"""

import ee

ee.Authenticate(auth_mode="notebook")
ee.Initialize()

COLLECTION_PATH = "projects/ee-maptheforests/assets/malheur_lidar"
BANDS = ["cover", "rh25", "rh50", "rh95", "rh100"]
IMAGES = [
    "projects/ee-maptheforests/assets/MAL2007",
    "projects/ee-maptheforests/assets/MAL2008_CampCreek",
    "projects/ee-maptheforests/assets/MAL2008_2009_MalheurRiver",
    "projects/ee-maptheforests/assets/MAL2010",
    "projects/ee-maptheforests/assets/MAL2014",
    "projects/ee-maptheforests/assets/MAL2016_CanyonCreek",
    "projects/ee-maptheforests/assets/MAL2017_Crow",
    "projects/ee-maptheforests/assets/MAL2017_JohnDay",
    "projects/ee-maptheforests/assets/MAL2018_Aldrich_UpperBear",
    "projects/ee-maptheforests/assets/MAL2018_Rattlesnake",
    "projects/ee-maptheforests/assets/MAL2019",
    "projects/ee-maptheforests/assets/MAL2020_UpperJohnDay",
]


def create_collection(path):
    """Create an empty ImageCollection at the given path."""
    try:
        ee.data.getAsset(path)
    except ee.EEException:
        ee.data.createAsset(
            value={"type": "ImageCollection"},
            opt_path=path,
        )


def export_lidar_to_collection(image_path, collection_path, band_names):
    """Export a LiDAR image to an ImageCollection."""
    name = image_path.split("/")[3]
    asset_id = f"{collection_path}/{name}"

    try:
        # Early return if the asset already exists
        return ee.data.getAsset(asset_id)
    except ee.EEException:
        pass

    img = ee.Image(image_path).rename(band_names)
    mask = img.select("cover").mask()

    # Height bands are masked below 1.5m, so fill those values with 0
    img = img.unmask(0).updateMask(mask)

    proj = img.projection().getInfo()
    task = ee.batch.Export.image.toAsset(
        image=img,
        description=name,
        assetId=f"{COLLECTION_PATH}/{name}",
        crs=proj["wkt"],
        crsTransform=proj["transform"],
    )
    task.start()

    return task


if __name__ == "__main__":
    create_collection(COLLECTION_PATH)
    for image_path in IMAGES:
        task = export_lidar_to_collection(image_path, COLLECTION_PATH, BANDS)
        if isinstance(task, ee.batch.Task):
            print(f"Exporting {image_path}...")
        else:
            print(f"Asset {image_path} already exists.")
