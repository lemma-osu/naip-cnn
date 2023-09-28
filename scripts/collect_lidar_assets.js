/*
Band names are lost when images are uploaded to Earth Engine, so this script takes 
uploaded LiDAR images and re-exports them to an ImageCollection while setting the band
names. This script should be run in the Code Editor.
*/

var COLLECTION_PATH = "projects/ee-maptheforests/assets/malheur_lidar";
var BANDS = ["cover", "rh25", "rh50", "rh95", "rh100"];
var IMAGES = [
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
];


function export_images_to_collection(image_paths, band_names) {
  for (var i=0; i<image_paths.length; i++) {
    var img = ee.Image(image_paths[i]).rename(band_names);
    var mask = img.select("cover").mask();
    
    // Height bands are masked below 1.5m, so fill those values with 0
    img = img.unmask(0).updateMask(mask)
    
    var name = image_paths[i].split("/")[3];
    var proj = img.projection().getInfo();
    
    Export.image.toAsset({
      image: img,
      description: name,
      assetId: COLLECTION_PATH + "/" + name,
      crs: proj.wkt,
      crsTransform: proj.transform,
    })
  }
}

// Create an empty collection to store images and start exports.
ee.data.createAsset(
  {type: "ImageCollection"}, 
  COLLECTION_PATH, 
  false,
  null,
  function() {
    export_images_to_collection(IMAGES, BANDS);
  }
)