from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import ee
import h5py
import numpy as np
import spyndex
import tensorflow as tf
import tensorflow_io as tfio

from naip_cnn.acquisitions import Acquisition
from naip_cnn.augment import Augment
from naip_cnn.config import (
    BANDS,
    BASE_TRANSFORM,
    CRS,
    GRID_SNAP,
    NAIP_RES,
    TFRECORD_DIR,
    TRAIN_DIR,
)
from naip_cnn.utils.parsing import float_to_str, str_to_float
from naip_cnn.utils.transform import compose_transform, compute_snapped_origin


class _HDF5DatasetMixin:
    """A dataset of features and labels stored in an HDF5 file."""

    def __init__(self, path, feature_name: str):
        self.path = path
        self.feature_name = feature_name

    @cached_property
    def n_samples(self) -> int:
        """The total number of samples, based on the first dataset."""
        with h5py.File(self.path, "r") as f:
            first_key = list(f.keys())[0]
            return f[first_key].shape[0]

    def __len__(self) -> int:
        """Return the total number of samples, based on the first dataset."""
        return self.n_samples

    def _load(
        self,
        label: str,
        feature_preprocessor: Callable | None = None,
        label_preprocessor: Callable | None = None,
        augmenter: Augment | None = None,
    ) -> tf.data.Dataset:
        """Load a zipped dataset of features and labels from an HDF5 file."""
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist")
        features = tfio.IODataset.from_hdf5(
            self.path.as_posix(), dataset=f"/{self.feature_name}"
        )
        labels = tfio.IODataset.from_hdf5(self.path.as_posix(), dataset=f"/{label}")
        ds = tf.data.Dataset.zip((features, labels))

        if feature_preprocessor is not None or label_preprocessor is not None:
            feature_preprocessor = feature_preprocessor or (lambda x: x)
            label_preprocessor = label_preprocessor or (lambda x: x)
            ds = ds.map(
                lambda x, y: (feature_preprocessor(x), label_preprocessor(y)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if augmenter is not None:
            ds = ds.map(
                lambda x, y: augmenter(x, y), num_parallel_calls=tf.data.AUTOTUNE
            )

        return ds.apply(tf.data.experimental.assert_cardinality(self.n_samples))


class _NAIPHDF5Dataset(_HDF5DatasetMixin):
    """A dataset of NAIP images and forest attribute labels in an HDF5 file."""

    def __init__(self, path):
        return super().__init__(path=path, feature_name="image")

    def _load(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
        augmenter: Augment | None = None,
    ) -> tf.data.Dataset:
        """Load a Tensorflow Dataset of NAIP images from an HDF5 file.

        Parameters
        ----------
        label : str
            The name of the label variable.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        veg_indices : list[str]
            Vegetation indices to calculate and append to the NAIP images. Indices must
            be supported by spyndex.
        augmenter : function
            The function to use for data augmentation. Must take and return a tuple of
            (image, label) tensors.
        """

        def preprocess_naip(image: tf.Tensor) -> tf.Tensor:
            """Cast byte image to float and select/calculate bands."""
            image = tf.cast(image, tf.float32) / 255.0

            EXTRA_VI_PARAMS = {
                "EVI": {"L": 2.5, "C1": 6, "C2": 7.5, "g": 2.5},
                "SAVI": {"L": 0.5},
                "ARVI": {"gamma": 1.0},
            }

            for index in veg_indices:
                index_params = {
                    "R": image[..., 0],
                    "G": image[..., 1],
                    "B": image[..., 2],
                    "N": image[..., 3],
                }

                index_params.update(EXTRA_VI_PARAMS.get(index, {}))
                vi = spyndex.computeIndex(index, params=index_params)
                # Band ratios can produce NaNs
                vi = tf.where(tf.math.is_nan(vi), tf.zeros_like(vi), vi)

                vi = tf.expand_dims(vi, axis=-1)

                image = tf.concat([image, vi], axis=-1)

            # Select out any unwanted bands
            band_idxs = [BANDS.index(b) for b in bands]
            vi_idxs = list(range(len(BANDS), len(BANDS) + len(veg_indices)))
            keep_bands = band_idxs + vi_idxs

            return tf.gather(image, keep_bands, axis=-1)

        return super()._load(
            label=label,
            feature_preprocessor=preprocess_naip,
            augmenter=augmenter,
        )


class NAIPDatasetWrapper:
    """A wrapper around a NAIP training dataset that stores additional metadata and
    provides convenience methods for loading the dataset.

    This class is used both to define and generate sample datasets and to load
    previously generated datasets.
    """

    def __init__(
        self,
        acquisitions: list[Acquisition],
        naip_res=1.0,
        lidar_res=30.0,
        footprint: tuple[int, int] = (30, 30),
        spacing: float = None,
    ) -> None:
        """
        Parameters
        ----------
        acquisition : Acquisition
            The NAIP-LiDAR acquisition used to build the dataset.
        naip_res : float
            The NAIP resolution in meters.
        lidar_res : float
            The LiDAR resolution in meters.
        footprint : tuple[int, int]
            The size of the sampled footprints in meters.
        spacing : float
            The spacing between sampled footprints in meters. If None, defaults to
            the footprint width.
        """
        self.acquisitions = acquisitions
        self.naip_res = naip_res
        self.lidar_res = lidar_res
        self.footprint = footprint
        self.spacing = spacing if spacing is not None else footprint[0]
        self.csv_path = TRAIN_DIR / (self.name + ".csv")

        # The shape of the image and labels in pixels
        h, w = self.footprint
        self.naip_shape = int(h // naip_res), int(w // naip_res)
        self.lidar_shape = int(h // lidar_res), int(w // lidar_res)

        self._train_paths = [
            TRAIN_DIR / ("-".join([a.name, self.suffix]) + "_train.h5")
            for a in self.acquisitions
        ]
        self._val_paths = [
            TRAIN_DIR / ("-".join([a.name, self.suffix]) + "_val.h5")
            for a in self.acquisitions
        ]
        self._test_paths = [
            TRAIN_DIR / ("-".join([a.name, self.suffix]) + "_test.h5")
            for a in self.acquisitions
        ]

    def load_train(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
        augmenter: Augment | None = None,
    ):
        datasets = [_NAIPHDF5Dataset(path) for path in self._train_paths]
        n_samples = sum([ds.n_samples for ds in datasets])

        loaded = [
            ds._load(
                label=label, bands=bands, veg_indices=veg_indices, augmenter=augmenter
            )
            for ds in datasets
        ]
        merged = tf.data.Dataset.sample_from_datasets(loaded)

        return merged.apply(tf.data.experimental.assert_cardinality(n_samples))

    def load_val(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
    ):
        # TODO: Refactor out the duplication
        datasets = [_NAIPHDF5Dataset(path) for path in self._val_paths]
        n_samples = sum([ds.n_samples for ds in datasets])

        loaded = [
            ds._load(label=label, bands=bands, veg_indices=veg_indices)
            for ds in datasets
        ]
        merged = tf.data.Dataset.sample_from_datasets(loaded)

        return merged.apply(tf.data.experimental.assert_cardinality(n_samples))

    def load_test(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
    ):
        # TODO: Refactor out the duplication:
        datasets = [_NAIPHDF5Dataset(path) for path in self._test_paths]
        n_samples = sum([ds.n_samples for ds in datasets])

        loaded = [
            ds._load(label=label, bands=bands, veg_indices=veg_indices)
            for ds in datasets
        ]
        merged = tf.data.Dataset.sample_from_datasets(loaded)

        return merged.apply(tf.data.experimental.assert_cardinality(n_samples))

    def __repr__(self) -> str:
        return f"<NAIPDatasetWrapper name={self.name}>"

    @property
    def name(self) -> str:
        """Name that describes the acquisitions and sampling metadata."""
        acquisition_names = "_".join([a.name for a in self.acquisitions])
        return "-".join([acquisition_names, self.suffix])

    @property
    def suffix(self) -> str:
        """
        Name suffix that describes metadata about how the acquisitions were sampled.
        """
        footprint_str = f"{self.footprint[0]}x{self.footprint[1]}"
        naip_res_str = float_to_str(self.naip_res)
        lidar_res_str = float_to_str(self.lidar_res)
        spacing_str = float_to_str(self.spacing)
        return "-".join(
            [
                naip_res_str,
                lidar_res_str,
                footprint_str,
                spacing_str,
            ]
        )

    @staticmethod
    def from_filename(filename: str, **kwargs) -> NAIPDatasetWrapper:
        """Load a NAIPDatasetWrapper from a filename."""
        basename = Path(filename).stem
        name, naip_res, lidar_res, footprint, spacing = basename.split("-")

        return NAIPDatasetWrapper(
            acquisitions=[Acquisition.from_name(name)],
            naip_res=str_to_float(naip_res),
            lidar_res=str_to_float(lidar_res),
            footprint=tuple(map(int, footprint.split("x"))),
            spacing=str_to_float(spacing),
            **kwargs,
        )

    def load_lidar(self) -> ee.Image:
        """Load the LiDAR labels for the dataset."""
        if len(self.acquisitions) == 1:
            return self.acquisitions[0].load_lidar(self.lidar_res)

        return ee.ImageCollection(
            [a.load_lidar(self.lidar_res) for a in self.acquisitions]
        ).mosaic()

    def load_naip(self) -> ee.Image:
        """Load the NAIP mosaic for the dataset."""
        if len(self.acquisitions) == 1:
            return self.acquisitions[0].load_naip(self.naip_res)

        return ee.ImageCollection(
            [a.load_naip(self.naip_res) for a in self.acquisitions]
        ).mosaic()

    @property
    def lidar_assets(self) -> list[str]:
        """Return the LiDAR assets used to build the dataset."""
        return [a.lidar_asset for a in self.acquisitions]

    @property
    def start_date(self) -> str:
        """Return the earliest acquisition date."""
        return min([a.start_date for a in self.acquisitions])

    @property
    def end_date(self) -> str:
        """Return the latest acquisition date."""
        return max([a.end_date for a in self.acquisitions])


@dataclass
class NAIPTFRecord:
    """A TFRecord dataset of NAIP tiles for inference.

    This class is used both for exporting NAIP imagery to TFRecords and for
    loading, parsing, and predicting on those TFRecords.
    """

    id: str
    footprint: tuple[float, float]
    year: int
    bounds: ee.Geometry = None
    res: float = NAIP_RES

    def __post_init__(self) -> None:
        footprint_str = f"{self.footprint[0]}x{self.footprint[1]}"
        res_str = float_to_str(self.res)

        self.name = "-".join(
            [
                self.id,
                str(self.year),
                res_str,
                footprint_str,
            ]
        )

        self.mixer_path = TFRECORD_DIR / f"{self.name}-mixer.json"
        self.mask_path = TFRECORD_DIR / f"{self.name}-mask.tif"

    @property
    def naip_shape(self):
        return int(self.footprint[0] // self.res), int(self.footprint[1] // self.res)

    def load_naip(self) -> ee.Image:
        if self.bounds is None:
            raise ValueError("Bounds must be set before loading NAIP imagery.")

        return (
            ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(self.bounds)
            .filterDate(str(self.year), str(self.year + 1))
            .mosaic()
            .select(BANDS)
            .reproject(CRS)
            .uint8()
        )

    def export_to_drive(
        self,
        mask: ee.Image = None,
        clip: ee.Feature = None,
        export_mask: bool = False,
        **kwargs,
    ):
        """
        Export the NAIP image to Google Drive.

        Parameters
        ----------
        mask : ee.Image
            An optional image mask to apply to the NAIP image before exporting.
        clip : ee.Feature
            An optional feature to clip the NAIP image to before exporting.
        export_mask : bool
            If true, a GeoTIFF mask will be exported alongside the TFRecord.
        kwargs : dict
            Additional arguments to pass to `ee.batch.Export.image.toDrive`.

        Returns
        -------
        ee.batch.Task
            The started image export task.
        """
        img = self.load_naip()
        if mask is not None:
            img = img.updateMask(mask)
        if clip is not None:
            img = img.clip(clip)

        snapped_origin = compute_snapped_origin(
            region=self.bounds,
            snap_size=GRID_SNAP,
            proj=ee.Projection(CRS),
        )

        if export_mask:
            mask_task = ee.batch.Export.image.toDrive(
                image=img.select(0).mask().uint8(),
                description=f"{self.name}-mask",
                region=self.bounds,
                fileFormat="GeoTIFF",
                maxPixels=1e13,
                crs=CRS,
                # The mask will be at the output LiDAR resolution
                crsTransform=compose_transform(
                    BASE_TRANSFORM,
                    origin=snapped_origin,
                    scale=30.0,
                ),
                **kwargs,
            )
            mask_task.start()

        task = ee.batch.Export.image.toDrive(
            image=img,
            description=self.name,
            region=self.bounds,
            fileFormat="TFRecord",
            maxPixels=1e13,
            # Note that explicitly setting the CRS changes the output to north-up,
            # compared to the south-up orientation of the training data. This isn't a
            # problem as long we correct the orientation during inference.
            # See https://github.com/lemma-osu/naip-cnn/issues/25
            crs=CRS,
            crsTransform=compose_transform(
                BASE_TRANSFORM,
                origin=snapped_origin,
                scale=self.res,
            ),
            formatOptions={
                "patchDimensions": self.naip_shape,
                "compressed": True,
                "maxFileSize": 1e9,
            },
            **kwargs,
        )
        task.start()
        return task

    def __repr__(self) -> str:
        return f"<NAIPTFRecord name={self.name}>"

    @cached_property
    def profile(self) -> dict:
        with open(self.mixer_path) as f:
            mixer = json.load(f)

        return {
            "width": mixer["patchesPerRow"],
            "height": mixer["totalPatches"] // mixer["patchesPerRow"],
            "crs": mixer["projection"]["crs"],
            "transform": mixer["projection"]["affine"]["doubleMatrix"],
        }

    def n_batches(self, batch_size: int) -> int:
        return int(
            np.ceil((self.profile["width"] * self.profile["height"]) / batch_size)
        )

    def __len__(self) -> int:
        """
        Calculate the number of samples in the TFRecord dataset.

        Note that this is a *very* slow process for large datasets.
        """
        tfrecords = list(TFRECORD_DIR.glob(f"{self.name}*.tfrecord.gz"))

        # Iterating in batches is faster, but batch size doesn't seem to matter much.
        data = tf.data.TFRecordDataset(tfrecords, compression_type="GZIP").batch(16)
        return data.reduce(0, lambda acc, batch: acc + len(batch)).numpy()

    def load_dataset(self, bands: tuple[str] = BANDS) -> tf.data.Dataset:
        """Load and parse the TFRecord dataset."""
        tfrecords = list(TFRECORD_DIR.glob(f"{self.name}*.tfrecord.gz"))
        if not tfrecords:
            raise FileNotFoundError(f"No TFRecord files found for {self.name}")

        # TFRecords are exported from Earth Engine with sequential numeric IDs. It's
        # easy for a few files to get lost during the download and unzipping process,
        # which will cause prediction to fail because the number of records won't match
        # the shape described in the sidecar file. Try to catch that here.
        tfrecord_ids = sorted(
            [int(f.stem.split("-")[-1].split(".")[0]) for f in tfrecords]
        )
        missing_ids = set(range(tfrecord_ids[-1])) - set(tfrecord_ids)
        if missing_ids:
            msg = (
                f"Expected at least {tfrecord_ids[-1] + 1} TFRecord files but found "
                f"{len(tfrecords)}. Missing TFRecord IDs: {missing_ids}"
            )
            raise FileNotFoundError(msg)

        data = tf.data.TFRecordDataset(tfrecords, compression_type="GZIP")
        return data.map(lambda x: self._parse(x, bands=bands))

    def _parse(
        self,
        serialized_example: bytes,
        bands: tuple[str],
    ) -> tf.Tensor:
        """Parse a single example from a TFRecord file containing NAIP imagery.

        Note that this currently only supports encoded byte images.
        """

        def decode_band(band: tf.Tensor) -> tf.Tensor:
            """Decode a byte string into a 2D uint8 Tensor."""
            data = tf.io.decode_raw(band, out_type=tf.uint8)
            return tf.cast(tf.reshape(data, self.naip_shape), tf.float32) / 255.0

        # uint8 data types must be parsed as strings and then decoded to uint8.
        # See https://issuetracker.google.com/issues/296941927
        features = {b: tf.io.FixedLenFeature(shape=[], dtype=tf.string) for b in bands}
        example = tf.io.parse_single_example(serialized_example, features)
        image = tf.stack([decode_band(example[b]) for b in bands], axis=-1)

        # Training data was extracted with NAIP imagery oriented south-up (positive
        # pixel height). To avoid domain shift during inference, ensure that the
        # TFRecord imagery is oriented the same way based on the affine transform.
        # See https://github.com/lemma-osu/naip-cnn/issues/25
        # Note that GEE affine transform matrices are stored as [a b c d e f] compared
        # to the GDAL convention of [c a b f d e] or shapely [a b d e c f] where:
        # a = pixel width
        # b = row rotation (typically zero)
        # c = x-coordinate of the origin pixel
        # d = column rotation (typically zero)
        # e = pixel height (typically negative)
        # f = y-coordinate of the origin pixel
        # Also see https://gdal.org/en/stable/tutorials/geotransforms_tut.html
        if self.profile["transform"][4] < 0:
            image = tf.image.flip_up_down(image)
        # This could theoretically happen in the x-direction as well.
        if self.profile["transform"][0] < 0:
            image = tf.image.flip_left_right(image)

        return tf.expand_dims(image, axis=-1)


def count_duplicate_images(train_path, val_path, key: str = "image") -> int:
    """
    Count the number of duplicate images between a train and validation HDF dataset.

    This is useful for ensuring that data leakage didn't occur during data splitting.

    References
    ----------
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    with h5py.File(val_path, "r") as f:
        val_image = f[key][:]
    with h5py.File(train_path, "r") as f:
        train_image = f[key][:]

    # Flatten the arrays
    val_image = val_image.reshape(val_image.shape[0], -1)
    train_image = train_image.reshape(train_image.shape[0], -1)

    val_dtype = {
        "names": [f"f{i}" for i in range(val_image.shape[1])],
        "formats": val_image.shape[1] * [val_image.dtype],
    }
    train_dtype = {
        "names": [f"f{i}" for i in range(train_image.shape[1])],
        "formats": train_image.shape[1] * [train_image.dtype],
    }

    return len(np.intersect1d(val_image.view(val_dtype), train_image.view(train_dtype)))
