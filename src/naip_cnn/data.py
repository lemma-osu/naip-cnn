from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import ee
import h5py
import tensorflow as tf
import tensorflow_io as tfio

from naip_cnn.acquisitions import Acquisition
from naip_cnn.config import BANDS, TRAIN_DIR
from naip_cnn.utils import float_to_str


@tf.autograph.experimental.do_not_convert
def _augment(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment an image Tensor.

    Parameters
    ----------
    img : tf.Tensor
        The image Tensor to augment.
    label : tf.Tensor
        The label Tensor to augment.

    Returns
    -------
    tf.Tensor
        The augmented image Tensor.
    """
    flip_lr = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    flip_ud = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # We need a channel dimension to use the tf.image functions
    label = tf.expand_dims(label, axis=-1)

    img = tf.cond(flip_lr > 0.5, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(flip_ud > 0.5, lambda: tf.image.flip_up_down(img), lambda: img)
    label = tf.cond(
        flip_lr > 0.5, lambda: tf.image.flip_left_right(label), lambda: label
    )
    label = tf.cond(flip_ud > 0.5, lambda: tf.image.flip_up_down(label), lambda: label)

    label = tf.squeeze(label, axis=-1)

    contrast_factor = tf.random.uniform([], 0.5, 1.5)
    brightness_factor = tf.random.uniform([], -0.2, 0.2)
    img = tf.clip_by_value(img * contrast_factor + brightness_factor, 0.0, 1.0)

    return img, label


class TrainTestValidationDataset(ABC):
    """A dataset that can be split into train, test, and validation sets."""

    def __init__(self, train_split=0.8, val_split=0.1, test_split=0.1):
        if train_split + val_split + test_split != 1.0:
            raise ValueError("train_split + val_split + test_split must equal 1.0")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        ...

    @abstractmethod
    def _load(self, **kwargs):
        """Load the full dataset."""
        ...

    @property
    def n_train(self):
        return int(len(self) * self.train_split)

    @property
    def n_val(self):
        return int(len(self) * self.val_split)

    @property
    def n_test(self):
        return len(self) - self.n_train - self.n_val

    def load_train(self, **kwargs):
        return self._load(**kwargs).take(self.n_train)

    def load_val(self, **kwargs):
        return self._load(**kwargs).skip(self.n_train).take(self.n_val)

    def load_test(self, **kwargs):
        return self._load(**kwargs).skip(self.n_train + self.n_val).take(self.n_test)


class HDF5Dataset(TrainTestValidationDataset):
    """A dataset of features and labels stored in an HDF5 file."""

    def __init__(self, path, feature_name: str, **kwargs):
        self.path = path
        self.feature_name = feature_name
        super().__init__(**kwargs)

    @cached_property
    def n_samples(self) -> int:
        """The total number of samples, based on the first dataset."""
        with h5py.File(self.path, "r") as f:
            first_key = list(f.keys())[0]
            return f[first_key].shape[0]

    def __len__(self) -> int:
        """Return the total number of samples, based on the first dataset."""
        return self.n_samples

    def _load(self, label: str, shuffle: bool = True, seed: int = 0) -> tf.data.Dataset:
        """Load a zipped dataset of features and labels from an HDF5 file."""
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist")
        features = tfio.IODataset.from_hdf5(
            self.path.as_posix(), dataset=f"/{self.feature_name}"
        )
        labels = tfio.IODataset.from_hdf5(self.path.as_posix(), dataset=f"/{label}")
        ds = tf.data.Dataset.zip((features, labels))

        if shuffle:
            ds = ds.shuffle(self.n_samples, seed=seed)

        return ds.apply(tf.data.experimental.assert_cardinality(self.n_samples))


class TrainingDataset(HDF5Dataset):
    """A dataset of NAIP images and associated forest attribute labels.

    This class is used both to define and generate sample datasets and to load
    previously generated datasets.
    """

    def __init__(
        self,
        acquisition: Acquisition,
        naip_res=1.0,
        lidar_res=30.0,
        footprint: tuple[int, int] = (150, 150),
        spacing: float = None,
        **kwargs,
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
        **kwargs
            Additional arguments passed to HDF5Dataset, e.g. train_split, val_split,
        """
        self.acquisition = acquisition
        self.naip_res = naip_res
        self.lidar_res = lidar_res
        self.footprint = footprint
        self.spacing = spacing if spacing is not None else footprint[0]

        super().__init__(path=self.hdf_path, feature_name="image", **kwargs)

    def __repr__(self) -> str:
        return f"<TrainingDataset name={self.name}>"

    @property
    def name(self):
        footprint_str = f"{self.footprint[0]}x{self.footprint[1]}"
        naip_res_str = float_to_str(self.naip_res)
        lidar_res_str = float_to_str(self.lidar_res)
        spacing_str = float_to_str(self.spacing)
        return "-".join(
            [
                self.acquisition.name,
                naip_res_str,
                lidar_res_str,
                footprint_str,
                spacing_str,
            ]
        )

    @staticmethod
    def from_filename(filename: str) -> TrainingDataset:
        """Load a TrainingDataset from a filename."""
        basename = Path(filename).stem
        name, naip_res, lidar_res, footprint, spacing = basename.split("-")

        return TrainingDataset(
            acquisition=Acquisition.from_name(name),
            naip_res=float(naip_res),
            lidar_res=float(lidar_res),
            footprint=tuple(map(int, footprint.split("x"))),
            spacing=float(spacing),
        )

    @property
    def hdf_path(self):
        """Path to the HDF5 file."""
        return TRAIN_DIR / (self.name + ".h5")

    @property
    def csv_path(self):
        """Path to the raw CSV file."""
        return TRAIN_DIR / (self.name + ".csv")

    @property
    def naip_shape(self):
        """The shape of the NAIP images in pixels."""
        return int(self.footprint[0] // self.naip_res), int(
            self.footprint[1] // self.naip_res
        )

    @property
    def lidar_shape(self):
        """The shape of the LiDAR labels in pixels."""
        return int(self.footprint[0] // self.lidar_res), int(
            self.footprint[1] // self.lidar_res
        )

    def load_lidar(self) -> ee.Image:
        """Load the LiDAR labels for the dataset."""
        return self.acquisition.load_lidar(self.lidar_res)

    def load_naip(self) -> ee.Image:
        """Load the NAIP mosaic for the dataset."""
        return self.acquisition.load_naip(self.naip_res)

    def _load(
        self, label: str, bands: tuple[str] = BANDS, shuffle: bool = True, seed: int = 0
    ):
        """Load a Tensorflow Dataset of NAIP images from an HDF5 file.

        Parameters
        ----------
        label : str
            The name of the label variable.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        """
        band_idxs = [BANDS.index(band) for band in bands]

        def preprocess_image(image: tf.Tensor) -> tf.Tensor:
            """Cast byte image to float and select bands."""
            image = tf.gather(image, band_idxs, axis=-1)
            return tf.cast(image, tf.float32) / 255

        return (
            super()
            ._load(label=label, shuffle=shuffle, seed=seed)
            .map(
                lambda x, y: (preprocess_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    def load_train(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        augmenter=_augment,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """Load a Tensorflow Dataset of training NAIP images from an HDF5 file.

        Parameters
        ----------
        label : str
            The name of the label variable.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        augmenter : function
            The function to use for data augmentation. Must take and return a tuple of
            (image, label) tensors.
        shuffle : bool
            If True, the dataset will be shuffled prior to splitting.
        seed : int
            The random seed to use for shuffling.
        """
        ds = super().load_train(label=label, bands=bands, shuffle=shuffle, seed=seed)
        if augmenter is not None:
            ds = ds.map(
                lambda x, y: augmenter(x, y), num_parallel_calls=tf.data.AUTOTUNE
            )
        return ds

    def load_val(
        self, label: str, bands: tuple[str] = BANDS, shuffle: bool = True, seed: int = 0
    ):
        """Load a Tensorflow Dataset of validation NAIP images from an HDF5 file.

        Parameters
        ----------
        label : str
            The name of the label variable.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        shuffle : bool
            If True, the dataset will be shuffled prior to splitting.
        seed : int
            The random seed to use for shuffling.
        """
        return super().load_val(label=label, bands=bands, shuffle=shuffle, seed=seed)

    def load_test(
        self, label: str, bands: tuple[str] = BANDS, shuffle: bool = True, seed: int = 0
    ):
        """Load a Tensorflow Dataset of testing NAIP images from an HDF5 file.

        Parameters
        ----------
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        shuffle : bool
            If True, the dataset will be shuffled prior to splitting.
        seed : int
            The random seed to use for shuffling.
        """
        return super().load_test(label=label, bands=bands, shuffle=shuffle, seed=seed)
