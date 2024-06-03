from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Callable

import ee
import h5py
import spyndex
import tensorflow as tf
import tensorflow_io as tfio

from naip_cnn.acquisitions import Acquisition
from naip_cnn.config import BANDS, TRAIN_DIR
from naip_cnn.utils import float_to_str, str_to_float


@tf.autograph.experimental.do_not_convert
def _flip_contrast_brightness_augment(
    img: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment an image and label Tensor.

    The image and label are randomly flipped, and the image is randomly adjusted in
    contrast and brightness. The image MUST be in the range [0, 1] prior to augmenting.

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


class _TrainTestValidationDataset(ABC):
    """
    A wrapper around a Tensorflow Dataset that can be split into train, test, and
    validation sets.
    """

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
    def _load(self, **kwargs) -> tf.data.Dataset:
        """Load the full dataset."""
        ...

    @property
    def n_train(self) -> int:
        return int(len(self) * self.train_split)

    @property
    def n_val(self) -> int:
        return int(len(self) * self.val_split)

    @property
    def n_test(self) -> int:
        return len(self) - self.n_train - self.n_val

    def load_train(
        self, augmenter: Callable | None = None, **kwargs
    ) -> tf.data.Dataset:
        ds = self._load(**kwargs).take(self.n_train)
        if augmenter is not None:
            ds = ds.map(
                lambda x, y: augmenter(x, y), num_parallel_calls=tf.data.AUTOTUNE
            )

        return ds

    def load_val(self, **kwargs) -> tf.data.Dataset:
        return self._load(**kwargs).skip(self.n_train).take(self.n_val)

    def load_test(self, **kwargs) -> tf.data.Dataset:
        return self._load(**kwargs).skip(self.n_train + self.n_val).take(self.n_test)


class _HDF5DatasetMixin:
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

    def _load(
        self,
        label: str,
        shuffle: bool = True,
        seed: int = 0,
        feature_preprocessor: Callable | None = None,
        label_preprocessor: Callable | None = None,
    ) -> tf.data.Dataset:
        """Load a zipped dataset of features and labels from an HDF5 file."""
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist")
        features = tfio.IODataset.from_hdf5(
            self.path.as_posix(), dataset=f"/{self.feature_name}"
        )
        labels = tfio.IODataset.from_hdf5(self.path.as_posix(), dataset=f"/{label}")
        ds = tf.data.Dataset.zip((features, labels))

        if shuffle:
            # Using `reshuffle_each_iteration` before splitting leaks data. See
            # https://github.com/tensorflow/tensorflow/issues/59279
            ds = ds.shuffle(
                # TODO: Decide how to handle buffer size for large datasets. Maybe I
                # should pre-split data to speed up training?
                buffer_size=10_000,
                seed=seed,
                reshuffle_each_iteration=False,
            )

        if feature_preprocessor is not None or label_preprocessor is not None:
            feature_preprocessor = feature_preprocessor or (lambda x: x)
            label_preprocessor = label_preprocessor or (lambda x: x)
            ds = ds.map(
                lambda x, y: (feature_preprocessor(x), label_preprocessor(y)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return ds.apply(tf.data.experimental.assert_cardinality(self.n_samples))


class _NAIPHDF5Dataset(_HDF5DatasetMixin, _TrainTestValidationDataset):
    """A dataset of NAIP images and forest attribute labels in an HDF5 file."""

    def _load(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
        shuffle: bool = True,
        seed: int = 0,
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
            shuffle=shuffle,
            seed=seed,
            feature_preprocessor=preprocess_naip,
        )

    def load_train(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
        augmenter: Callable | None = _flip_contrast_brightness_augment,
        shuffle: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        """Load a training Tensorflow Dataset of NAIP images from an HDF5 file,
        optionally applying data augmentation.

        Parameters
        ----------
        label : str
            The name of the label variable.
        bands : tuple[str]
            The bands to load. This can be used to select specific subsets.
        augmenter : function
            The function to use for data augmentation. Must take and return a tuple of
            (image, label) tensors.
        shuffle : bool
            If True, the dataset will be shuffled prior to splitting.
        seed : int
            The random seed to use for shuffling.
        **kwargs
            Additional arguments passed to _load.
        """
        return super().load_train(
            label=label,
            bands=bands,
            veg_indices=veg_indices,
            augmenter=augmenter,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )

    def load_val(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
        shuffle: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        """Load a validtion Tensorflow Dataset from an HDF5 file.

        Parameters
        ----------
        label : str
            The name of the label variable.
        bands : tuple[str]
            The bands to load. This can be used to select specific subsets.
        shuffle : bool
            If True, the dataset will be shuffled prior to splitting.
        seed : int
            The random seed to use for shuffling.
        **kwargs
            Additional arguments passed to _load.
        """
        return super().load_val(
            label=label,
            bands=bands,
            veg_indices=veg_indices,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )

    def load_test(
        self,
        label: str,
        bands: tuple[str] = BANDS,
        veg_indices: tuple[str] = tuple(),
        shuffle: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        """Load a testing Tensorflow Dataset from an HDF5 file.

        Parameters
        ----------
        label : str
            The name of the label variable.
        bands : tuple[str]
            The bands to load. This can be used to select specific subsets.
        shuffle : bool
            If True, the dataset will be shuffled prior to splitting.
        seed : int
            The random seed to use for shuffling.
        **kwargs
            Additional arguments passed to _load.
        """
        return super().load_test(
            label=label,
            bands=bands,
            veg_indices=veg_indices,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )


class NAIPDatasetWrapper:
    """A wrapper around a NAIP training dataset that stores additional metadata and
    provides convenience methods for loading the dataset.

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
            Additional arguments passed to _HDF5Dataset, e.g. train_split, val_split.
        """
        self.acquisition = acquisition
        self.naip_res = naip_res
        self.lidar_res = lidar_res
        self.footprint = footprint
        self.spacing = spacing if spacing is not None else footprint[0]
        self.name = self._get_name()
        self.hdf_path = TRAIN_DIR / (self.name + ".h5")
        self.csv_path = TRAIN_DIR / (self.name + ".csv")

        # The shape of the image and labels in pixels
        h, w = self.footprint
        self.naip_shape = int(h // naip_res), int(w // naip_res)
        self.lidar_shape = int(h // lidar_res), int(w // lidar_res)

        self.dataset = _NAIPHDF5Dataset(
            path=self.hdf_path, feature_name="image", **kwargs
        )

    def __repr__(self) -> str:
        return f"<NAIPDatasetWrapper name={self.name}>"

    def _get_name(self) -> str:
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
    def from_filename(filename: str, **kwargs) -> NAIPDatasetWrapper:
        """Load a NAIPDatasetWrapper from a filename."""
        basename = Path(filename).stem
        name, naip_res, lidar_res, footprint, spacing = basename.split("-")

        return NAIPDatasetWrapper(
            acquisition=Acquisition.from_name(name),
            naip_res=str_to_float(naip_res),
            lidar_res=str_to_float(lidar_res),
            footprint=tuple(map(int, footprint.split("x"))),
            spacing=str_to_float(spacing),
            **kwargs,
        )

    def load_lidar(self) -> ee.Image:
        """Load the LiDAR labels for the dataset."""
        return self.acquisition.load_lidar(self.lidar_res)

    def load_naip(self) -> ee.Image:
        """Load the NAIP mosaic for the dataset."""
        return self.acquisition.load_naip(self.naip_res)
