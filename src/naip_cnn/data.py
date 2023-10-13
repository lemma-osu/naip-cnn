from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import h5py
import tensorflow as tf
import tensorflow_io as tfio

from naip_cnn.config import BANDS


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

        self.n_train = int(len(self) * self.train_split)
        self.n_val = int(len(self) * self.val_split)
        self.n_test = len(self) - self.n_train - self.n_val

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        ...

    @abstractmethod
    def _load(self, **kwargs):
        """Load the full dataset."""
        ...

    def load_train(self, **kwargs):
        return self._load(**kwargs).take(self.n_train)

    def load_val(self, **kwargs):
        return self._load(**kwargs).skip(self.n_train).take(self.n_val)

    def load_test(self, **kwargs):
        return self._load(**kwargs).skip(self.n_train + self.n_val).take(self.n_test)


class HDF5Dataset(TrainTestValidationDataset):
    """A dataset of features and labels stored in an HDF5 file."""

    def __init__(self, path, feature_name: str, label_name: str, **kwargs):
        self.path = path
        self.feature_name = feature_name
        self.label_name = label_name
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

    def _load(self, shuffle: bool = True, seed: int = 0) -> tf.data.Dataset:
        """Load a zipped dataset of features and labels from an HDF5 file."""
        features = tfio.IODataset.from_hdf5(self.path, dataset=f"/{self.feature_name}")
        labels = tfio.IODataset.from_hdf5(self.path, dataset=f"/{self.label_name}")
        ds = tf.data.Dataset.zip((features, labels))

        if shuffle:
            ds = ds.shuffle(self.n_samples, seed=seed)

        return ds.apply(tf.data.experimental.assert_cardinality(self.n_samples))


class NAIPDataset(HDF5Dataset):
    """A dataset of NAIP images and associated forest attribute labels."""

    def __init__(
        self,
        shape: tuple[int, int] = (150, 150),
        name: str = "MAL2016_CanyonCreek",
        label: str = "cover",
        root_dir: str = "../data/training",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the image, in pixels.
        name : str
            The name of the dataset to load.
        label : str
            The name of the label variable.
        root_dir : str
            The root directory of the data.
        **kwargs
            Additional arguments passed to HDF5Dataset.
        """
        self.shape = shape
        self.name = name
        self.root_dir = Path(root_dir)

        str_shape = f"{self.shape[0]}x{self.shape[1]}"
        path = self.root_dir / ("_".join([self.name, str_shape]) + ".h5")
        super().__init__(
            path=path.as_posix(), feature_name="image", label_name=label, **kwargs
        )

    def _load(self, bands: tuple[str] = BANDS, shuffle: bool = True, seed: int = 0):
        """Load a Tensorflow Dataset of NAIP images from an HDF5 file.

        Parameters
        ----------
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
            ._load(shuffle=shuffle, seed=seed)
            .map(
                lambda x, y: (preprocess_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    def load_train(
        self,
        bands: tuple[str] = BANDS,
        augmenter=_augment,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """Load a Tensorflow Dataset of training NAIP images from an HDF5 file.

        Parameters
        ----------
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
        ds = super().load_train(bands=bands, shuffle=shuffle, seed=seed)
        if augmenter is not None:
            ds = ds.map(
                lambda x, y: augmenter(x, y), num_parallel_calls=tf.data.AUTOTUNE
            )
        return ds

    def load_val(self, bands: tuple[str] = BANDS, shuffle: bool = True, seed: int = 0):
        """Load a Tensorflow Dataset of validation NAIP images from an HDF5 file.

        Parameters
        ----------
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        shuffle : bool
            If True, the dataset will be shuffled prior to splitting.
        seed : int
            The random seed to use for shuffling.
        """
        return super().load_val(bands=bands, shuffle=shuffle, seed=seed)

    def load_test(self, bands: tuple[str] = BANDS, shuffle: bool = True, seed: int = 0):
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
        return super().load_test(bands=bands, shuffle=shuffle, seed=seed)
