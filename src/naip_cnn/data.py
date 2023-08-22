from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import tensorflow_io as tfio

from naip_cnn.config import BANDS


class Dataset:
    """A Dataset of NAIP images and associated forest attribute labels, stored in H5."""

    def __init__(
        self,
        shape: tuple[int, int] = (30, 30),
        n: int = 30_000,
        name: str = "malheur",
        label: str = "cancov",
        root_dir: str = "../data",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the image, in pixels.
        n : int
            The number of samples in the dataset. Note that this is currently used to
            identify datasets, not to adjust the number of samples.
        name : str
            The name of the dataset to load.
        label : str
            The name of the label variable.
        root_dir : str
            The root directory of the data.
        train_split : float
            The proportion of the dataset to use for training.
        val_split : float
            The proportion of the dataset to use for validation.
        test_split : float
            The proportion of the dataset to use for testing.

        Raises
        ------
        ValueError
            If the train, validation, and test splits do not sum to 1.0.
        """
        if train_split + val_split + test_split != 1.0:
            raise ValueError("train_split + val_split + test_split must equal 1.0")

        self.shape = shape
        self.n = n
        self.name = name
        self.label = label
        self.root_dir = Path(root_dir)
        self.n_train = int(n * train_split)
        self.n_val = int(n * val_split)
        self.n_test = n - self.n_train - self.n_val

    def _get_path(self, year: int = 2020) -> Path:
        shape = f"{self.shape[0]}x{self.shape[1]}"
        filename = "_".join([self.name, str(self.n), shape, str(year)]) + ".h5"
        return self.root_dir / filename

    def _load(self, year: int = 2020, bands: tuple[str] = BANDS):
        """Load a Tensorflow Dataset of NAIP images from an HDF5 file.

        Parameters
        ----------
        year : int
            The year of the NAIP images to load.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        """

        def preprocess_image(image: tf.Tensor) -> tf.Tensor:
            """Cast byte image to float and select bands."""
            image = tf.gather(image, band_idxs, axis=-1)
            return tf.cast(image, tf.float32) / 255

        band_idxs = [BANDS.index(band) for band in bands]

        path = self._get_path(year).as_posix()
        images = tfio.IODataset.from_hdf5(path, dataset="/image").map(
            preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        labels = tfio.IODataset.from_hdf5(path, dataset=f"/{self.label}")

        ds = tf.data.Dataset.zip((images, labels))
        # Set the number of samples in the dataset
        return ds.apply(tf.data.experimental.assert_cardinality(self.n))

    def load_train(self, year: int = 2020, bands: tuple[str] = BANDS):
        """Load a Tensorflow Dataset of training NAIP images from a Parquet file.

        Parameters
        ----------
        year : int
            The year of the NAIP images to load.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        """
        ds = self._load(year=year, bands=bands)
        ds_train = ds.take(self.n_train)

        return ds_train.map(
            lambda x, y: (_augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    def load_val(self, year: int = 2020, bands: tuple[str] = BANDS):
        """Load a Tensorflow Dataset of validation NAIP images from a Parquet file.

        Parameters
        ----------
        year : int
            The year of the NAIP images to load.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        """
        ds = self._load(year=year, bands=bands)
        return ds.skip(self.n_train).take(self.n_val)

    def load_test(self, year: int = 2020, bands: tuple[str] = BANDS):
        """Load a Tensorflow Dataset of testing NAIP images from a Parquet file.

        Parameters
        ----------
        year : int
            The year of the NAIP images to load.
        bands : tuple[str]
            The bands to parse. This can be used to select specific subsets.
        """
        ds = self._load(year=year, bands=bands)
        return ds.skip(self.n_train + self.n_val).take(self.n_test)


@tf.autograph.experimental.do_not_convert
def _augment_image(img: tf.Tensor) -> tf.Tensor:
    """Augment an image Tensor.

    Parameters
    ----------
    img : tf.Tensor
        The image Tensor to augment.

    Returns
    -------
    tf.Tensor
        The augmented image Tensor.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    contrast_factor = tf.random.uniform([], 0.5, 1.5)
    brightness_factor = tf.random.uniform([], -0.2, 0.2)

    return tf.clip_by_value(img * contrast_factor + brightness_factor, 0.0, 1.0)
