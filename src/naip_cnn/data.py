from __future__ import annotations

import geopandas as gpd
import numpy as np
import tensorflow as tf

from naip_cnn.sampling import _parse_pixel_array


def load_parquet_dataset(
    path: str, target_names: tuple[str], **kwargs
) -> tf.data.Dataset:
    """Load a Tensorflow Dataset from a Parquet file.

    Parameters
    ----------
    path : str
        Path to the Parquet file.
    target_names : tuple[str]
        Names of the target columns to load.
    **kwargs
        Keyword arguments to pass to `naip_cnn.sampling._parse_pixel_array`.
    """
    df = gpd.read_parquet(path)
    images, labels = list(
        zip(
            *(
                (_parse_pixel_array(row, **kwargs), row.loc[list(target_names)].values)
                for _, row in df.iterrows()
            )
        )
    )
    images = np.stack(images).astype(np.float32) / 255
    labels = np.stack(labels).astype(np.float32) / 100

    return tf.data.Dataset.from_tensor_slices((images, labels))
