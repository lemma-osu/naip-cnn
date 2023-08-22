from __future__ import annotations

import json

import tensorflow as tf


def parse_imagery_tfrecord(
    serialized_example: bytes,
    bands: tuple[str] = ("R", "G", "B", "N"),
    shape: tuple[int, int] = (30, 30),
    dtype: tf.dtypes.DType = tf.int64,
) -> tf.Tensor:
    """Parse a single example from a TFRecord file containing NAIP imagery."""
    flat_shape = (shape[0] * shape[1],)
    features = {b: tf.io.FixedLenFeature(shape=flat_shape, dtype=dtype) for b in bands}
    example = tf.io.parse_single_example(serialized_example, features)

    image = tf.stack([tf.reshape(example[b] / 255, shape) for b in bands], axis=-1)

    return tf.expand_dims(image, axis=-1)


def load_mixer_as_profile(mixer_path: str) -> dict:
    """Convert a mixer.json file to a rasterio profile."""
    with open(mixer_path) as f:
        mixer = json.load(f)

    return {
        "width": mixer["patchesPerRow"],
        "height": mixer["totalPatches"] // mixer["patchesPerRow"],
        "crs": mixer["projection"]["crs"],
        "transform": mixer["projection"]["affine"]["doubleMatrix"],
    }
