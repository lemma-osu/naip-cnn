from __future__ import annotations

import json

import tensorflow as tf

from naip_cnn.config import BANDS


def parse_imagery_tfrecord(
    serialized_example: bytes,
    bands: tuple[str] = BANDS,
    shape: tuple[int, int] = (150, 150),
) -> tf.Tensor:
    """Parse a single example from a TFRecord file containing NAIP imagery.

    Note that this currently only supports encoded byte images.
    """

    def decode_band(band: tf.Tensor) -> tf.Tensor:
        """Decode a byte string into a 2D uint8 Tensor."""
        data = tf.io.decode_raw(band, out_type=tf.uint8)
        return tf.cast(tf.reshape(data, shape), tf.float32) / 255.0

    # uint8 data types must be parsed as strings and then decoded to uint8.
    # See https://issuetracker.google.com/issues/296941927
    features = {b: tf.io.FixedLenFeature(shape=[], dtype=tf.string) for b in bands}
    example = tf.io.parse_single_example(serialized_example, features)
    image = tf.stack([decode_band(example[b]) for b in bands], axis=-1)

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
