from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, TypeVar

import tensorflow as tf

TensorPair = Tuple[tf.Tensor, tf.Tensor]


class AugmentFunction(ABC):
    """
    An abstract augment function that modifies an image, label pair.

    Note that images should be pre-processed to the range [0, 1] prior to augmentation.
    """

    def __init__(self, **kwargs):
        # Override default attributes with passed parameters
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # All parameters must be defined as attributes
                raise ValueError(f"Invalid param {k}")

            setattr(self, k, v)

        # Expose any non-function public attributes as parameters
        self.params = {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and not callable(getattr(self, k))
        }

    @abstractmethod
    @tf.function
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> TensorPair:
        """Apply the augmentation function to the image and label tensors."""


class RandomFlip(AugmentFunction):
    """Randomly flip an image, label pair horizontally and/or vertically."""

    left_right: bool = True
    up_down: bool = True

    @tf.function
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> TensorPair:
        if self.left_right:
            flip_lr = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
            image = tf.cond(
                flip_lr > 0.5, lambda: tf.image.flip_left_right(image), lambda: image
            )
            label = tf.cond(
                flip_lr > 0.5, lambda: tf.image.flip_left_right(label), lambda: label
            )

        if self.up_down:
            flip_ud = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
            image = tf.cond(
                flip_ud > 0.5, lambda: tf.image.flip_up_down(image), lambda: image
            )
            label = tf.cond(
                flip_ud > 0.5, lambda: tf.image.flip_up_down(label), lambda: label
            )

        return image, label


class RandomRotation(AugmentFunction):
    """Randomly rotate an image, label pair by 90, 180, or 270 degrees."""

    @tf.function
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> TensorPair:
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        label = tf.image.rot90(label, k)
        return image, label


class RandomContrast(AugmentFunction):
    """Randomly adjust an image's contrast."""

    min_delta: float = 0.5
    max_delta: float = 1.5

    @tf.function
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> TensorPair:
        contrast_factor = tf.random.uniform([], self.min_delta, self.max_delta)
        return tf.clip_by_value(image * contrast_factor, 0.0, 1.0), label


class RandomBrightness(AugmentFunction):
    """Randomly adjust an image's brightness."""

    min_factor: float = -0.2
    max_factor: float = 0.2

    @tf.function
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> TensorPair:
        brightness_factor = tf.random.uniform([], self.min_factor, self.max_factor)
        return tf.clip_by_value(image + brightness_factor, 0.0, 1.0), label


AugmentFunctionType = TypeVar("AugmentFunctionType", bound=AugmentFunction)


class Augment:
    """A collection of augmenting functions to apply to an image, label pair."""

    def __init__(self, *args: AugmentFunctionType):
        self.augmenters = [*args]

    def to_json(self):
        """Retrieve a JSON representation of all functions and their parameters."""
        return {a.__class__.__name__: a.params for a in self.augmenters}

    @tf.function(jit_compile=True)
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> TensorPair:
        # We need a channel dimension to use the tf.image functions
        label = tf.expand_dims(label, axis=-1)

        for augmenter in self.augmenters:
            image, label = augmenter(image, label)

        # Remove the extra channel
        label = tf.squeeze(label, axis=-1)

        return image, label
