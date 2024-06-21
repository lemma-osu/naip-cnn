import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class EpochTracker(Callback):
    """A callback that records the last completed epoch number."""

    def __init__(self):
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch += 1


class R2Score2D(tf.metrics.R2Score):
    """An R2Score metric that accepts batched 2D inputs."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]

        y_true = tf.reshape(y_true, (batch_size, -1))
        y_pred = tf.reshape(y_pred, (batch_size, -1))

        super().update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        for v in self.variables:
            # Backport of a bugfix
            # https://github.com/keras-team/keras/commit/402dcdcb2a2df441cdc34c2079dfdb12e7740d10
            v.assign(tf.zeros(v.shape, dtype=v.dtype))
