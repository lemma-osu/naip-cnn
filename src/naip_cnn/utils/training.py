from pathlib import Path
from warnings import warn

import tensorflow as tf
from keras.callbacks import ModelCheckpoint


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


class SaveBestWeights(ModelCheckpoint):
    """A ModelCheckpoint callback that saves the best weights and records the epoch."""

    def __init__(self, model_directory, **kwargs):
        self.best_epoch = 0
        self.last_epoch = 0

        # We need to initialize the current epoch before the super().__init__ call
        # sets the best attribute
        self._current_epoch = 0

        super().__init__(
            filepath=Path(model_directory) / ".weights.h5",
            save_best_only=True,
            save_weights_only=True,
            **kwargs,
        )

    @property
    def best(self):
        # ModelCheckpoint tracks a "best" attribute and sets it whenever a new best is
        # saved. We can turn that into a property and hijack the setter to also record
        # the epoch at which the best weights were saved.
        return self._best

    @best.setter
    def best(self, value):
        # When ModelCheckpoint sets a new best, record the epoch.
        prev_best = getattr(self, "_best", None)

        self._best = value
        self.best_epoch = self._current_epoch

        if prev_best:
            print(f"\nLoss improved by {prev_best - value:.4f}. Weights saved!\n")

    def on_epoch_end(self, epoch, logs=None):
        """Record the last completed epoch number."""
        self.last_epoch = epoch
        super().on_epoch_end(epoch, logs)

    def load_best_weights(self, delete_weights=True) -> None:
        """Load the best weights into the model and optionally delete the checkpoint."""
        weights = Path(self.filepath)
        if not weights.exists():
            warn("No best weights found.", UserWarning, stacklevel=0)
            return

        # Increment by 1 to match TF logging
        print(f"Loading best weights from epoch {self.best_epoch + 1}...")
        self.model.load_weights(weights)

        if delete_weights:
            weights.unlink()
