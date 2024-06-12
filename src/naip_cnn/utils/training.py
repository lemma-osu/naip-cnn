from tensorflow.keras.callbacks import Callback


class EpochTracker(Callback):
    """A callback that records the last completed epoch number."""

    def __init__(self):
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch += 1
