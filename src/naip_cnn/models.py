from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from naip_cnn.config import MODEL_DIR
from naip_cnn.data import NAIPDatasetWrapper


@dataclass
class ModelRun:
    """A model run with associated model, dataset, and parameters."""

    model: tf.keras.Model
    dataset: NAIPDatasetWrapper
    label: str
    bands: tuple[str]
    suffix: str = None

    def __post_init__(self):
        band_str = "".join(self.bands)
        parts = [self.model.name, self.dataset.name, self.label, band_str]
        if self.suffix:
            parts.append(self.suffix)

        self.name = "-".join(parts)
        self.model_path = MODEL_DIR / f"{self.name}.keras"

    def __repr__(self) -> str:
        return f"<ModelRun name={self.name}>"

    def load_best_checkpoint(self, delete_checkpoints=True) -> int:
        """Load the best weights from a checkpoint and return the associated epoch.

        This assumes that checkpoints are named ".checkpoint_{run.name}_{epoch}.h5" and
        were generated with `save_best_only=True`, so that the last checkpoint saved is
        the best. All checkpoints are optionally deleted after the best one is loaded.
        """
        checkpoints = list(MODEL_DIR.glob(f".checkpoint_{self.name}_*.h5"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found.")

        checkpoints.sort()
        checkpoint_path = checkpoints[-1]
        epoch = int(checkpoint_path.stem.split("_")[-1])

        self.model.load_weights(checkpoint_path)

        if delete_checkpoints:
            for file in checkpoints:
                file.unlink()

        return epoch

    def save_model(self) -> Path:
        """Save the model to disk."""
        self.model.save(self.model_path)
        return self.model_path

    @staticmethod
    def from_filename(filename: str) -> ModelRun:
        """Load a model run from a filename."""
        basename = Path(filename).stem
        parts = basename.split("-")
        bands = tuple(parts[-1])
        label = parts[-2]
        dataset_name = "-".join(parts[1:-2])

        model = tf.keras.models.load_model(MODEL_DIR / f"{basename}.keras")
        bands = tuple(bands)
        dataset = NAIPDatasetWrapper.from_filename(dataset_name)

        return ModelRun(model, dataset, label, bands)


def CNN_v1(
    kernel_dim=5,
    filter_no=32,
    Dense1_no=256,
    Dense2_no=32,
    shape: tuple[int, int, int] = (30, 30, 4),
):
    kernel_size = (kernel_dim, kernel_dim, 4)

    return tf.keras.models.Sequential(
        [
            layers.Conv3D(
                filters=filter_no,
                kernel_size=kernel_size,
                input_shape=(*shape, 1),
                padding="valid",
                activation="relu",
                use_bias=True,
            ),
            layers.Flatten(),
            layers.Dense(Dense1_no, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(Dense2_no, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(units=1, activation="linear"),
        ],
        name="CNN_v1",
    )


def CNN_v2(shape: tuple[int, int, int] = (150, 150, 4), out_shape=(5, 5)):
    return tf.keras.models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(units=out_shape[0] * out_shape[1], activation="linear"),
            layers.Reshape(out_shape),
        ],
        name="CNN_v2",
    )
