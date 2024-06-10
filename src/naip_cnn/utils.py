from __future__ import annotations

import json
from typing import TYPE_CHECKING

import h5py
import numpy as np
import wandb

from naip_cnn.config import WANDB_PROJECT

if TYPE_CHECKING:
    from naip_cnn.augment import Augment
    from naip_cnn.data import NAIPDatasetWrapper
    from naip_cnn.models import ModelRun


def float_to_str(f: float) -> str:
    """Stringify a float for a filename.

    For example:
    - 0.5 -> '0p5'
    - 1.0 -> '1'
    """
    f = float(f)
    if f.is_integer():
        return str(int(f))
    return str(f).replace(".", "p")


def str_to_float(s: str) -> float:
    """Parse a string into a float.

    For example:
    - '0p5' -> 0.5
    - '1' -> 1.0
    """
    return float(s.replace("p", "."))


def count_duplicate_images(train_path, val_path, key: str = "image") -> int:
    """
    Count the number of duplicate images between a train and validation HDF dataset.

    This is useful for ensuring that data leakage didn't occur during data splitting.

    References
    ----------
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    """
    with h5py.File(val_path, "r") as f:
        val_image = f[key][:]
    with h5py.File(train_path, "r") as f:
        train_image = f[key][:]

    # Flatten the arrays
    val_image = val_image.reshape(val_image.shape[0], -1)
    train_image = train_image.reshape(train_image.shape[0], -1)

    val_dtype = {
        "names": [f"f{i}" for i in range(val_image.shape[1])],
        "formats": val_image.shape[1] * [val_image.dtype],
    }
    train_dtype = {
        "names": [f"f{i}" for i in range(train_image.shape[1])],
        "formats": train_image.shape[1] * [train_image.dtype],
    }

    return np.intersect1d(val_image.view(val_dtype), train_image.view(train_dtype))


def initialize_wandb_run(
    *,
    dataset: NAIPDatasetWrapper,
    model_run: ModelRun,
    bands: tuple[str],
    label: str,
    batch_size: int,
    learn_rate: float,
    epochs: int,
    n_train: int,
    n_val: int,
    augmenter: Augment | None = None,
    allow_duplicate: bool = False,
) -> wandb.apis.public.runs.Run:
    """Initialize a W&B run for tracking an experiment."""
    group = f"{dataset.lidar_res:n}m_{label}"

    config = _build_wandb_config(
        dataset=dataset,
        model_run=model_run,
        bands=bands,
        label=label,
        batch_size=batch_size,
        learn_rate=learn_rate,
        epochs=epochs,
        n_train=n_train,
        n_val=n_val,
        augmenter=augmenter,
    )

    if not allow_duplicate:
        prev_runs = wandb.Api().runs(WANDB_PROJECT)

        for prev_run in prev_runs:
            if _configs_are_equal(prev_run.config, config):
                raise ValueError(
                    f"Configuration matches an existing run ({prev_run.url}). "
                    "To allow duplicate configurations, set `allow_duplicate=True`."
                )

    return wandb.init(project=WANDB_PROJECT, config=config, group=group, save_code=True)


def _build_wandb_config(
    *,
    dataset: NAIPDatasetWrapper,
    model_run: ModelRun,
    bands: tuple[str],
    label: str,
    batch_size: int,
    learn_rate: float,
    epochs: int,
    n_train: int,
    n_val: int,
    augmenter: Augment | None = None,
) -> dict:
    """Build a configuration dictionary for tracking an experiment with W&B."""
    return {
        "training": {
            "batch_size": batch_size,
            "learning_rate": learn_rate,
            "epochs": epochs,
        },
        "model": {
            "architecture": model_run.model.name,
            "path": model_run.model_path.as_posix(),
            "params": model_run.model_params,
        },
        "data": {
            "train": {
                "path": dataset._train.path.as_posix(),
                "n_samples": n_train,
                "augmentation": augmenter.to_json() if augmenter is not None else None,
            },
            "val": {
                "path": dataset._val.path.as_posix(),
                "n_samples": n_val,
            },
            "date": {
                "start": dataset.acquisition.start_date,
                "end": dataset.acquisition.end_date,
            },
            "footprint": {
                "shape": dataset.footprint,
                "spacing": dataset.spacing,
                "units": "meters",
            },
            "imagery": {
                "bands": "-".join(bands),
                "resolution": dataset.naip_res,
                "acquisition": dataset.acquisition.name,
            },
            "lidar": {
                "label": label,
                "resolution": dataset.lidar_res,
                "asset": dataset.acquisition.lidar_asset,
            },
        },
    }


def _configs_are_equal(config1, config2):
    """
    Compare two configuration dictionaries for equality.

    Note that we implement this from scratch to rather than a simple equality check
    because values may be modified by W&B, e.g. converting tuples to lists and floats
    to ints.
    """
    # Normalize to JSON to, e.g. convert tuples to lists
    config1 = json.loads(json.dumps(config1))
    config2 = json.loads(json.dumps(config2))

    # Check if the comparison is None, which may occur for nested checks
    if config2 is None:
        return False

    for key, value in config1.items():
        # Keys are mismatched
        if key not in config2:
            return False

        # Compare nested dictionaries
        if isinstance(value, dict) and not _configs_are_equal(value, config2[key]):
            return False

        # Values are mismatched
        if value != config2[key]:
            return False

    return True
