from __future__ import annotations

import difflib
import json
import tempfile
from pathlib import Path
from typing import Any

import tensorflow as tf

import wandb
from naip_cnn.acquisitions import Acquisition
from naip_cnn.augment import Augment
from naip_cnn.config import WANDB_PROJECT
from naip_cnn.data import NAIPDatasetWrapper
from naip_cnn.models import ModelRun

WandBRun = wandb.apis.public.runs.Run


class ConfigDict(dict):
    """
    A dictionary subclass that supports semantic equality checks for W&B configurations.

    W&B configurations may be modified when logged, e.g. converting tuples to lists and
    floats to ints. This class provides a custom equality check that accounts for these
    modifications that do not modify semantic meaning.
    """

    def __init__(self, d: dict):
        """
        Create a ConfigDict from a dictionary.

        This is done by:
        1. Recursively converting nested dictionaries into ConfigDicts.
        2. Serializing and unserializing to ensure JSON-compatible types.
        """
        cleaned = json.loads(json.dumps(d))

        super().__init__(
            {k: ConfigDict(v) if isinstance(v, dict) else v for k, v in cleaned.items()}
        )

    def __eq__(self, other: Any):
        # The other object may not be a dictionary in the case of nested comparisons
        if other is None or not isinstance(other, dict):
            return False

        other = ConfigDict(other)

        for key, value in self.items():
            # Keys are mismatched
            if key not in other:
                return False

            # Values are mismatched
            if value != other[key]:
                return False

        return True


def load_wandb_model(run_path: str) -> tf.keras.Model:
    """Load a model logged with a W&B run."""
    run = wandb.Api().run(run_path)

    model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if len(model_artifacts) != 1:
        raise ValueError(f"Expected one model artifact, found {len(model_artifacts)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(model_artifacts[0].download(root=tmpdir))
        # Download returns a directory with one model file
        model_path = next(model_dir.glob("*.keras"))
        return tf.keras.models.load_model(model_path)


def load_wandb_model_run(run_path: str) -> ModelRun:
    """Load a model run from a W&B run."""
    run = wandb.Api().run(run_path)
    cfg = run.config
    model = load_wandb_model(run_path)

    model_params = cfg["model"]["params"]
    bands = tuple(cfg["data"]["imagery"]["bands"].split("-"))
    label = cfg["data"]["lidar"]["label"]

    footprint = cfg["data"]["footprint"]["shape"]
    spacing = cfg["data"]["footprint"]["spacing"]
    lidar_res = cfg["data"]["lidar"]["resolution"]
    naip_res = cfg["data"]["imagery"]["resolution"]
    acquisition_names = cfg["data"]["lidar"]["acquisitions"]
    acquisitions = [Acquisition.from_name(name) for name in acquisition_names]

    dataset = NAIPDatasetWrapper(
        acquisitions,
        naip_res=naip_res,
        lidar_res=lidar_res,
        footprint=footprint,
        spacing=spacing,
    )

    return ModelRun(model, model_params, dataset, label, bands)


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
    mode: str = "online",
) -> WandBRun:
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
        for prev_run in wandb.Api().runs(WANDB_PROJECT):
            if config == prev_run.config and group == prev_run.group:
                raise ValueError(
                    f"Configuration matches an existing run ({prev_run.url}). "
                    "To allow duplicate configurations, set `allow_duplicate=True`."
                )

    return wandb.init(
        project=WANDB_PROJECT, config=config, group=group, save_code=False, mode=mode
    )


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
) -> ConfigDict:
    """Build a configuration dictionary for tracking an experiment with W&B."""
    return ConfigDict(
        {
            "training": {
                "batch_size": batch_size,
                "learning_rate": learn_rate,
                "epochs": epochs,
                "loss_function": model_run.model.loss,
            },
            "model": {
                "architecture": model_run.model.name,
                "path": model_run.model_path.as_posix(),
                "params": model_run.model_params,
            },
            "data": {
                "train": {
                    "paths": [path.as_posix() for path in dataset._train_paths],
                    "n_samples": n_train,
                    "augmentation": augmenter.to_json()
                    if augmenter is not None
                    else None,
                },
                "val": {
                    "paths": [path.as_posix() for path in dataset._val_paths],
                    "n_samples": n_val,
                },
                "date": {
                    "start": dataset.start_date,
                    "end": dataset.end_date,
                },
                "footprint": {
                    "shape": dataset.footprint,
                    "spacing": dataset.spacing,
                    "units": "meters",
                },
                "imagery": {
                    "bands": "-".join(bands),
                    "resolution": dataset.naip_res,
                    "years": [a.naip_year for a in dataset.acquisitions],
                },
                "lidar": {
                    "label": label,
                    "resolution": dataset.lidar_res,
                    "assets": dataset.lidar_assets,
                    "acquisitions": [a.name for a in dataset.acquisitions],
                },
            },
        }
    )


def _get_missing_keys(reference: dict, other: dict) -> set[str]:
    """Get the keys that are present in `reference` but missing from `other`."""
    missing_keys = []

    for key, value in reference.items():
        if key not in other:
            missing_keys.append(key)

        if isinstance(value, dict):
            missing_keys += _get_missing_keys(value, other.get(key, {}))

    return set(missing_keys)


def _get_mismatched_keys(
    reference: ConfigDict, other: ConfigDict, parent_key: str | None = None
) -> dict[str, tuple]:
    """Get the keys that are present in both dictionaries but have different values."""
    mismatched_keys = {}

    for key, value in reference.items():
        # Include the key of parent dictionaries for nested cases
        full_key = parent_key + "." + key if parent_key else key
        if isinstance(value, dict):
            mismatched_keys.update(
                _get_mismatched_keys(value, other.get(key, {}), parent_key=full_key)
            )

        elif value != other.get(key, None):
            mismatched_keys[full_key] = (value, other.get(key, None))

    return mismatched_keys


def compare_runs(reference_path: str, other_path: str) -> None:
    """Compare the configuration of two W&B runs and print a summary of differences."""
    config1 = wandb.Api().run(reference_path).config
    config2 = wandb.Api().run(other_path).config

    if config1 == config2:
        print("Runs are equivalent.")
        return

    missing_keys = _get_missing_keys(config2, config1)
    extra_keys = _get_missing_keys(config1, config2)
    mismatched_keys = _get_mismatched_keys(config1, config2)

    msg = "Runs have different configurations.\n"

    if extra_keys:
        msg += "\nExtra keys:\n----------------\n"
        for key in extra_keys:
            msg += f"\t{key}\n"

    if missing_keys:
        msg += "\nMissing keys:\n----------------\n"
        for key in missing_keys:
            msg += f"\t{key}\n"

    if mismatched_keys:
        msg += "\nMismatches:\n----------------\n"
        for key, (val1, val2) in mismatched_keys.items():
            msg += f"\n{key}:\n"
            for line in difflib.ndiff(str(val1).splitlines(), str(val2).splitlines()):
                msg += f"\t{line}\n"

    print(msg)
