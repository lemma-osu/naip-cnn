from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def build_wandb_config(
    *,
    dataset: NAIPDatasetWrapper,
    model_run: ModelRun,
    bands: tuple[str],
    label: str,
    batch_size: int,
    learn_rate: float,
    epochs: int,
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
        },
        "data": {
            "path": dataset._train.path.as_posix(),
            "date": {
                "start": dataset.acquisition.start_date,
                "end": dataset.acquisition.end_date,
            },
            "footprint": {
                "shape": dataset.footprint,
                "spacing": dataset.spacing,
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
