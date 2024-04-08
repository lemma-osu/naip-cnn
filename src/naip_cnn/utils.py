from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from naip_cnn.data import NAIPDatasetWrapper
    from naip_cnn.models import ModelRun


def float_to_str(f: float) -> str:
    """Stringify a float for a filename."""
    f = float(f)
    if f.is_integer():
        return str(int(f))
    return str(f).replace(".", "p")


def build_wandb_config(
    *,
    dataset: NAIPDatasetWrapper,
    model_run: ModelRun,
    bands: tuple[str],
    label: str,
    batch_size: int,
    learn_rate: float,
    epochs: int
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
            "path": dataset.hdf_path.as_posix(),
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