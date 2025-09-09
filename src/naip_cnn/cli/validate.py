from __future__ import annotations

import tensorflow as tf

import wandb
from naip_cnn.acquisitions import Acquisition
from naip_cnn.data import NAIPDatasetWrapper
from naip_cnn.models import ModelRun


def validate(
    run_path: str,
    acquisition_name: str,
    batch_size: int = 256,
    dry_run: bool = False,
) -> dict[str, float]:
    """
    Validate a model using a given validation dataset.

    Parameters
    ----------
    run_path : str
        The path to the W&B run to validate, e.g. "team/project/run_id".
    acquisition_name : str
        The name of the acquisition to use for validation.
    batch_size : int, default 256
        The batch size to use for validation.
    dry_run : bool, default False
        If True, print the validation results without updating W&B.

    Returns
    -------
    dict[str, float]
        A dictionary of metric names and their values.
    """
    run = wandb.Api().run(run_path)

    model_run = ModelRun.from_wandb_run(run_path)

    cfg = run.config
    model_run = ModelRun.from_wandb_run(run_path)
    footprint = cfg["data"]["footprint"]["shape"]
    spacing = cfg["data"]["footprint"]["spacing"]
    lidar_res = cfg["data"]["lidar"]["resolution"]
    naip_res = cfg["data"]["imagery"]["resolution"]
    label = cfg["data"]["lidar"]["label"]

    val_acquisition = Acquisition.from_name(acquisition_name)
    wrapper = NAIPDatasetWrapper(
        acquisitions=[val_acquisition],
        naip_res=naip_res,
        lidar_res=lidar_res,
        footprint=footprint,
        spacing=spacing,
    )

    all_bands = cfg["data"]["imagery"]["bands"].split("-")
    # Split bands into optical bands R, G, B, N and vegetation indices
    opt_bands = ["R", "G", "B", "N"]
    veg_indices = [b for b in all_bands if b not in opt_bands]
    bands = [b for b in all_bands if b in opt_bands]

    val = (
        wrapper.load_val(label=label, bands=bands, veg_indices=veg_indices)
        .cache()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    metric_vals = model_run.model.evaluate(val)
    results = dict(zip(model_run.model.metrics_names, metric_vals))

    if dry_run:
        print(results)
        return results

    def convert_nested_dict(d):
        if hasattr(d, "items"):
            return {k: convert_nested_dict(v) for k, v in d.items()}

        return d

    # Append new results to the existing validation results. Note that wandb uses a
    # dict-like object for every nested dictionary in the summary, which causes issues
    # when updating. We need to convert each dict-like to a regular dict, and then let
    # wandb handle the conversion back to their dict-like when we update the run.
    prev_validation = convert_nested_dict(run.summary.get("validation", {}))
    prev_validation[acquisition_name] = results
    run.summary["validation"] = prev_validation
    run.update()
    print(f"Updated validation results for {run.name}")
    return results
