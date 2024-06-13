from pathlib import Path

import wandb
from naip_cnn.models import ModelRun


def predict(run_path: str, dataset_id: str, year: int, batch_size: int = 256) -> Path:
    run = wandb.Api().run(run_path)
    model_run = ModelRun.from_wandb_run(run_path)

    return model_run.predict(
        dataset_id=dataset_id,
        year=year,
        filename=f"{dataset_id}_{year}-{model_run.label}-{run.name}",
        batch_size=batch_size,
        verbose=1,
    )
