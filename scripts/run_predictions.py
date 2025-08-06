"""
Automate generating predictions for the combined and single-year cover and RH95 models.
"""

import wandb
from naip_cnn.cli.predict import predict
from naip_cnn.config import PRED_DIR
from naip_cnn.models import ModelRun

runs = {
    # cover - combined
    "aazuspan-team/naip-cnn/dmrpcb14": [2009, 2011, 2012, 2014, 2016, 2020, 2022],
    # cover - 2009
    "aazuspan-team/naip-cnn/fkx5bh62": [2009],
    # cover - 2016
    "aazuspan-team/naip-cnn/5bihh19b": [2016],
    # cover - 2020
    "aazuspan-team/naip-cnn/dx2qsyp5": [2020],
    # rh95 - combined
    "aazuspan-team/naip-cnn/dhuhb3d9": [2009, 2011, 2012, 2014, 2016, 2020, 2022],
    # rh95 - 2009
    "aazuspan-team/naip-cnn/ts0egdg6": [2009],
    # rh95 - 2016
    "aazuspan-team/naip-cnn/jsyizfiw": [2016],
    # rh95 - 2020
    "aazuspan-team/naip-cnn/s4a6mnwt": [2020],
}


if __name__ == "__main__":
    dataset_id = "STUDY_REGION"

    for run_path, year_list in runs.items():
        run = wandb.Api().run(run_path)
        model_run = ModelRun.from_wandb_run(run_path)

        for year in year_list:
            filename = f"{dataset_id}_{year}-{model_run.label}-{run.name}"
            pred_path = PRED_DIR / f"{filename}.tif"
            if pred_path.exists():
                print(f"Skipping {pred_path} as it already exists.")
                continue
            predict(run_path, dataset_id=dataset_id, year=year)
