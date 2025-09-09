import click

from naip_cnn.utils.wandb import compare_runs

from .check import check_data_split
from .predict import predict
from .train import train
from .validate import validate


@click.group()
def cli():
    pass


@cli.command(name="predict")
@click.argument("run_path")
@click.argument("dataset_id")
@click.argument("year", type=int)
@click.option("--batch-size", type=int, default=256)
@click.option("--apply-mask", is_flag=True)
def predict_cmd(
    run_path: str, dataset_id: str, year: int, batch_size: int, apply_mask: bool
) -> None:
    """
    Predict on a dataset using a W&B run.

    This requires that a dataset with the given ID and year has already been saved to
    the project data directory in TFRecord format.
    """
    path = predict(run_path, dataset_id, year, batch_size, apply_mask=apply_mask)
    click.echo(f"Saved prediction to {path}")


@cli.command(name="compare")
@click.argument("run_path")
@click.argument("other_path")
def compare_cmd(run_path: str, other_path: str) -> None:
    """Compare the configuration of two W&B runs."""
    compare_runs(run_path, other_path)


@cli.command(name="train")
@click.option(
    "--allow-duplicate", is_flag=True, help="Allow duplicate run configurations."
)
@click.option(
    "--allow-cpu", is_flag=True, help="Allow training on CPU if GPU is unavailable."
)
@click.option("--dry-run", is_flag=True, help="Print a summary without training.")
@click.option("--debug", is_flag=True, help="Train a test model without W&B logging.")
def train_cmd(allow_duplicate: bool, allow_cpu: bool, dry_run: bool, debug: bool):
    """Train a new model and log it to W&B."""
    train(
        allow_duplicate=allow_duplicate,
        allow_cpu=allow_cpu,
        dry_run=dry_run,
        debug=debug,
    )


@cli.command(name="check")
@click.argument("dataset")
def check_cmd(dataset: str) -> None:
    """
    Check for duplicate images in a split [DATASET].

    For example:
    $ python -m src.naip_cnn.cli check MAL2016_CanyonCreek-1-30-30x30-30
    """
    check_data_split(dataset)


@cli.command(name="validate")
@click.argument("run_path")
@click.argument("acquisition_name")
@click.option("--batch-size", type=int, default=256)
@click.option("--dry-run", is_flag=True)
def validation_command(
    run_path: str, acquisition_name: str, batch_size: int = 256, dry_run: bool = False
):
    """
    Validate a model using a given validation dataset.
    """
    validate(
        run_path,
        acquisition_name=acquisition_name,
        batch_size=batch_size,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    cli()
