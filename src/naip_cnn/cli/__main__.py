import click

from naip_cnn.utils.wandb import compare_runs

from .predict import predict
from .train import train


@click.group()
def cli():
    pass


@cli.command(name="predict")
@click.argument("run_path")
@click.argument("dataset_id")
@click.argument("year", type=int)
@click.option("--batch-size", type=int, default=256)
def predict_cmd(run_path: str, dataset_id: str, year: int, batch_size: int) -> None:
    """
    Predict on a dataset using a W&B run.

    This requires that a dataset with the given ID and year has already been saved to
    the project data directory in TFRecord format.
    """
    path = predict(run_path, dataset_id, year, batch_size)
    click.echo(f"Saved prediction to {path}")


@cli.command(name="compare")
@click.argument("run_path")
@click.argument("other_path")
def compare_cmd(run_path: str, other_path: str) -> None:
    """Compare the configuration of two W&B runs."""
    compare_runs(run_path, other_path)


@cli.command(name="train")
@click.option(
    "--allow-duplicate-runs", is_flag=True, help="Allow duplicate run configurations."
)
@click.option(
    "--allow-cpu", is_flag=True, help="Allow training on CPU if GPU is unavailable."
)
def train_cmd(allow_duplicate_runs: bool, allow_cpu: bool):
    """Train a new model and log it to W&B."""
    train(allow_duplicate_runs, allow_cpu)


if __name__ == "__main__":
    cli()
