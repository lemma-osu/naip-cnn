import click

from naip_cnn.utils.wandb import compare_runs

from .train import train


@click.group()
def cli():
    pass


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
