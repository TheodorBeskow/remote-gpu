"""CLI for remote-gpu."""

import json
from pathlib import Path

import click

from .config import load_config


@click.group()
def main():
    """Run Python on Kaggle GPUs from your local machine."""
    pass


@main.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("--cpu", is_flag=True, help="Run on Kaggle CPU (doesn't use GPU quota)")
def run(script, cpu):
    """Run a .py or .ipynb on Kaggle and pull the results back."""
    from .kaggle_manager import KaggleRunner

    config = load_config(Path(script).parent)
    if cpu:
        config.kaggle.gpu_enabled = False

    KaggleRunner(config).run(Path(script))


@main.command()
def status():
    """Show Kaggle auth + quota info."""
    for name in ("kaggle.json", "credentials.json"):
        creds = Path.home() / ".kaggle" / name
        if creds.is_file():
            user = json.loads(creds.read_text()).get("username", "?")
            click.echo(f"authenticated: {user}")
            break
    else:
        click.echo("not authenticated — run `remote-gpu setup`")
    click.echo("quota tracking: TODO")


@main.command()
def setup():
    """Verify Kaggle credentials are in place."""
    import os

    if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_USERNAME"):
        click.echo("authenticated via environment variables")
        return

    for name in ("kaggle.json", "credentials.json"):
        creds = Path.home() / ".kaggle" / name
        if creds.is_file():
            user = json.loads(creds.read_text()).get("username", "?")
            click.echo(f"authenticated as {user} ({name})")
            return

    click.echo("Not authenticated. Either:")
    click.echo("  1. Run `kaggle auth login` (browser OAuth), or")
    click.echo("  2. kaggle.com -> Settings -> API -> Create New Token,")
    click.echo("     place kaggle.json in ~/.kaggle/")


if __name__ == "__main__":
    main()
