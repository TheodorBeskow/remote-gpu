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
@click.option("--detach", is_flag=True, help="Push and return immediately — use `logs`/`pull` later")
def run(script, cpu, detach):
    """Run a .py or .ipynb on Kaggle and pull the results back."""
    from .kaggle_manager import KaggleRunner

    config = load_config(Path(script).parent)
    if cpu:
        config.kaggle.gpu_enabled = False

    runner = KaggleRunner(config)
    if detach:
        runner.launch(Path(script))
    else:
        runner.run(Path(script))


@main.command()
@click.option("--all", "show_all", is_flag=True, help="Full log instead of last 100 entries")
@click.option("--follow", is_flag=True, help="Keep streaming until the run finishes")
@click.option("--save", type=click.Path(), help="Also append output to this file")
def logs(show_all, follow, save):
    """Show the latest kernel's log."""
    from .kaggle_manager import KaggleRunner

    KaggleRunner(load_config()).stream_logs(show_all, follow, save)


@main.command()
def pull():
    """Download output once the latest kernel is done."""
    from .kaggle_manager import KaggleRunner

    KaggleRunner(load_config()).pull()


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
