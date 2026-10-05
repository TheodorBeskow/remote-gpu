"""CLI for remote-gpu."""

import json
import shutil
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
@click.option("--internet/--no-internet", default=None, help="Override Kaggle internet for this run")
@click.option("--dry-run", is_flag=True, help="Show kernel settings and datasets without pushing")
@click.option("--detach", is_flag=True, help="Push and return immediately - use `logs`/`pull` later")
def run(script, cpu, internet, dry_run, detach):
    """Run a .py or .ipynb on Kaggle and pull the results back.

    Looks for remote-gpu-settings.yaml or settings.yaml in the script's
    directory and parent directories. The nearer directory wins. If
    both exist in the same directory, remove one. local_input and
    local_output paths are relative to the script's directory.
    """
    from .kaggle_manager import KaggleRunner

    config = load_config(Path(script).parent)
    if cpu:
        config.kaggle.gpu_enabled = False
    if internet is not None:
        config.kaggle.internet_enabled = internet

    runner = KaggleRunner(config)
    if dry_run:
        runner.preview(Path(script))
    elif detach:
        runner.launch(Path(script))
    else:
        runner.run(Path(script))


@main.command()
@click.option("--all", "show_all", is_flag=True, help="Full log instead of last 100 entries (polling mode)")
@click.option("--follow", is_flag=True, help="Use Kaggle's live stream when available; otherwise poll until done")
@click.option("--save", type=click.Path(), help="Also append output to this file")
def logs(show_all, follow, save):
    """Show available kernel logs (Kaggle may not expose live cell output)."""
    from .kaggle_manager import KaggleRunner

    KaggleRunner(load_config()).stream_logs(show_all, follow, save)


@main.command()
def pull():
    """Download output once the latest kernel is done."""
    from .kaggle_manager import KaggleRunner

    KaggleRunner(load_config()).pull()


@main.command()
@click.option("--yes", is_flag=True, help="Delete download staging without asking")
def clean(yes):
    """Delete retained Kaggle output downloads under .remote-gpu/downloads."""
    downloads = load_config().project_dir / ".remote-gpu" / "downloads"
    if not downloads.exists():
        click.echo("no retained downloads")
        return
    if not yes:
        click.confirm(f"Delete all retained downloads under {downloads}?", abort=True)
    shutil.rmtree(downloads)
    click.echo(f"removed {downloads}")


@main.command()
def status():
    """Verify local Kaggle credentials and API access."""
    import os

    from .kaggle_manager import _kaggle

    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        for name in ("kaggle.json", "credentials.json"):
            creds = Path.home() / ".kaggle" / name
            if creds.is_file():
                user = json.loads(creds.read_text(encoding="utf-8")).get("username", "?")
                break

    res = _kaggle("datasets", "list", "--mine", "-p", "1", "--format", "csv")
    if res.returncode != 0:
        detail = res.stderr.strip() or res.stdout.strip() or "no details provided"
        source = f"credentials for {user}" if user else "no local credentials found"
        raise click.ClickException(f"Kaggle API check failed ({source}): {detail}")
    click.echo(f"authenticated: {user}" if user else "authenticated")
    click.echo("Kaggle API access: ok")
    click.echo("quota tracking: not implemented")


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
