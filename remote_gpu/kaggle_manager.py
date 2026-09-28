"""Push code to a Kaggle kernel, wait for it, pull results back.

Reuses ONE dataset (the input folder) and ONE kernel per project —
no spam, no clutter on your Kaggle account.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import click

from .config import Config
from .kaggleify import kaggleify_notebook, kaggleify_script

POLL_INTERVAL = 15  # seconds between status checks


def _username(config: Config) -> str:
    """Kaggle username: env -> kaggle.json -> credentials.json (OAuth) -> yaml."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for name in ("kaggle.json", "credentials.json"):
        creds = Path.home() / ".kaggle" / name
        if creds.is_file():
            data = json.loads(creds.read_text(encoding="utf-8"))
            if data.get("username"):
                return data["username"]
    if config.kaggle.user:
        return config.kaggle.user
    raise click.UsageError(
        "Kaggle username not found. Run `kaggle auth login`, create "
        "~/.kaggle/kaggle.json, or add `user:` under `kaggle:` in your yaml."
    )


def _kaggle(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def _hash_dir(directory: Path) -> str:
    """sha256 over all files' relative paths + contents."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(directory)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


class KaggleRunner:
    def __init__(self, config: Config):
        self.config = config
        self.user = _username(config)
        self.dataset_slug = f"{self.user}/{config.kaggle.dataset_name}"
        self.kernel_slug = f"{self.user}/{config.kaggle.notebook_name}"

    def run(self, entry: Path) -> None:
        self._sync_dataset(entry)
        self._push_kernel(entry)
        self._wait()
        self._fetch_output(entry)

    @property
    def _state_file(self) -> Path:
        return self.config.project_dir / ".remote-gpu" / "state.json"

    def _load_state(self) -> dict:
        if self._state_file.is_file():
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        return {}

    def _save_state(self, state: dict) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(state), encoding="utf-8")

    def _dataset_exists(self) -> bool:
        res = _kaggle("datasets", "status", self.dataset_slug)
        return res.returncode == 0 and "not found" not in res.stderr.lower()

    def _sync_dataset(self, entry: Path) -> None:
        """Upload input/ as (or version onto) our single dataset."""
        input_dir = entry.parent / _relpath(self.config.paths.local_input)
        if not input_dir.is_dir():
            click.echo("no input dir — skipping dataset upload")
            return

        digest = _hash_dir(input_dir)
        state = self._load_state()
        exists = self._dataset_exists()

        if digest == state.get("dataset_hash") and exists:
            click.echo("input unchanged — skipping dataset upload")
            return

        click.echo("syncing dataset...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copytree(input_dir, tmp / "data")
            (tmp / "data" / "dataset-metadata.json").write_text(
                json.dumps(
                    {
                        "id": self.dataset_slug,
                        "title": self.config.kaggle.dataset_name,
                        "licenses": [{"name": "CC0-1.0"}],
                    }
                ),
                encoding="utf-8",
            )

            cmd = ["datasets", "version" if exists else "create",
                   "-p", str(tmp / "data")]
            if exists:
                cmd += ["-m", "remote-gpu sync"]
            res = _kaggle(*cmd)
            if res.returncode != 0:
                raise click.ClickException(f"dataset sync failed: {res.stderr}")

        # New/updated datasets need processing time before kernels can mount them
        self._wait_dataset_ready()

        state["dataset_hash"] = digest
        self._save_state(state)

    def _wait_dataset_ready(self) -> None:
        for _ in range(40):
            res = _kaggle("datasets", "status", self.dataset_slug)
            out = res.stdout.lower()
            if "ready" in out:
                return
            if "error" in out or "failed" in out:
                raise click.ClickException(f"dataset processing failed: {res.stdout}")
            time.sleep(3)
        raise click.ClickException("dataset still processing after 2 minutes")

    def _push_kernel(self, entry: Path) -> None:
        """Write the kaggleified notebook + metadata, then push."""
        click.echo("pushing kernel...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            is_nb = entry.suffix == ".ipynb"
            code_name = f"solve{entry.suffix}"

            if is_nb:
                kaggleify_notebook(entry, tmp / code_name, self.config)
            else:
                kaggleify_script(entry, tmp / code_name, self.config)

            meta = {
                "id": self.kernel_slug,
                "title": self.config.kaggle.notebook_name,
                "code_file": code_name,
                "language": "python",
                "kernel_type": "notebook" if is_nb else "script",
                "is_private": True,
                "enable_gpu": self.config.kaggle.gpu_enabled,
                "enable_internet": False,
                "dataset_sources": [self.dataset_slug],
            }
            (tmp / "kernel-metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )

            res = _kaggle("kernels", "push", "-p", str(tmp))
            if res.returncode != 0:
                raise click.ClickException(f"kernel push failed: {res.stderr}")

    def _wait(self) -> None:
        click.echo("running on Kaggle (polling every "
                   f"{POLL_INTERVAL}s, Ctrl+C to stop watching)...")
        while True:
            res = _kaggle("kernels", "status", self.kernel_slug)
            out = (res.stdout + res.stderr).lower()
            if "complete" in out:
                click.echo("done")
                return
            if "error" in out or "cancel" in out:
                raise click.ClickException(
                    f"kernel failed — check https://kaggle.com/code/{self.kernel_slug}"
                )
            time.sleep(POLL_INTERVAL)

    def _fetch_output(self, entry: Path) -> None:
        click.echo("downloading output...")
        with tempfile.TemporaryDirectory() as tmp:
            res = _kaggle("kernels", "output", self.kernel_slug, "-p", tmp)
            if res.returncode != 0:
                click.echo(f"warning: download failed: {res.stderr}")
                return

            out_dir = entry.parent / _relpath(self.config.paths.local_output)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Unwrap the kaggle working dir: files written to ./output on
            # Kaggle land in tmp/<output>/ — merge its contents back, and
            # drop stray top-level files (logs, executed nb) alongside them.
            inner = Path(tmp) / _relpath(self.config.paths.local_output)
            for item in Path(tmp).iterdir():
                if item == inner:
                    shutil.copytree(item, out_dir, dirs_exist_ok=True)
                elif item.is_dir():
                    shutil.copytree(item, out_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, out_dir / item.name)

        click.echo(f"output -> {out_dir}")


def _relpath(p: str) -> str:
    return p.replace("\\", "/").removeprefix("./").rstrip("/")
