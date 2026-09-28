"""Push code to a Kaggle kernel, wait for it, pull results back.

Reuses ONE dataset (the input folder) and ONE kernel per project —
no spam, no clutter on your Kaggle account.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
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
        self.launch(entry)
        if self._wait():
            self._fetch_output(entry.parent)

    def launch(self, entry: Path) -> None:
        """Sync dataset + push kernel, then return (kernel runs async)."""
        self._sync_dataset(entry)
        self._push_kernel(entry)
        state = self._load_state()
        state["last_kernel"] = self.kernel_slug
        state["entry_dir"] = str(entry.parent.resolve())
        self._save_state(state)
        click.echo(f"running: https://kaggle.com/code/{self.kernel_slug}")

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

    def _fetch_log_entries(self) -> list:
        res = _kaggle("kernels", "logs", self.kernel_slug)
        if res.returncode != 0:
            return []
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return []

    def _wait(self) -> bool:
        """Poll + stream logs until the kernel finishes. False if detached."""
        click.echo(f"watching (Ctrl+C detaches — kernel keeps running)...")
        seen = 0
        try:
            while True:
                seen = self._print_new_logs(seen)
                out = _kaggle("kernels", "status",
                              self.kernel_slug).stdout.lower()
                if "complete" in out:
                    self._print_new_logs(seen)  # flush tail
                    click.echo("done")
                    return True
                if "error" in out or "cancel" in out:
                    self._print_new_logs(seen)
                    raise click.ClickException(
                        f"kernel failed — https://kaggle.com/code/{self.kernel_slug}"
                    )
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            click.echo(
                f"\ndetached — kernel still running: "
                f"https://kaggle.com/code/{self.kernel_slug}\n"
                f"resume: `remote-gpu logs --follow` / download: `remote-gpu pull`"
            )
            return False

    def _print_new_logs(self, seen: int) -> int:
        entries = self._fetch_log_entries()
        for e in entries[seen:]:
            sys.stdout.write(e.get("data", ""))
        sys.stdout.flush()
        return len(entries)

    def stream_logs(self, show_all: bool, follow: bool, save: str | None) -> None:
        """`remote-gpu logs`: print/follow the latest kernel's log."""
        fh = Path(save).open("a", encoding="utf-8") if save else None
        seen = None
        try:
            while True:
                entries = self._fetch_log_entries()
                start = 0 if show_all else max(0, len(entries) - 100)
                if seen is not None:
                    start = seen
                for e in entries[start:]:
                    data = e.get("data", "")
                    sys.stdout.write(data)
                    if fh:
                        fh.write(data)
                sys.stdout.flush()
                seen = len(entries)
                if not follow:
                    return
                out = _kaggle("kernels", "status",
                              self.kernel_slug).stdout.lower()
                if "complete" in out or "error" in out or "cancel" in out:
                    return
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            pass
        finally:
            if fh:
                fh.close()

    def pull(self) -> None:
        """`remote-gpu pull`: download output once the kernel is done."""
        out = _kaggle("kernels", "status", self.kernel_slug).stdout.lower()
        if "complete" not in out:
            click.echo(f"not done yet (status: {out.strip()})")
            return
        entry_dir = Path(self._load_state().get("entry_dir", "."))
        self._fetch_output(entry_dir)

    def _fetch_output(self, entry_dir: Path) -> None:
        click.echo("downloading output...")
        with tempfile.TemporaryDirectory() as tmp:
            res = _kaggle("kernels", "output", self.kernel_slug, "-p", tmp)
            if res.returncode != 0:
                click.echo(f"warning: download failed: {res.stderr}")
                return

            out_dir = entry_dir / _relpath(self.config.paths.local_output)
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
