"""Push code to a Kaggle kernel, wait for it, pull results back.

Reuses ONE dataset (the input folder) and ONE kernel per project —
no spam, no clutter on your Kaggle account.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import click

from .config import Config
from .kaggleify import kaggleify_notebook, kaggleify_script

POLL_INTERVAL = 15  # seconds between status checks
NO_LIVE_LOGS_NOTICE = (
    "No logs available yet. Kaggle may not expose live cell output while running; "
    "RUNNING does not confirm progress. Check the Kaggle UI or write progress "
    "to checkpoint files for retrieval after the run stops"
)


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


def _terminal_status(output: str) -> str | None:
    match = re.search(
        r"\b(complete|error|failed|failure|cancelled|canceled|cancel|timeout|timed out|timed_out)\b",
        output,
    )
    return match.group(1) if match else None


class KaggleRunner:
    def __init__(self, config: Config):
        self.config = config
        self.user = _username(config)
        self.dataset_slug = f"{self.user}/{config.kaggle.dataset_name}"
        self.kernel_slug = f"{self.user}/{config.kaggle.notebook_name}"
        self._managed_exists = False

    def run(self, entry: Path) -> None:
        self.launch(entry)
        status = self._wait()
        if status is None:
            return
        try:
            downloaded = self._fetch_output(entry.parent)
        except click.ClickException as exc:
            if status != "complete":
                raise click.ClickException(f"kernel {status}; {exc.message}") from exc
            raise
        if status != "complete":
            outcome = "downloaded available output" if downloaded else "Kaggle returned no output files"
            raise click.ClickException(f"kernel {status}; {outcome}")

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
            self._managed_exists = False
            click.echo("no input dir - skipping dataset upload")
            return

        digest = _hash_dir(input_dir)
        state = self._load_state()
        exists = self._dataset_exists()

        if digest == state.get("dataset_hash") and exists:
            self._managed_exists = True
            click.echo("input unchanged - skipping dataset upload")
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
        self._managed_exists = True

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

    def preview(self, entry: Path) -> None:
        """Print planned kernel metadata without syncing or pushing."""
        managed_input = (entry.parent / _relpath(self.config.paths.local_input)).is_dir()
        click.echo(json.dumps(self._kernel_metadata(entry, managed_input), indent=2))

    def _kernel_metadata(self, entry: Path, managed_input: bool) -> dict:
        if not PurePosixPath(self.config.paths.kaggle_output).is_relative_to(
            "/kaggle/working"
        ):
            raise click.UsageError(
                "paths.kaggle_output must be within /kaggle/working to download results"
            )
        return {
            "id": self.kernel_slug,
            "title": self.config.kaggle.notebook_name,
            "code_file": f"solve{entry.suffix}",
            "language": "python",
            "kernel_type": "notebook" if entry.suffix == ".ipynb" else "script",
            "is_private": True,
            "enable_gpu": self.config.kaggle.gpu_enabled,
            "enable_internet": self.config.kaggle.internet_enabled,
            "dataset_sources": [
                *self.config.datasets.values(),
                *([self.dataset_slug] if managed_input else []),
            ],
        }

    def _push_kernel(self, entry: Path) -> None:
        """Write the kaggleified notebook + metadata, then push."""
        meta = self._kernel_metadata(entry, self._managed_exists)
        click.echo("pushing kernel...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            code_name = meta["code_file"]

            if entry.suffix == ".ipynb":
                kaggleify_notebook(
                    entry, tmp / code_name, self.config, self._managed_exists, self.dataset_slug
                )
            else:
                kaggleify_script(
                    entry, tmp / code_name, self.config, self._managed_exists, self.dataset_slug
                )

            (tmp / "kernel-metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )

            res = _kaggle("kernels", "push", "-p", str(tmp))
            if res.returncode != 0:
                raise click.ClickException(f"kernel push failed: {res.stderr}")

    def _fetch_log_entries(self, strict: bool = False) -> list:
        res = _kaggle("kernels", "logs", self.kernel_slug)
        if res.returncode != 0:
            if strict:
                raise click.ClickException(
                    f"Kaggle logs unavailable: {res.stderr.strip() or res.stdout.strip() or 'no details provided'}"
                )
            return []
        if not res.stdout.strip():
            return []
        try:
            entries = json.loads(res.stdout)
        except json.JSONDecodeError as exc:
            if strict:
                raise click.ClickException("Kaggle logs response was not valid JSON") from exc
            return []
        if not isinstance(entries, list):
            if strict:
                raise click.ClickException("Kaggle logs response was not a list")
            return []
        return entries

    def _wait(self) -> str | None:
        """Poll + stream logs until the kernel finishes, or detach."""
        click.echo("watching (Ctrl+C detaches - kernel keeps running)...")
        seen = 0
        notice_shown = False
        try:
            while True:
                seen = self._print_new_logs(seen)
                res = _kaggle("kernels", "status", self.kernel_slug)
                if res.returncode != 0:
                    raise click.ClickException(f"kernel status failed: {res.stderr.strip()}")
                status = _terminal_status(res.stdout.lower())
                if status:
                    self._print_new_logs(seen)  # flush tail
                    if status == "complete":
                        click.echo("done")
                    return status
                if seen == 0 and not notice_shown and "running" in res.stdout.lower():
                    click.echo(NO_LIVE_LOGS_NOTICE, err=True)
                    notice_shown = True
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            click.echo(
                f"\ndetached - kernel still running: "
                f"https://kaggle.com/code/{self.kernel_slug}\n"
                f"resume: `remote-gpu logs --follow` / download: `remote-gpu pull`"
            )
            return None

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
        notice_shown = False
        try:
            while True:
                entries = self._fetch_log_entries(strict=True)
                status_res = None
                if not entries and not notice_shown:
                    status_res = _kaggle("kernels", "status", self.kernel_slug)
                    if status_res.returncode != 0:
                        message = "No logs available; unable to check kernel status"
                    elif _terminal_status(status_res.stdout.lower()):
                        message = "No logs available from Kaggle for this run"
                    else:
                        message = NO_LIVE_LOGS_NOTICE
                    click.echo(message, err=True)
                    notice_shown = True
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
                status_res = status_res or _kaggle("kernels", "status", self.kernel_slug)
                if status_res.returncode != 0:
                    raise click.ClickException(f"kernel status failed: {status_res.stderr.strip()}")
                if _terminal_status(status_res.stdout.lower()):
                    return
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            pass
        finally:
            if fh:
                fh.close()

    def pull(self) -> None:
        """`remote-gpu pull`: download available output after the kernel stops."""
        res = _kaggle("kernels", "status", self.kernel_slug)
        if res.returncode != 0:
            raise click.ClickException(f"kernel status failed: {res.stderr.strip()}")
        status = _terminal_status(res.stdout.lower())
        if not status:
            click.echo(f"not done yet (status: {res.stdout.strip()})")
            return
        entry_dir = Path(self._load_state().get("entry_dir", "."))
        downloaded = self._fetch_output(entry_dir)
        if status != "complete":
            if not downloaded:
                raise click.ClickException(f"kernel {status}; Kaggle returned no output files")
            click.echo(f"kernel {status}; downloaded available output")

    def _fetch_output(self, entry_dir: Path) -> bool:
        click.echo("downloading output...")
        with tempfile.TemporaryDirectory() as tmp:
            res = _kaggle("kernels", "output", self.kernel_slug, "-p", tmp)
            if res.returncode != 0:
                raise click.ClickException(
                    f"Kaggle output unavailable: {res.stderr.strip() or res.stdout.strip() or 'no details provided'}"
                )
            if not any(Path(tmp).iterdir()):
                click.echo("Kaggle returned no output files")
                return False

            out_dir = entry_dir / _relpath(self.config.paths.local_output)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Unwrap the kaggle working dir: files written to ./output on
            # Kaggle land in tmp/<output>/ — merge its contents back, and
            # drop stray top-level files (logs, executed nb) alongside them.
            try:
                output_root = PurePosixPath(self.config.paths.kaggle_output).relative_to(
                    "/kaggle/working"
                )
            except ValueError as exc:
                raise click.UsageError(
                    "paths.kaggle_output must be within /kaggle/working to download results"
                ) from exc
            inner = Path(tmp) / str(output_root) / _relpath(self.config.paths.local_output)
            if inner.is_dir():
                shutil.copytree(inner, out_dir, dirs_exist_ok=True)
            for item in Path(tmp).iterdir():
                if item == inner or item in inner.parents:
                    continue
                if item.is_dir():
                    shutil.copytree(item, out_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, out_dir / item.name)

        click.echo(f"output -> {out_dir}")
        return True


def _relpath(p: str) -> str:
    return p.replace("\\", "/").removeprefix("./").rstrip("/")
