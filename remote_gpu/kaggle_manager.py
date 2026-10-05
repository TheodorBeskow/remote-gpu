"""Push code to a Kaggle kernel, wait for it, pull results back.

Reuses ONE dataset (the input folder) and ONE kernel per project —
no spam, no clutter on your Kaggle account.
"""

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path, PurePosixPath

import click

from .config import Config
from .kaggleify import kaggleify_notebook, kaggleify_script

POLL_INTERVAL = 15  # seconds between status checks
MAX_STATUS_FAILURES = 3
MAX_DOWNLOAD_ATTEMPTS = 3
STALE_OUTPUT_NOTICE = "Kaggle may return files from an earlier run - verify downloaded output"
NO_LIVE_LOGS_NOTICE = (
    "No logs available yet. Kaggle may not expose live cell output while running; "
    "RUNNING does not confirm progress. Try `remote-gpu logs --follow` with "
    "Kaggle CLI 2.2.3+, check the Kaggle UI, or write checkpoint files for "
    "retrieval after the run stops"
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
    return subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


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
        r"\b(cancel_acknowledged|complete|error|failed|failure|cancelled|canceled|cancel|timeout|timed out|timed_out)\b",
        output.lower(),
    )
    return match.group(1) if match else None


def _transient_transfer_error(detail: str) -> bool:
    text = detail.lower()
    if "no output" in text or "no files" in text:
        return False
    markers = (
        "403", "429", "500", "502", "503", "504", "connection", "disconnected",
        "dns", "eof", "network", "remote end closed", "reset", "temporarily",
        "timed out", "timeout",
    )
    return any(marker in text for marker in markers)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


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
            if status == "cancel_acknowledged" and downloaded:
                outcome += f"; {STALE_OUTPUT_NOTICE}"
            raise click.ClickException(f"kernel {status}; {outcome}")

    def launch(self, entry: Path) -> None:
        """Sync dataset + push kernel, then return (kernel runs async)."""
        self._sync_dataset(entry)
        self._push_kernel(entry)
        state = self._load_state()
        state["last_kernel"] = self.kernel_slug
        state["entry_dir"] = str(entry.parent.resolve())
        state["download_run_id"] = uuid.uuid4().hex
        state["download_kernel"] = self.kernel_slug
        self._save_state(state)
        click.echo(f"running: https://kaggle.com/code/{self.kernel_slug}")

    @property
    def _state_file(self) -> Path:
        return self.config.project_dir / ".remote-gpu" / "state.json"

    def _temp_dir(self):
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=self._state_file.parent)

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
        with self._temp_dir() as tmp:
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
        with self._temp_dir() as tmp:
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
        status_failures = 0
        try:
            while True:
                seen = self._print_new_logs(seen)
                res = _kaggle("kernels", "status", self.kernel_slug)
                if res.returncode != 0:
                    status_failures += 1
                    detail = res.stderr.strip() or res.stdout.strip() or "no details provided"
                    if status_failures >= MAX_STATUS_FAILURES:
                        raise click.ClickException(f"kernel status failed: {detail}")
                    delay = min(5 * status_failures, 30)
                    click.echo(
                        f"kernel status failed ({detail}); retrying in {delay}s "
                        f"({status_failures}/{MAX_STATUS_FAILURES})...",
                        err=True,
                    )
                    time.sleep(delay)
                    continue
                status_failures = 0
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
        if follow:
            help_result = _kaggle("kernels", "logs", "--help")
            if help_result.returncode == 0 and "--follow" in help_result.stdout:
                self._stream_live_logs(save)
                return
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

    def _stream_live_logs(self, save: str | None) -> None:
        fh = Path(save).open("a", encoding="utf-8", newline="") if save else None
        try:
            process = subprocess.Popen(
                ["kaggle", "kernels", "logs", "-f", self.kernel_slug],
                stdout=subprocess.PIPE, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="")
            received = False
            try:
                while chunk := stream.read(1):
                    received = True
                    sys.stdout.write(chunk)
                    if fh:
                        fh.write(chunk)
                    if chunk in "\r\n":
                        sys.stdout.flush()
                        if fh:
                            fh.flush()
                sys.stdout.flush()
                if process.wait() != 0:
                    raise click.ClickException("Kaggle live logs failed; check Kaggle CLI output")
                if not received:
                    click.echo("No logs available from Kaggle for this run", err=True)
            except KeyboardInterrupt:
                pass
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                stream.close()
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
            if status == "cancel_acknowledged":
                click.echo(STALE_OUTPUT_NOTICE, err=True)

    def _output_stage(self) -> Path:
        state = self._load_state()
        if state.get("download_kernel") != self.kernel_slug or not state.get("download_run_id"):
            state["download_run_id"] = uuid.uuid4().hex
            state["download_kernel"] = self.kernel_slug
            self._save_state(state)
        run_id = state["download_run_id"]
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise click.ClickException("invalid download run ID in remote-gpu state")
        stage = self.config.project_dir / ".remote-gpu" / "downloads" / run_id
        stage.mkdir(parents=True, exist_ok=True)
        return stage

    def _download_output(self, stage: Path) -> tuple[int, deque, int, int]:
        process = subprocess.Popen(
            ["kaggle", "kernels", "output", self.kernel_slug, "-p", str(stage), "--page-size", "200"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )
        recent = deque(maxlen=10)
        downloaded = downloaded_bytes = 0
        started = time.monotonic()
        stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace")
        try:
            for line in stream:
                marker = "Output file downloaded to "
                if marker in line:
                    downloaded += 1
                    try:
                        downloaded_bytes += Path(line.split(marker, 1)[1].strip()).stat().st_size
                    except OSError:
                        pass
                    if downloaded == 1 or downloaded % 100 == 0:
                        elapsed = time.monotonic() - started
                        click.echo(
                            f"downloaded {downloaded} file{'s' if downloaded != 1 else ''} "
                            f"({_format_bytes(downloaded_bytes)}) in {elapsed:.0f}s..."
                        )
                elif line.strip():
                    recent.append(line.strip())
            return process.wait(), recent, downloaded, downloaded_bytes
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            stream.close()

    def _fetch_output(self, entry_dir: Path) -> bool:
        stage = self._output_stage()
        click.echo(f"downloading output to {stage} (reuses existing files on retry)...")
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                code, recent, _, _ = self._download_output(stage)
            except KeyboardInterrupt as exc:
                raise click.ClickException(
                    f"output download interrupted; staged files retained at {stage}; retry with `remote-gpu pull`"
                ) from exc
            if code == 0:
                break
            detail = "; ".join(recent) or "no details provided"
            if attempt < MAX_DOWNLOAD_ATTEMPTS and _transient_transfer_error(detail):
                delay = min(5 * attempt, 30)
                click.echo(
                    f"download attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS} failed ({detail}); "
                    f"retrying in {delay}s...",
                    err=True,
                )
                time.sleep(delay)
                continue
            raise click.ClickException(
                f"Kaggle output download failed after {attempt} attempt{'s' if attempt != 1 else ''}: "
                f"{detail}; staged files retained at {stage}; retry with `remote-gpu pull`"
            )

        files = [path for path in stage.rglob("*") if path.is_file()]
        if not files:
            click.echo("Kaggle returned no output files")
            try:
                stage.rmdir()
            except OSError:
                pass
            return False
        staged_bytes = sum(path.stat().st_size for path in files)
        click.echo(
            f"download finished; {len(files)} file{'s' if len(files) != 1 else ''} "
            f"({_format_bytes(staged_bytes)}) staged; copying output..."
        )
        out_dir = entry_dir / _relpath(self.config.paths.local_output)

        # Unwrap the kaggle working dir: files written to ./output on
        # Kaggle land in stage/<output>/ — merge its contents back, and
        # drop stray top-level files (logs, executed nb) alongside them.
        try:
            output_root = PurePosixPath(self.config.paths.kaggle_output).relative_to("/kaggle/working")
        except ValueError as exc:
            raise click.UsageError(
                "paths.kaggle_output must be within /kaggle/working to download results"
            ) from exc
        inner = stage / str(output_root) / _relpath(self.config.paths.local_output)
        ancestor = inner.relative_to(stage).parts[:1] if inner.is_dir() else ()
        copied = skipped = copied_bytes = 0
        try:
            for source in files:
                if source.is_symlink():
                    raise click.ClickException(f"refusing symlink in staged output: {source}")
                if inner in source.parents:
                    relative = source.relative_to(inner)
                else:
                    relative = source.relative_to(stage)
                    if ancestor and relative.parts[:1] == ancestor:
                        continue
                destination = out_dir / relative
                if destination.is_file() and destination.stat().st_size == source.stat().st_size and destination.stat().st_mtime_ns == source.stat().st_mtime_ns:
                    skipped += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied += 1
                copied_bytes += source.stat().st_size
                if copied % 100 == 0:
                    click.echo(f"copied {copied} files ({_format_bytes(copied_bytes)}) to {out_dir}...")
        except OSError as exc:
            raise click.ClickException(
                f"output copy failed after {copied} files; staged files retained at {stage}; retry with `remote-gpu pull`: {exc}"
            ) from exc
        click.echo(
            f"transfer complete: {copied} copied ({_format_bytes(copied_bytes)}), "
            f"{skipped} unchanged; output -> {out_dir}"
        )
        self._remove_stage_if_safe(stage, out_dir)
        return True

    @staticmethod
    def _remove_stage_if_safe(stage: Path, out_dir: Path) -> None:
        try:
            resolved_stage = stage.resolve()
            resolved_output = out_dir.resolve()
            if resolved_stage == resolved_output or resolved_stage in resolved_output.parents:
                click.echo(f"keeping staged files because output is inside the stage: {stage}", err=True)
                return
            shutil.rmtree(resolved_stage)
        except OSError as exc:
            click.echo(f"warning: could not remove staged files at {stage}: {exc}", err=True)


def _relpath(p: str) -> str:
    return p.replace("\\", "/").removeprefix("./").rstrip("/")
