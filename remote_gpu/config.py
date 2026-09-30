"""Load and manage remote-gpu configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml

CONFIG_FILENAME = "remote-gpu-settings.yaml"
CONFIG_FILENAMES = (CONFIG_FILENAME, "settings.yaml")

# Defaults used when the yaml omits a path
DEFAULT_LOCAL_INPUT = "./input"
DEFAULT_LOCAL_OUTPUT = "./output"
DEFAULT_KAGGLE_INPUT = "/kaggle/input"
DEFAULT_KAGGLE_OUTPUT = "/kaggle/working"


@dataclass
class Paths:
    local_input: str = DEFAULT_LOCAL_INPUT
    local_output: str = DEFAULT_LOCAL_OUTPUT
    kaggle_input: str = DEFAULT_KAGGLE_INPUT
    kaggle_output: str = DEFAULT_KAGGLE_OUTPUT


@dataclass
class KaggleConfig:
    user: str | None = None
    notebook_name: str = "remote-gpu-runner"
    dataset_name: str = "remote-gpu-data"
    gpu_enabled: bool = True
    internet_enabled: bool = False


@dataclass
class RuntimeConfig:
    auto_gpu_detect: bool = True


@dataclass
class Config:
    project_dir: Path
    name: str
    paths: Paths = field(default_factory=Paths)
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    datasets: dict = field(default_factory=dict)  # local dir name -> owner/slug


def find_config(start: Path | None = None) -> Path:
    """Search upward from `start` (default: cwd) for a config file."""
    start = start or Path.cwd()
    for directory in (start, *start.parents):
        matches = [directory / filename for filename in CONFIG_FILENAMES if (directory / filename).is_file()]
        if len(matches) > 1:
            raise click.UsageError(
                f"both {' and '.join(CONFIG_FILENAMES)} found in {directory}; keep only one"
            )
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {' or '.join(CONFIG_FILENAMES)} in {start} or any parent directory"
    )


def load_config(start: Path | None = None) -> Config:
    """Load a config file, applying defaults for missing keys."""
    config_path = find_config(start)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    name = raw.get("name") or config_path.parent.name
    paths_raw = raw.get("paths") or {}
    kaggle_raw = raw.get("kaggle") or {}
    runtime_raw = raw.get("runtime") or {}
    if "quota_warning_hours" in runtime_raw:
        click.echo(
            "warning: runtime.quota_warning_hours has no effect; quota warnings are not implemented",
            err=True,
        )

    return Config(
        project_dir=config_path.parent,
        name=name,
        paths=Paths(
            local_input=paths_raw.get("local_input", DEFAULT_LOCAL_INPUT),
            local_output=paths_raw.get("local_output", DEFAULT_LOCAL_OUTPUT),
            kaggle_input=paths_raw.get("kaggle_input", DEFAULT_KAGGLE_INPUT),
            kaggle_output=paths_raw.get("kaggle_output", DEFAULT_KAGGLE_OUTPUT),
        ),
        kaggle=KaggleConfig(
            user=kaggle_raw.get("user"),
            notebook_name=kaggle_raw.get("notebook_name", f"remote-gpu-{name}"),
            dataset_name=kaggle_raw.get("dataset_name", f"remote-gpu-{name}-data"),
            gpu_enabled=kaggle_raw.get("gpu_enabled", True),
            internet_enabled=kaggle_raw.get("internet_enabled", False),
        ),
        runtime=RuntimeConfig(
            auto_gpu_detect=runtime_raw.get("auto_gpu_detect", True),
        ),
        datasets=raw.get("datasets") or {},
    )
