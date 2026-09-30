"""Transform local-path code into a Kaggle-runnable version.

Instead of rewriting path strings inside user code (fragile), we inject a
preamble that makes the local paths exist on Kaggle via symlinks/mkdirs.
Every spelling of a relative path then resolves correctly.
"""

import json
import os
import sys
from pathlib import Path, PurePosixPath

from .config import Config, load_config

PREAMBLE_MARKER = "# [remote-gpu] injected path setup"


def _relpath(p: str) -> str:
    """'./input' -> 'input', 'data/train' -> 'data/train' (posix-style)."""
    return PurePosixPath(p.replace("\\", "/")).as_posix().removeprefix("./")


def build_preamble(
    config: Config, managed_input: bool = False, managed_dataset: str | None = None
) -> str:
    """Generate the injected cell. No-ops everywhere except on Kaggle."""
    local_input = _relpath(config.paths.local_input)
    local_output = _relpath(config.paths.local_output)
    kaggle_input = config.paths.kaggle_input.rstrip("/")
    kaggle_output = str(PurePosixPath(config.paths.kaggle_output) / local_output)
    # managed dataset (synced input/) + any user-attached datasets —
    # values are dataset slugs we glob for under /kaggle/input/**/
    if managed_dataset is None:
        managed_dataset = (
            f"{config.kaggle.user}/{config.kaggle.dataset_name}"
            if config.kaggle.user else config.kaggle.dataset_name
        )
    mounts = {local_input: managed_dataset} if managed_input else {}
    mounts.update(config.datasets)
    lines = [
        PREAMBLE_MARKER,
        "import os, glob, posixpath",
        'if os.path.exists("/kaggle"):',
        # Kaggle mounts datasets under /kaggle/input/datasets/<owner>/<slug>/
        # when attached via API, or /kaggle/input/<slug>/ via the web editor —
        # resolve by globbing instead of hardcoding a convention.
        f"    _mounts = {mounts!r}",
        f"    _input = {kaggle_input!r}",
        "    for _rel, _source in _mounts.items():",
        "        if not os.path.exists(_rel):",
        '            _slug = _source.rsplit("/", 1)[-1]',
        '            _hits = sorted(set(p for p in glob.glob(posixpath.join(_input, "**", glob.escape(_slug)), recursive=True) if os.path.isdir(p)))',
        '            _owner = _source.split("/", 1)[0] if "/" in _source else None',
        '            _owned = [p for p in _hits if posixpath.basename(posixpath.dirname(p)) == _owner] if _owner else []',
        '            _flat_sources = {source for source in _mounts.values() if source.rsplit("/", 1)[-1] == _slug}',
        "            if len(_owned) > 1 or (not _owned and _hits and (len(_hits) > 1 or len(_flat_sources) > 1)):",
        '                raise RuntimeError(f"[remote-gpu] ambiguous dataset {_source}: {_hits}")',
        '            _hit = _owned[0] if _owned else (_hits[0] if _hits and (not _owner or _hits[0] == posixpath.join(_input, _slug)) else None)',
        "            if _hit:",
        '                _parent = os.path.dirname(_rel)',
        "                if _parent:",
        "                    os.makedirs(_parent, exist_ok=True)",
        "                os.symlink(_hit, _rel)",
        '                print(f"[remote-gpu] {_rel} -> {_hit}")',
        "            else:",
        '                print(f"[remote-gpu] WARNING: dataset {_source} not mounted")',
        '    print("[remote-gpu]", _input + ":", os.listdir(_input) if os.path.exists(_input) else "MISSING")',
        f"    _output = {kaggle_output!r}",
        f"    _local_output = {local_output!r}",
        "    os.makedirs(_output, exist_ok=True)",
        "    if os.path.abspath(_local_output) != os.path.abspath(_output):",
        "        os.makedirs(os.path.dirname(_local_output) or '.', exist_ok=True)",
        "        if not os.path.lexists(_local_output):",
        "            os.symlink(_output, _local_output)",
        "    try:",
        "        import torch",
        '        print("[remote-gpu] GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")',
        "    except ImportError:",
        "        pass",
    ]
    return "\n".join(lines)


def make_cell(source: str) -> dict:
    """Wrap source code in a notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def kaggleify_notebook(
    src: Path, dst: Path, config: Config, managed_input: bool | None = None,
    managed_dataset: str | None = None,
) -> Path:
    """Copy `src` notebook to `dst` with the preamble cell injected."""
    nb = json.loads(src.read_text(encoding="utf-8"))
    if managed_input is None:
        managed_input = (src.parent / _relpath(config.paths.local_input)).is_dir()

    # Don't double-inject
    already = any(
        PREAMBLE_MARKER in "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
    )
    if not already:
        nb.setdefault("cells", []).insert(
            0, make_cell(build_preamble(config, managed_input, managed_dataset))
        )

    # ensure_ascii=True escapes non-ASCII chars — the file becomes pure ASCII,
    # immune to wrong-codepage reads during upload (Windows cp1252 mangling).
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="ascii")
    return dst


def kaggleify_script(
    src: Path, dst: Path, config: Config, managed_input: bool | None = None,
    managed_dataset: str | None = None,
) -> Path:
    """Copy `src` python file to `dst` with the preamble prepended."""
    code = src.read_text(encoding="utf-8")
    if managed_input is None:
        managed_input = (src.parent / _relpath(config.paths.local_input)).is_dir()
    if PREAMBLE_MARKER not in code:
        code = build_preamble(config, managed_input, managed_dataset) + "\n\n" + code
    dst.write_text(code, encoding="utf-8")
    return dst


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m remote_gpu.kaggleify <src> <dst>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = load_config()
    out = (
        kaggleify_notebook(src, dst, cfg)
        if src.suffix == ".ipynb"
        else kaggleify_script(src, dst, cfg)
    )
    print(f"wrote {out}")
