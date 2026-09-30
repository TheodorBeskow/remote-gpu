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


def build_preamble(config: Config) -> str:
    """Generate the injected cell. No-ops everywhere except on Kaggle."""
    local_input = _relpath(config.paths.local_input)
    # managed dataset (synced input/) + any user-attached datasets —
    # values are dataset slugs we glob for under /kaggle/input/**/
    mounts = {local_input: config.kaggle.dataset_name}
    mounts.update({name: slug.split("/")[-1] for name, slug in config.datasets.items()})
    lines = [
        PREAMBLE_MARKER,
        "import os, glob",
        'if os.path.exists("/kaggle"):',
        # Kaggle mounts datasets under /kaggle/input/datasets/<owner>/<slug>/
        # when attached via API, or /kaggle/input/<slug>/ via the web editor —
        # resolve by globbing instead of hardcoding a convention.
        f"    _mounts = {mounts!r}",
        "    for _rel, _slug in _mounts.items():",
        "        if not os.path.exists(_rel):",
        '            _hits = glob.glob(f"/kaggle/input/**/{_slug}", recursive=True)',
        "            if _hits:",
        '                _parent = os.path.dirname(_rel)',
        "                if _parent:",
        "                    os.makedirs(_parent, exist_ok=True)",
        "                os.symlink(_hits[0], _rel)",
        '                print(f"[remote-gpu] {_rel} -> {_hits[0]}")',
        "            else:",
        '                print(f"[remote-gpu] WARNING: dataset {_slug} not mounted")',
        '    print("[remote-gpu] /kaggle/input:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "MISSING")',
        f'    os.makedirs("{_relpath(config.paths.local_output)}", exist_ok=True)',
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


def kaggleify_notebook(src: Path, dst: Path, config: Config) -> Path:
    """Copy `src` notebook to `dst` with the preamble cell injected."""
    nb = json.loads(src.read_text(encoding="utf-8"))

    # Don't double-inject
    already = any(
        PREAMBLE_MARKER in "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
    )
    if not already:
        nb.setdefault("cells", []).insert(0, make_cell(build_preamble(config)))

    # ensure_ascii=True escapes non-ASCII chars — the file becomes pure ASCII,
    # immune to wrong-codepage reads during upload (Windows cp1252 mangling).
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="ascii")
    return dst


def kaggleify_script(src: Path, dst: Path, config: Config) -> Path:
    """Copy `src` python file to `dst` with the preamble prepended."""
    code = src.read_text(encoding="utf-8")
    if PREAMBLE_MARKER not in code:
        code = build_preamble(config) + "\n\n" + code
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
