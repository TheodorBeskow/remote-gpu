# remote-gpu

Run local Python scripts/notebooks on Kaggle's free GPUs, from your terminal.

## Install

```bash
pip install -e .
pip install kaggle
```

## Auth

```bash
kaggle auth login
```

## Usage

```bash
remote-gpu run path/to/script.ipynb          # GPU run
remote-gpu run path/to/script.ipynb --cpu   # CPU (no GPU quota used)
remote-gpu status
remote-gpu setup
```

## Project layout

```
my-project/
├── remote-gpu-settings.yaml
├── solve.ipynb          # your code, uses ./input and ./output
├── input/               # uploaded as a Kaggle dataset (deduped by hash)
└── output/              # results downloaded here
```

## remote-gpu-settings.yaml

```yaml
name: my-project          # optional, defaults to folder name
paths:                    # all optional, these are the defaults
  local_input: "./input"
  local_output: "./output"
kaggle:
  gpu_enabled: true
```

## How it works

- `input/` is synced to a private dataset `you/remote-gpu-<name>-data`
  (skipped when unchanged)
- Your notebook gets an injected preamble that symlinks `input` → the mounted
  dataset and creates `output`, then is pushed to kernel `you/remote-gpu-<name>`
- When the run finishes, everything written to `/kaggle/working` is downloaded
  back into `output/`
