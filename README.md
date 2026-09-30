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
  kaggle_input: "/kaggle/input"
  kaggle_output: "/kaggle/working"
kaggle:
  gpu_enabled: true
  internet_enabled: false

datasets:                      # attach existing Kaggle datasets (optional)
  iris: uciml/iris             # ./iris/ resolves to the mounted dataset
```

Set `kaggle.internet_enabled: true` for runs that download model weights (such as StreetCLIP), or attach pre-downloaded weights via `datasets`. `kaggle_input` is the dataset mount root; `kaggle_output` is the remote output root and must be inside `/kaggle/working` for results to be downloaded. If there is no local input directory, no managed input dataset is attached or mounted.

## Commands

```bash
remote-gpu run x.ipynb --detach   # push and return; kernel runs async
remote-gpu logs                   # tail of latest run's log
remote-gpu logs --follow          # stream until done
remote-gpu logs --save run.log    # also write to file
remote-gpu pull                   # download output after detach
```

## How it works

- `input/` is synced to a private dataset `you/remote-gpu-<name>-data`
  (skipped when unchanged)
- Your notebook gets an injected preamble that symlinks `input` → the mounted
  dataset and creates `output`, then is pushed to kernel `you/remote-gpu-<name>`
- When the run finishes, everything written to `/kaggle/working` is downloaded
  back into `output/`
