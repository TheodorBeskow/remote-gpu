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

## Configuration

Name the file `remote-gpu-settings.yaml` or `settings.yaml`. The search starts in the script's directory and moves upward. The nearest directory wins; if both names exist there, remote-gpu reports an error and asks you to keep only one. `local_input` and `local_output` are relative to the script's directory.

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

Set `kaggle.internet_enabled: true` for runs that download model weights (such as StreetCLIP), or attach pre-downloaded weights via `datasets`. Dataset mounts check the exact `<kaggle_input>/datasets/<owner>/<slug>` and `<kaggle_input>/<slug>` locations first, then perform a bounded, directory-only fallback instead of recursively traversing dataset contents. A flat mount is used only when unambiguous. `kaggle_input` is the dataset mount root; `kaggle_output` is the remote output root and must be inside `/kaggle/working` for results to be downloaded. If there is no local input directory, no managed input dataset is attached or mounted. `runtime.quota_warning_hours` is deprecated and has no effect; GPU quota warnings are not implemented.

## Commands

```bash
remote-gpu run x.ipynb --detach   # push and return; kernel runs async
remote-gpu run x.ipynb --internet  # enable internet for this run only
remote-gpu run x.ipynb --no-internet  # disable internet for this run only
remote-gpu run x.ipynb --dry-run  # preview kernel metadata without pushing
remote-gpu logs                   # tail of latest run's log
remote-gpu logs --follow          # stream until done
remote-gpu logs --save run.log    # also write to file
remote-gpu pull                   # download available output after a run stops
```

`--internet` and `--no-internet` override the YAML value for one run; without either flag, the YAML setting is used. `--dry-run` prints the planned kernel metadata, including GPU, internet, and dataset sources, without uploading data, contacting Kaggle, or saving run state. A local input directory is shown as a planned managed dataset; dry-run does not verify whether it has already been uploaded. A Kaggle username must still be configured so the preview can show the kernel and dataset IDs.

A failed, cancelled, or timed-out kernel can still have downloadable output. `remote-gpu run` attempts to recover it before reporting the failure; `remote-gpu pull` also attempts a download after those statuses. Recovery depends on what Kaggle makes available, and an unavailable or empty output is reported as an error.

Kaggle may not expose live cell output for a running notebook through its logs API. When `remote-gpu run` or `remote-gpu logs` gets no entries while the kernel is running, it explains this once rather than silently waiting. A RUNNING status does not prove progress; check the Kaggle UI and write periodic checkpoint or progress files to `/kaggle/working` so they can be downloaded after the run stops if Kaggle makes them available. `--follow` cannot force live cell output to appear.

## How it works

- `input/` is synced to a private dataset `you/remote-gpu-<name>-data`
  (skipped when unchanged)
- Your notebook gets an injected preamble that symlinks `input` → the mounted
  dataset and creates `output`, then is pushed to kernel `you/remote-gpu-<name>`
- The uploaded copy gets stable, unique cell IDs (nbformat 4.5); the local notebook is unchanged
- After the run stops, files Kaggle makes available from `/kaggle/working` are
  downloaded into `output/`
