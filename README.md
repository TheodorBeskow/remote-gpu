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
remote-gpu clean                  # delete retained .remote-gpu/downloads staging
```

`remote-gpu status` verifies credentials with a Kaggle API call instead of only checking that a credentials file exists; quota tracking is not implemented. `--internet` and `--no-internet` override the YAML value for one run; without either flag, the YAML setting is used. `--dry-run` prints the planned kernel metadata, including GPU, internet, and dataset sources, without uploading data, contacting Kaggle, or saving run state. A local input directory is shown as a planned managed dataset; dry-run does not verify whether it has already been uploaded. A Kaggle username must still be configured so the preview can show the kernel and dataset IDs.

A failed, cancelled (including Kaggle's `CANCEL_ACKNOWLEDGED` status), or timed-out kernel can still have downloadable output. `remote-gpu run` attempts to recover it before reporting the failure; `remote-gpu pull` also attempts a download after those statuses. Recovery depends on what Kaggle makes available, and an unavailable or empty output is reported as an error. Kaggle may return files from an earlier completed version after cancellation; check that downloaded files belong to the run you intended before treating them as recovered partial results.

Output downloads stage under `<config directory>/.remote-gpu/downloads/<run-id>/` instead of a temporary `tmp*` directory. Temporary upload files also stay under `.remote-gpu/`. The existing `.gitignore` ignores `.remote-gpu/`; if you use remote-gpu in another project, ignore that directory there too. Download and copy counts include file and byte totals, with a final completion message. Transient download failures (including common HTTP 403/5xx and connection errors) are retried up to three times with the same staged files. If retries are exhausted or the transfer is interrupted, staged files are kept and the error shows their location; run `remote-gpu pull` again after the kernel stops. Retries reuse the same run's stage and Kaggle CLI can skip up-to-date files; it may still re-download files when it cannot verify freshness. A new kernel push uses a new stage. After a successful copy, that run's staging directory is removed; retained staging from interrupted runs can be removed with `remote-gpu clean`. A completed transfer confirms that the files Kaggle supplied were copied locally, not that Kaggle exposed every file from the notebook. For large outputs, use Kaggle CLI 2.2.2 or newer, which downloads all output pages; older versions may only retrieve the first page.

Kaggle's one-shot logs may be empty until the run finishes. `remote-gpu logs --follow` uses `kaggle kernels logs -f` when the installed Kaggle CLI supports it; Kaggle CLI 2.2.3+ streams live session logs via SSE. Older versions fall back to polling persisted logs. `remote-gpu run` still polls persisted logs and may show no progress; its notice suggests `remote-gpu logs --follow` in another terminal. Live streams may still omit notebook UI updates such as tqdm displays. A RUNNING status does not prove progress; check the Kaggle UI and write periodic checkpoint or progress files to `/kaggle/working` for retrieval after the run stops.

To stop a run, open the notebook using the Kaggle link printed by `remote-gpu run` and use Kaggle's Stop/Cancel control for the running session or version in the web UI. Ctrl+C or closing `remote-gpu logs --follow` only detaches the local watcher; it does not stop the remote run. The installed Kaggle CLI has no safe kernel stop command: do not use `kaggle kernels delete` to stop a run, because it deletes the notebook. After stopping, try `remote-gpu pull` to recover any output Kaggle makes available.

## How it works

- `input/` is synced to a private dataset `you/remote-gpu-<name>-data`
  (skipped when unchanged)
- Your notebook gets an injected preamble that symlinks `input` → the mounted
  dataset and creates `output`, then is pushed to kernel `you/remote-gpu-<name>`
- The uploaded copy gets stable, unique cell IDs (nbformat 4.5); the local notebook is unchanged
- After the run stops, files Kaggle makes available from `/kaggle/working` are
  downloaded into `output/`
