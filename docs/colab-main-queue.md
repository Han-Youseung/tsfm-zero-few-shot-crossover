# Colab main-study queue

## Files and scope

- `notebooks/42_ttm_main_queue.ipynb`: TTM-only session.
- `notebooks/43_moirai1_main_queue.ipynb`: MOIRAI 1.1-only session.
- `scripts/colab_queue.py`: standard-library orchestration embedded identically in both notebooks.

The engine remains pinned to `3e18305032484239966026f14395b903c9abf268`.
This is a notebook-side orchestration change, not a new training engine or adapter.
It does not change checkpoints, packages, learning rates, steps, precision, data
splits, sampling, validation selection, final-test windows, or metrics. The original
one-condition CLI verifies each completed result before skipping it. No final-test
score controls queue order or configuration. Existing notebooks 40/41 remain intact.

## Usage

1. Finish or safely stop any existing condition before using this notebook. Do not
   run overlapping conditions in two notebooks. Use separate model-family runtimes.
2. Upload the notebook to Colab. Use Python 3.12 and a compatible CUDA runtime.
3. Edit only the scope/session settings cell. Defaults are ETT four datasets,
   H96, three seeds (1729, 2718, 31415): 108 conditions including zero-shot.
   `DATASETS = ["ALL"]` selects the approved nine datasets; set
   `HORIZONS = [96, 192, 336, 720]` to include four horizons. All nine datasets and
   four horizons with three seeds are 972 conditions per model, not one-session work.
4. Run settings, queue tools, runtime, repository, Drive, installation, and plan in
   order. Drive authorization is performed by the user. No GitHub write token needed.
5. Click run once. No further prompts: it prepares each needed dataset once,
   validates its canonical fingerprint, and executes the selected queue sequentially.
6. Run export when convenient (also after a partial run) to download one aggregate ZIP.

The condition IDs, commit-specific output directory, checkpoint and test-progress
files are identical to the original notebook, allowing reuse of existing results.
Model/HF/Xet caches stay on local disk; result/checkpoint storage stays on Drive.
Changing GPU/software during a partial condition remains blocked by the original
engine. Never remove `running.lock` without checking the old process first.

## Progress and recovery

The queue reports index/total, condition ID, verified-existing versus newly completed,
elapsed wall time and the latest bounded log tail every 30 seconds. A process-alive
heartbeat is **not** proof of optimization progress. A recent `test windows x/y`
message indicates evaluation progress. No synthetic training percentage or ETA.

The default session budget is 18 hours measured from the runtime-check cell's first
execution in this kernel. It reserves 30 minutes before that user-defined deadline
and checks again after data preparation. It does not know the actual Colab expiry;
it cannot guarantee that a condition will finish before disconnection. It does not
terminate an active condition at the deadline. The 972-condition limit is an
orchestration cap, not a model optimizer-step budget. Re-running the run cell does
not reset the session clock.

Failures, nonzero CLI exits, missing results (including resource-pending), result
identity mismatches, and backup errors stop the queue. It never silently reduces
channels, changes precision, skips failed conditions, deletes locks or modifies old
failure files. Keyboard interruption requests termination only of the subprocess
started by this queue. Runtime death may leave a running journal/lock; this is not a
successful result. Review it before resuming. There is no unattended reconnection,
keep-alive hack, or automatic Drive authentication.

## Backup layout

- Original results/checkpoints: `MyDrive/tsfm-main-study/<engine-commit>/<family>/`
- Condition ZIPs, SHA-256 sidecars and invocation-specific queue journal:
  `MyDrive/tsfm-main-study/queue-exports/<engine-commit>/<family>/`
- Manually requested aggregate ZIP: `MyDrive/tsfm-main-study/exports/`

Each completed/failed condition attempts a unique-name ZIP. ZIP CRC and SHA-256 are
checked; earlier exports are never overwritten. Queue journals use atomic replacement
only for the current invocation. ZIPs contain `plan.json` and small result/runtime/
provenance/selection/start/failure/pending JSON, with the original directory layout.
They exclude weights, raw data, token files, full training history, samples, optimizer
checkpoints and test-progress arrays. Full subprocess logs are temporary local files;
engine failure summaries and queue error summaries persist on Drive. Checkpoints and
progress needed for resume remain on Drive, not in the export ZIP.

A forced disconnect can prevent final journal/ZIP writes. Completed original results
already persisted on Drive remain the source of truth; re-run the export cell after
recovery. Download via Drive directly if the browser blocks `files.download`.
An archive is not a declaration that every selected condition succeeded. Continue
using the existing local main-result provenance/import checks before analysis.

## Validation boundary

CPU tests cover actual approved-grid selection, order, completed-result validation,
failure/resource/interrupt stop, budgets, archive contents/integrity and embedded-source
parity. Tiny real CPU subprocess tests exercise the launcher. They do not download
weights or run a GPU. Actual end-to-end Colab execution of these new queue notebooks
remains **not_run** until returned evidence is verified.
