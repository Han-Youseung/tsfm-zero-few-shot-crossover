# Phase 5 shared experiment adapters

`AdapterContract` is implemented by `TTMAdapter` and `MoiraiAdapter`. Imports of
the core package and factory do not import torch or vendor packages. Use separate
model environments. The selected revision, config SHA256 and official code origin
are checked before loading. A local source installation must match a clean pinned
checkout byte-for-byte for its Python files; a VCS installation uses PEP610 commit
metadata and the expected package version.

The interface uses `WindowBatch` with `(batch,512,channels)` past and exactly
`(batch,horizon,channels)` future. Channel names/order and dataset/sampling hashes
are part of the condition identity. The existing WindowIndex, chronological split
and nested sampling manifest construct batches; there is no second sampling policy.
The engineering entry point rejects test windows and does not fit an external scaler.
Channel counts are dynamic; previous GPU evidence covers only seven channels.

Each adapter instance loads a fresh pinned pretrained checkpoint. Zero-Shot creates
no optimizer and verifies state hash stability. Configuration of full fine-tuning
checks `requires_grad` and exact optimizer membership. Official losses remain intact;
see [source review](gpu-evidence-review.md). No sample-rate warm start is allowed.
The runner executes at most two optimizer steps on a small subset of selected windows;
it records the total selected count separately from actually executed windows. These
smoke budgets and optimizer settings are not pilot decisions.

Checkpoint files contain model/optimizer state, condition configuration, revision,
pretrained config, global step, exposures, Python/NumPy/CPU/CUDA RNG states, and an
explicit null GradScaler/scheduler for FP32. Restore requires a fresh adapter and
exactly matching condition identity. Checkpoints use torch serialization and must
only be loaded from trusted local training runs. They are never committed.

Prediction saves/restores all RNG streams and model mode, and uses a configured
prediction seed; calling evaluation does not advance subsequent training randomness.
MOIRAI uses the official sample tensor and `torch.median(dim=1)` provisionally. For
even sample counts this is the lower middle sample, matching the earlier probe.
TTM exposes official `prediction_outputs`, including its internal crop at 192 and
caller-side left zero padding at 720. The target always has exactly the requested
horizon. Missing/nonfinite input currently fails explicitly.

## Small integration execution

Use `notebooks/20_adapter_integration.ipynb` twice, in separate fresh GPU sessions,
selecting `ttm` and `moirai1`. Python 3.11/3.12, a CUDA GPU and an immutable **phase-5**
commit are required. Run all cells in order: runtime; pinned repository; optional
user-mounted Drive; dedicated environment; raw ETTh1 fingerprint; four conditions;
download. Kernel and probe interpreters are printed, and installation/probes use the
same dedicated interpreter. The notebook requires no GitHub write credentials.

Each condition is a separate process and atomically writes a small JSON. Successful
conditions resume after exact source/config matching; failed attempts are retained
and require a new filename for a rerun. A Drive destination survives runtime loss;
temporary runtime storage does not. Download `{family}-adapter-results.zip` and
return it for validation. Expect exit 0, status passed, every check true and point
shape `[2,horizon,7]`. CUDA unavailable leaves pending_gpu. Any failed/missing check
must be investigated; prior probe results cannot fill it in.

Example from the repository root, inside the selected model environment:

```bash
python scripts/smoke_adapters.py --config configs/adapters/ttm_smoke.yaml \
  --data data/official_raw/ETTh1.csv --device cuda --horizon 96 \
  --expected-commit FULL_PHASE5_COMMIT --output results/raw/ttm-adapter-h96.json
```

For MOIRAI replace the config with `configs/adapters/moirai1_smoke.yaml`. Local CPU
verification uses `--device cpu --local-files-only` and cached official weights.
`--allow-dirty` is only permitted for CPU development; resulting records explicitly
include dirty_worktree and source SHA256 and cannot claim clean-commit GPU evidence.
Actual adapter GPU validation remains pending until this new route is executed.

## Completed local verification

Both real official models passed CPU FP32 integration on identical ETTh1 train and
validation windows: context 512, horizon 96, seven channels, batch two, one optimizer
step. Zero-Shot left parameters unchanged; full optimizer coverage, finite gradients,
parameter updates, prediction RNG preservation and fresh-instance checkpoint restore
passed. Restored validation predictions had maximum absolute error 0.0 in both runs.
The small records are in `results/manifests/adapters/`; no checkpoint or arrays are stored.

These are explicitly dirty-worktree development runs based on commit
`9db06598504d2abe77f5809281256272914640de`, with verified source SHA256
`04ac8be8b2d47e1fef9626d082c57b92ed9020df6579ce1a70428a923a6b98a2`.
They are not relabeled as runs of the later commit that publishes the implementation.
Default tests: 161 passed, one optional torch module skipped. A separate torch
environment passed all seven tensor integration tests. Adapter GPU, other horizons
with the new adapters, AMP and high-channel feasibility remain unverified.

## Next pilot decisions

Sampling rate controls distinct train windows, whereas optimizer steps control
updates and repeated exposure. Select model-specific maximum steps, evaluation
interval and patience together; equivalent epochs/exposures follow from the selected
window count and batch size. Learning rate, batch size and accumulation, primary
metric and channel/dataset aggregation, point statistic/sample count, FP32/AMP,
and high-channel memory feasibility remain validation-only pilot choices.
