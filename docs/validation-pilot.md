# Phase 6 validation-only A100 pilot

Status: implementation/preparation complete; actual pilot GPU execution **pending**.
No pilot score, convergence claim, final learning rate or runtime extrapolation is available.
Local torch reports CPU-only. Returned phase-5 adapter evidence contains eight passed
conditions at `689a284dac1c2b688b3efb6c1e2141fd966068d6`, on **Tesla T4**, not A100.
Those tests are reused, not rerun. They cover batch two, seven channels, FP32, one update;
MOIRAI's 100-sample **adapter** inference is still a pilot feasibility check.

## Data, split and sampling decisions

The actual `chronological_split` is floor-based 60:20:20. There is **no additional
validation split within train**. Train context and targets stay in train; validation
targets stay in validation, with historical train context allowed. Test is blocked.
No external input scaler is fitted. Train population standard deviations serve only
the evaluation metric, not model inputs.

The primary research grid already includes 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%.
The new requested 2/5/10/20/50/100 grid is a subset, not a second sampler. Keep the
original nested master order; pilot uses only 5%. Removing 0.5/1% from the final study
is not automatically approved. Existing `SamplingManifest` produces identical selected
window IDs for both models at a fixed dataset/horizon/seed. Manifest hash, requested
and effective ratio, counts, unique timestamp coverage and actual visited windows are
recorded. Overlapping windows do not imply the same fraction of raw observed timestamps.

Sampling ratio controls unique windows; optimizer steps control learning exposure.
Recommend the previously agreed **within-model common maximum-step rule** for the main
study. A fixed epoch cap grants larger subsets more updates; a fixed step cap repeats
small subsets more often. Record equivalent epochs and average window exposures, and
use matched-epoch/compute controls only as explicitly separate robustness studies near
candidate crossovers. The 200-step pilot cap is not the final training budget.

## Static model constraints (not measured GPU feasibility)

Fixed source links: [MOIRAI module](https://github.com/SalesforceAIResearch/uni2ts/blob/cfd46d4510ed8896f263116f32928eede05b0a75/src/uni2ts/model/moirai/module.py),
[forecast packing](https://github.com/SalesforceAIResearch/uni2ts/blob/cfd46d4510ed8896f263116f32928eede05b0a75/src/uni2ts/model/moirai/forecast.py),
[RotaryProjection](https://github.com/SalesforceAIResearch/uni2ts/blob/cfd46d4510ed8896f263116f32928eede05b0a75/src/uni2ts/module/position/attn_projection.py).

MOIRAI patch 64 yields per-channel time tokens `8 + ceil(horizon/64)`: 10/11/14/20.
Joint sequence length multiplies by channels. `max_seq_len=512` initializes rotary
time-position buffers; the pinned implementation extends them from `time_id.max()+1`.
It is **not an enforced 512-total-token cap** in this direct forecast path. The
`max_dim` training-transform sampling setting is not a fixed channel embedding table;
our existing adapter uses all supplied channels and does not use that random transform.
However, attention masks/biases and attention pairs remain quadratic in joint tokens.

At horizon 720: seven channels produce 140 tokens; Electricity 370 produces 7,400
(54,760,000 attention pairs/sample); Traffic 862 would produce 17,240 (297,217,600).
These are shape calculations, **not VRAM predictions**. A100 capacity and allocator,
attention kernels, backward graphs and sample count all matter; batch one may still OOM.
No channels are split/dropped and no patch/context change or recursive forecast is enabled.

TTM retains its pinned horizon-specific checkpoints, official crop at 192 and 512-point
left padding for the native 1024/720 checkpoint. Its common-channel shared-parameter path
has no cross-channel mixer here; activation memory still grows with channel count/batch.
The input channel axis is preserved. This is not an external univariate loop.
All 56 dataset/horizon shape calculations are in `structural_risks.json`.

## Metrics and point statistics, declared before execution

The candidate primary/checkpoint selection metric is **channel-macro normalized MAE**.
For each channel, sum absolute or squared errors and divide by the actual valid element
count. Normalize MAE by train population std and MSE by its square. Report raw and
normalized MAE/MSE per channel and macro averages. This avoids a large-unit channel
dominating an original-unit micro-average; raw results remain available for interpretation.

Train std <= 1e-8 or no valid train values makes normalized channel scores undefined
(null), with exclusion count; raw valid scores remain. No valid targets gives null.
A nonfinite prediction on a valid target fails. Zero baseline error gives undefined
relative improvement, not infinity or invented gain. Dataset macro gives each dataset
one vote; aggregate only matching metric/model/horizon/sample/seed conditions and disclose
which datasets are included. Streaming float64/Python-double sums include the last batch.
No full prediction/sample array is saved.

TTM uses official `prediction_outputs`; MOIRAI uses provisional `torch.median`, the
lower middle sample for even sample counts, with 100 samples in this pilot. A predictive
mean corresponds to squared-error optimality, a median to absolute-error optimality.
The primary MAE candidate therefore matches the provisional median better; MSE remains
a secondary diagnostic. The model-specific **official training objectives** are unchanged
and their loss magnitudes are not directly compared.

MOIRAI adapter prediction restores training RNG. For canonical score calculation every
validation window is processed individually with the fixed prediction seed; evaluation
batch is intentionally fixed at one across LR candidates and Zero/Few-Shot. Throughput
calibration separately compares the first window in batched versus single inference and
records its difference. No batch-invariance claim is made for sample allocation.

## Bounded plan

`configs/pilot/a100_validation.yaml` is provisional and produces a complete explicit list.
With five ready datasets: **40 feasibility groups + 12 stability conditions = 52 planned**;
72 blocked feasibility groups remain listed for the other nine datasets (124 total).
Each model session has 20 feasibility groups and six stability conditions.

A: all four horizons for every ready dataset, high horizons first. Each group explores
inference and training independently at batch 1/2/4/8/16/32/64, stopping each ladder after
failure, insufficient distinct windows or less than 25% reserved-memory headroom.
At most 560 FP32 batch attempts across 40 groups. Each training attempt starts from
pretrained and performs one update. Each condition records warm-up, synchronized time,
allocated/reserved peaks, token shape, actual GPU/VRAM/driver and package versions.
Recommendations maximize observed throughput under the headroom rule, not largest batch.
They are **not automatically applied** to the stability config. `amp_probe: true` adds at
most two isolated batch-one bf16 probes/group if hardware supports it; default is false.
Failures do not invalidate FP32 success. No GradScaler is needed for these bf16 probes.

B: ETTh1 and **Electricity UCI-370 hourly**, chosen before scores for familiar low-channel
transformer measurements versus high-channel demand. This provides a deliberately useful
domain/channel/memory contrast, not a claim that Electricity fits. If its feasibility fails,
that representative remains pending; no substitute is silently chosen. Horizon 96, 5%,
one seed, up to 200 optimizer steps, validation every 50 steps on at most 64 preselected
midpoint strata of rolling origins, patience four evaluations. One fixed microbatch/effective
batch per LR comparison (default one, accumulation one). Tail batches are kept and recorded.

TTM LR candidates: 1e-6/1e-5/1e-4; MOIRAI: 5e-7/5e-6/5e-5. The first values anchor
earlier engineering smoke, not optimality; MOIRAI's first value is also in the pinned
[official small fine-tuning config](https://github.com/SalesforceAIResearch/uni2ts/blob/cfd46d4510ed8896f263116f32928eede05b0a75/cli/conf/finetune/model/moirai_1.1_R_small.yaml).
The other log-spaced values are explicit exploratory candidates, not official recommendations.
No search expansion is automatic. Constant LR, no scheduler; all model parameters remain
trainable. BF16/accumulation policy for the main study remains undecided. A bounded run
finishing at 200 steps is labeled convergence not established, not converged by assumption.

## Execution and interruption recovery

Use `notebooks/30_validation_pilot.ipynb` in two fresh Python 3.11/3.12 GPU sessions,
select `ttm` then `moirai1`, and enter the full published phase-6 commit. A100 is the
target; any allocated GPU is named honestly. Installation remains model-specific and
all model subprocesses use the printed dedicated interpreter. No model upgrade or pull.

Cell order: runtime/commit; repository; user-authorized Drive; dependencies; pinned data
verification; inspect printed plan; type RUN_A; review results; type RUN_B; export ZIP.
`completed` for feasibility means the ladder finished, **not all batches succeeded**:
check `fp32_batch1_supported`, individual statuses and `memory_headroom_ok`.
The notebook blocks B unless the representative's horizon-96 batch-one A result passed.

Results/checkpoints live on the user's mounted Drive. `last.pt` atomically contains model,
optimizer, all RNG streams, configuration/identity, step, sampler order/position/epoch,
visited windows, early-stopping state, best metric/path and log history. Scheduler and
AMP scaler are explicit null in required FP32. Best checkpoints have step-specific names;
writing a newer best cannot corrupt an older last transaction. Resume replays at most
the work since the last persisted checkpoint (default ten steps). Different condition
identities cannot resume; candidates start from pretrained independently.

Exclusive `running.lock` prevents duplicate writers. A killed runtime may leave it behind:
verify the old process/session is dead, then remove **only that condition's lock** before
retrying. Do not remove last.pt or rewrite identity to force resume. Completed outputs skip;
failed attempts remain separate JSON. Batch-calibration attempts also persist individually.

CLI from repository root, inside the selected environment:

```bash
python -m tsfm_crossover.experiments.pilot --expected-commit FULL_PHASE6_SHA --output DRIVE_OUTPUT
# Read DRIVE_OUTPUT/plan.json first; execute one explicit condition:
python -m tsfm_crossover.experiments.pilot --expected-commit FULL_PHASE6_SHA --output DRIVE_OUTPUT --condition-id PLAN_ID
```

Download `ttm-phase6-pilot-results.zip` and `moirai1-phase6-pilot-results.zip` containing
JSON/CSV summaries and preserved failures only. Checkpoints stay on Drive, never Git.
No total research duration is estimated without measured pilot throughput and validation/
checkpoint overhead. Final budgets, LR, batch sizes, precision, metrics, point statistic,
high-channel strategy and acceptance of alternative variants remain review decisions.

## Local verification record

175 default tests passed; two optional torch modules skipped in the torch-free base
environment. In the separate CPU torch environment, ten tensor tests passed, including
interrupted/resumed pilot equivalence, last-batch retention, failure-bounded batch ladders
and cached-attempt reuse. GPU absence yields pending, not success. These use toy tensors,
not fabricated model or GPU measurements. The earlier real adapter GPU evidence is
preserved separately. Editable install, lazy import, YAML/model-manifest checks, new
notebook JSON/syntax/empty outputs and scoped Ruff checks passed. No full experiment ran.
