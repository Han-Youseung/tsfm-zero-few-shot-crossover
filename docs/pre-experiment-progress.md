# Pre-experiment gates (2026-09-25)

The goal remains **all fourteen named datasets**. No dataset has been removed,
and test evaluation/main experiments remain disabled. This is a staged readiness
report, not a claim that every gate has passed.

## 1. Returned A100 pilot: validated

Execution commit: `486fb7c5151773a372b9371d433b9c05d578b1e5`.
Both returned archives report **NVIDIA A100-SXM4-40GB**, not the earlier 80GB gate.
`results/manifests/pilot/a100_review.json` preserves archive SHA256, canonical
record hashes, environment/package snapshots, every batch attempt including failures,
timing/memory, validation history and source member names. Original ZIPs stay local.

The importer regenerates the 52-condition plan, splits, sampling manifests and
validation windows; checks dataset/model revisions, installed official commit,
CUDA metadata and optimizer steps; compares separate attempt files against their
embedded records; and refuses incomplete, mismatched, test-using or frozen evidence.
It never loads a checkpoint or executes archive contents. The compact review does
not replace source archives; retain them on Drive and locally for full provenance.

- 40 feasibility groups: ETT four + Electricity, two models, four horizons.
  All support FP32 batch-1 inference and one-step full fine-tuning.
- 12 stability conditions: two representative datasets, three LRs per model,
  200 steps, 5% selected windows, 64 validation windows, batch 1.
- Seven higher-batch OOM attempts are preserved, not interpreted as model failure.
- MOIRAI used 100 samples. AMP was not run. No test scores or crossover computed.

Validation normalized MAE (smaller is better; selected best checkpoint):

| Model | Dataset | Zero-shot | LR candidates in ascending order | Best validation scores |
|---|---|---:|---|---|
| TTM | ETTh1 | 0.374348 | 1e-6 / 1e-5 / 1e-4 | 0.374391 / 0.374364 / 0.372952 |
| TTM | Electricity | 0.227423 | 1e-6 / 1e-5 / 1e-4 | 0.227066 / 0.225501 / 0.224967 |
| MOIRAI | ETTh1 | 0.386524 | 5e-7 / 5e-6 / 5e-5 | 0.387358 / 0.383984 / 0.396541 |
| MOIRAI | Electricity | 0.290667 | 5e-7 / 5e-6 / 5e-5 | 0.285740 / 0.274922 / 0.270815 |

TTM 1e-4 is the strongest of these bounded candidates on both representatives.
MOIRAI has a dataset tradeoff; do not pick a different LR per dataset or add a
search silently. Several best steps equal the 200-step limit: convergence and
the final training budget are **not established**. No final LR is applied here.

Electricity horizon 720 batch-1 allocated peak training memory was 12,501.3 MiB
(TTM) and 13,846.3 MiB (MOIRAI). These are measured one-step peaks, not guarantees
for Traffic's 862 channels, all datasets, all batches, or long-running training.
Full-study time cannot yet be estimated reliably for the unresolved datasets.

Reproduce review without GPU or downloads:

```sh
python -m tsfm_crossover.experiments.pilot_review ttm-phase6-pilot-results.zip moirai1-phase6-pilot-results.zip
```

## 2. Fourteen-dataset resolution: in progress

Solar and Traffic have independently obtained, versioned Monash distributions:
[Solar 4656144](https://zenodo.org/records/4656144) and
[Traffic 4656132](https://zenodo.org/records/4656132). Both record APIs declare
CC-BY-4.0 and identify the Lai et al. matrix distribution. These are data sources,
not implementations or experimental protocols we copy.

`scripts/prepare_monash_variants.py` preserves every series/value and the published
calendar (including its one-second start offset). Traffic matches the author
matrix exactly. Solar differs only by at most 1.4211e-14 from numeric serialization;
the manifest records exact/nonexact comparison and a 1e-12 absolute tolerance.
Neither is claimed to provide independently recovered original sensor timestamps.
The new `prepared_expansion.json` overlay preserves the five prior ready variants
and adds these two. Earlier manifests and GPU records remain unchanged.

Remaining seven resolutions:

| Dataset | Next required resolution |
|---|---|
| AQShunyi | Missing-input/target policy and official-loss masking verification |
| Weather | Exact duplicate equality, irregular timestamps and documented repair policy |
| PEMS08 | Document official flow field selection, calendar and data-use terms |
| Exchange | Original distribution terms and observation-index/calendar interpretation |
| Wind | Identify the provider/variant, not merely another wind dataset |
| ZafNoo | SAPFLUXNET site, variables, units, quality flags and span |
| CzeLan | SAPFLUXNET site, variables, units, quality flags and span |

SAPFLUXNET [0.1.5 record](https://zenodo.org/records/3971689) API confirms CC-BY-4.0;
its archive is 3,241,703,824 bytes. License discovery does not by itself establish
the two site/variable mappings. No blocked legacy bundle is promoted.

## 3. Remaining sequence (no automatic main experiment)

1. Resolve the seven data policies/provenance issues; test each conversion, pin hashes.
2. Run only new validation-only feasibility conditions for newly ready datasets,
   prioritizing Traffic horizon 720 on A100. Do not rerun the 52 validated conditions.
3. Review resource envelope across all fourteen; propose one fixed model-specific
   batch/step/optimizer policy. Any channel split, patch/context change or dataset
   substitution requires explicit research approval.
4. Resolve final metric, seed, sampling grid, point statistic and training-budget
   decisions using validation evidence only; document uncertainty/block inference.
5. Test main-run planning/resume/data leakage gates, calculate measured resource
   estimates, and present a pre-experiment checklist for approval. Do not run test.

Missing-data policy is not silently chosen: proposed context-only causal forward
fill, no future interpolation/backfill, and exclusion of missing targets from
official losses/metrics require review of both official objectives first. Leading
missing inputs, long gaps, quality flags and changed candidate-window counts must
be documented. Until approved and tested, affected data remain blocked.

## Local verification and next execution boundary

Base suite: **190 passed, 3 skipped** (optional NumPy/torch integration modules).
Separate tensor/preparation environment: **15 passed**. Ruff lint/format, editable
install, torch-free import, all configuration YAML/manifest JSON parsing and clean
notebook syntax/output checks passed. No new GPU run occurred during this review.
Solar/Traffic output fingerprints also reproduced in both separate local model
environments (pandas 3.x/NumPy 2.x and pandas 2.1.4/NumPy 1.26.4). An idempotent
manifest JSON tuple/list roundtrip issue was fixed and covered by a regression test.
The expanded plan has 68 ready groups and 56 blocked groups; 52 of the ready groups
already have validated historical evidence. The next GPU increment is only the
16 Solar/Traffic feasibility groups, not all 68 groups and not stability reruns.

After a clean committed checkout and the existing model-specific Colab installation:

```sh
python scripts/prepare_monash_variants.py
python -m tsfm_crossover.experiments.pilot --prepared results/manifests/pilot/prepared_expansion.json --expected-commit FULL_NEW_COMMIT --output DRIVE_NEW_OUTPUT
```

The first command reproduces Solar/Traffic; prior five prepared files must be
present if executing their conditions, but those conditions are not needed again.
Filter the printed plan to `kind == feasibility`, dataset in `{Solar, Traffic}`,
and the current session's model family. Execute one selected ID with the same
command plus `--condition-id ID`. Start with Traffic horizon 720, batch ladder
starts at 1 automatically. Use separate model sessions and a new Drive output
directory. Do not reuse old condition IDs under a new commit, copy old results
into a new run as if re-executed, or run the unfiltered notebook's full job list.

Additional raw-integrity findings (not test metrics): Weather's repeated
2020-05-12 06:00 rows are identical in every channel. Its jump from
2020-05-29 09:30 to 11:10 omits nine ten-minute timestamps. Deduplication alone
therefore does not fix the grid. Wind's legacy fields include `pred_w_speed`,
`pred_w_dir`, `pred_temp`, `pred_pressure`, `pred_humidity`, `ture_w_speed` and
`target`, spanning 2020-01-01 to 2021-05-22. The `pred_*` fields also require an
issuance-time/availability explanation; a name/shape match is insufficient to
rule out future-information leakage. Original source documentation is needed.
