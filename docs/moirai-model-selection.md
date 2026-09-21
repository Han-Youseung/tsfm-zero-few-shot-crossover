# MOIRAI model-selection gate

Checked on 2026-09-21. This gate selects a model for this study; it does not reproduce
TSFM-Bench. Only official SalesforceAIResearch/Uni2TS code, Salesforce weights, model
cards, releases, and the MOIRAI paper were used.

## Decision

`Salesforce/moirai-1.1-R-small` at Hugging Face revision
`0c24ab99db2c1a70ea2a0fc03bf113329772ac64` passed the local CPU functional gate with
Uni2TS 2.0.0 at commit `cfd46d4510ed8896f263116f32928eede05b0a75`. It is selected for
GPU confirmation and remains `pending_gpu`, not `frozen`. Because the first-priority
small candidate passed, MOIRAI 1.0 and all Base candidates were not downloaded or run.

The selected source is an exact commit on official `main` and reports package version
2.0.0. The official `2.0.0` release tag points to the earlier commit
`8062ef5a5660d2fea395fd1288ec9c397396c168`; both are recorded so the package label is
not mistaken for tag identity. The current pinned commit worked, so the 1.2.0 and 1.1.0
fallback tags were not used.

The official code is Apache-2.0. The selected weights are CC-BY-NC-4.0, so this study
must remain within that non-commercial license. The exact config SHA-256 is
`37b9eca5a0c3e58c2cdac3ecdbf897acbd009a20d6ba0f8a2ae0e8d92002fd47`; the 55,320,200
byte safetensors object has SHA-256
`cc46831272ea07e99d78fe05c09e730c7c76a1b272cddbdaf5173dae371cb238`.

## Official capability path

The official `MoiraiForecast` accepts `target_dim=7` and packs seven variates inside
one sample using `variate_id`; it is not seven univariate batch items. The official
`MoiraiFinetune` uses `finetune_pattern=full`, `PackedNLLLoss`, and its own AdamW
configuration. Its validation path draws probabilistic samples and uses their median
for point metrics. `MoiraiModule` applies `PackedStdScaler` internally and returns a
distribution on the original scale. External StandardScaler is disabled to avoid
double scaling.

The local smoke used fixed patch size 64 because this is the official small-model LSF
setting and the fine-tuning wrapper requires a fixed patch size. It was not selected
using test performance. Fixed patching also keeps Zero-Shot and Fine-Tuning on the same
path and avoids the extra history that `patch_size="auto"` uses to select a patch.
The full study's `num_samples=100` and sample-median rule are provisional until the GPU
pilot confirms cost and reproducibility; the CPU shape gate used eight samples only.

## CPU evidence

The probe used the grade-A official ETTh1 raw file, train data for one optimization
step, validation data for inference/validation, and no test values. Input shape was
`(1, 512, 7)`. All four horizons were produced in one forward pass per horizon:

| Horizon | Sample output | Point output | Packed tokens | Right padding | CPU seconds |
| ---: | --- | --- | ---: | ---: | ---: |
| 96 | `(1, 8, 96, 7)` | `(1, 96, 7)` | 70 | 32 | 0.556 |
| 192 | `(1, 8, 192, 7)` | `(1, 192, 7)` | 77 | 0 | 0.544 |
| 336 | `(1, 8, 336, 7)` | `(1, 336, 7)` | 98 | 48 | 0.330 |
| 720 | `(1, 8, 720, 7)` | `(1, 720, 7)` | 140 | 48 | 0.542 |

All outputs were finite, repeatable after resetting seed 1729, and left the parameter
hash unchanged. A deterministic predictive-distribution mean was permutation
equivariant after restoring channel order (maximum absolute difference
`8.58306884765625e-06`, tolerance `1e-5`). Comparing raw Monte Carlo draws would be an
invalid permutation test because a channel permutation changes random-number assignment.

The model has 13,827,528 parameters. All were trainable. One official optimizer step
gave finite loss and gradients, changed model parameters, and survived a state-dict
save/restore equality check. These values are smoke-test diagnostics, not performance
results. These are warm-cache timings, not benchmarks. Peak process RSS was 703.57 MiB.
GPU/AMP and peak GPU memory are untested and
must not be inferred from CPU memory.

## MOIRAI 2 exclusion

`Salesforce/moirai-2.0-R-small` is preserved as
`incompatible_with_required_multivariate_protocol`: univariate Zero-Shot was validated,
7-channel Zero-Shot failed in the official multi-token path, and an official full
fine-tuning wrapper/objective was not validated. No channel-wise loop, reshape patch,
custom loss, MOIRAI 1 wrapper reuse, PEFT substitution, recursive forecast, or source
patch was introduced.

## Remaining gate

Run `notebooks/12_moirai1_compatibility.ipynb` on Colab Pro+ in order. Success requires
the pinned revisions, all four 7-channel shapes, finite samples/loss/gradients, a real
parameter update, successful checkpoint restore, and reported CUDA/AMP peak memory.
Only then may the decision advance from `pending_gpu`; this CPU result cannot freeze the
model or authorize a production adapter by itself.
