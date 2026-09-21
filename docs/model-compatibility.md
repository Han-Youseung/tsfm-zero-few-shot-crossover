# Phase 4 model compatibility assessment

Checked on 2026-09-21. This phase is an official-source and interface assessment, not
a benchmark run. No paper metric, complete test evaluation, or crossover result was
produced. The only eligible engineering data are the grade-A official raw ETTh1,
ETTh2, ETTm1, and ETTm2 variants. Bundle variants remain blocked.

## Reproducible candidates

| Item | TTM | MOIRAI |
|---|---|---|
| Model | `ibm-granite/granite-timeseries-ttm-r3` | `Salesforce/moirai-2.0-R-small` |
| HF commit | `ea17cfd2e3edcaea21eb8dcecd18bf88971482fa` | `30f43ff08c8494f4943ae1521e9d4e94a0fbb389` |
| Official code | `ibm-granite/granite-tsfm` | `SalesforceAIResearch/uni2ts` |
| Code commit | `fe7a35697723e2a2f5246ae979474bfc554e26c0` | `cfd46d4510ed8896f263116f32928eede05b0a75` |
| Package version | tag `v0.3.9` | `2.0.0` |
| Parameters | 1,414,514 for selected R3 revisions | 11,387,208 |
| Code license | Apache-2.0 | Apache-2.0 |
| Weight license | Apache-2.0 | CC BY-NC 4.0 |
| Phase state | `finetune_validated` | `import_validated` |
| CPU model runtime | 512/96 inference and one training step validated | univariate 512/96 passed; required 7-channel 512/96 failed |
| GPU runtime | `pending_colab` | `pending_colab` |

The HF repository commit records the repository snapshot. TTM is a suite, so the
actual forecast checkpoints are separately pinned in `ttm_compatibility.json`: the
96, 336, and 720 revisions resolve to immutable commits. A candidate is not called
`frozen` until the pilot succeeds.

The repository-main TTM checkpoint has 1,414,514 parameters, but the selected models
are not all that size. Safetensors inspection gives 5,328,932 (512/96 decomposed),
6,227,972 (512/336 decomposed), and 14,574,015 (1024/720). The first count was also
confirmed from the loaded CPU model.

## Dependency isolation

The official pinned TTM source declares Python 3.11–3.13, Torch 2.10–2.11,
Transformers 4.57.6–5.x, NumPy below 3, and pandas 2.3.3–3.x. The pinned uni2ts source
declares Python 3.10+, Torch 2.1–2.4, NumPy 1.26, SciPy 1.11, GluonTS 0.14.3, and
einops 0.7. These ranges conflict. Use independent TTM and MOIRAI runtimes from
`requirements/ttm.txt` and `requirements/moirai.txt`; never resolve both files in one
environment. The Colab runs must capture the resolved package snapshot before the
candidate can advance.

CUDA version, AMP behavior, peak GPU memory, gradient checkpointing, optimal batch
size, timing, and CUDA determinism are unknown until the Colab probes run. They are
not inferred from source code.

## Output and scaling contracts

TTM consumes float time-series tensors shaped `[batch, context, channel]`. Selected
R3 configs use model-native standard scaling and expose `prediction_outputs` shaped
`[batch, horizon, channel]`; this direct point head is the provisional MAE/MSE point
forecast. R3 configs also contain multi-quantile heads, but this study will not apply
the old “100 samples then median” rule.

MOIRAI 2 consumes `past_target`, its observed mask, and padding information. Its
tensor forward supports a target dimension and returns nine direct quantiles shaped
`[batch, quantile, horizon, target_dim]`. The provisional point forecast is the 0.5
quantile. This is not a `num_samples` rule. The convenience `predict` method currently
states univariate-only in official source; multivariate probes must use and validate
the tensor forward. MOIRAI 2 uses `PackedStdScaler` and returns rescaled predictions.

Both candidates therefore provisionally use model-native scaling with project-level
external StandardScaler disabled. This avoids unvalidated double scaling. The policy
is not frozen: the pilot must verify raw-unit recovery and use the same path for
Zero-Shot and Few-Shot. Any later external scaler must be fit on train only and must
inverse-transform before metrics.

## Context and horizon assessment

Classifications below are based on pinned official configs and code inspection, not
completed forward passes.

TTM has native 512/96 and 512/336 R3 checkpoints. Horizon 192 is cropped from the
single-pass 336 output through the official `prediction_filter_length`. Context 256
requires the official `force_return="zeropad"` selection path. Horizon 720 requires
the native 1024/720 checkpoint and zero-padding when only 256 or 512 observations are
declared. None of these entries uses recursive forecasting, but the padding cases are
not native context matches and require pilot validation.

For MOIRAI 2, `max_seq_len=512`, patch size 16, and ETTh1 has seven target channels.
With no covariates, source-level packed-token accounting gives
`7 * (ceil(context/16) + ceil(horizon/16))`. Every 256-context candidate is at most
427 tokens. At context 512, horizons 96/192/336 require 266/308/371 tokens; horizon
720 requires 539 and exceeds the published limit. This is a channel-dependent limit,
not a universal horizon limit, and needs a runtime probe.

Static token accounting originally left context 256 as a possible common candidate,
but the runtime probe exposed an earlier blocker: official 7-channel context-512,
horizon-96 forward enters the multi-token path and fails when a 28-token tensor is
rearranged with `predict_token=4`. All requested horizons are longer than four
16-point prediction tokens, so the other multivariate combinations traverse the same
code branch; they are recorded as untested, not assumed successful. There is no
validated common multivariate condition at this phase. Test performance must not be
used to work around this blocker.

## Zero-Shot and fine-tuning evidence

Zero-Shot is defined only as: **no parameter update using the designated target-data
train split**. The contract enforces evaluation mode, no optimizer, no backward call,
and equal parameter hashes before and after prediction.

TTM's official materials expose Zero-Shot inference and full-model fine-tuning. On
CPU with Python 3.12.14, Torch 2.11.0+cpu and float32, the pinned 512/96 decomposed
checkpoint produced finite `(1, 96, 7)` synthetic and ETTh1 validation outputs.
Repeated synthetic inference was bitwise equal and the parameter hash was unchanged.
One AdamW step on the same train/validation-only smoke path produced finite loss and
gradients and changed parameters. A temporary model/optimizer/step state save, forced
parameter mutation, and reload restored the trained parameter hash. This establishes
API compatibility, not a chosen optimizer or a performance result. GPU memory, AMP,
and other horizons remain pending Colab.

MOIRAI 2 officially exposes pretrained inference. CPU loading succeeded with Python
3.12.14, Torch 2.4.1+cpu, uni2ts 2.0.0 and float32. Univariate context-512/horizon-96
synthetic and ETTh1 validation inference produced finite direct quantiles of shape
`(1, 9, 96)`; repeat inference was bitwise equal and the parameter hash was unchanged.
The required seven-channel call failed in the official internal rearrange described
above. Running seven independent univariate forecasts would change the multivariate
model condition and was therefore not adopted without user approval.

At the pinned uni2ts commit,
`Moirai2Module` is differentiable and has a training-mode forward, but the published
Moirai2 forecast wrapper has no `training_step`, `validation_step`, optimizer setup,
documented loss wrapper, or Moirai2 fine-tuning configuration. That is insufficient
evidence for an official full-parameter fine-tuning path. The phase records
`full_parameter_finetuning: null` and `finetune_failed`; it does not invent an
objective, reuse a 1.x recipe, freeze parameters, or substitute PEFT. This is a design
decision blocker for the planned like-for-like Few-Shot study.

## Pretraining overlap limitation

TTM reports selected non-leaking historical portions of GIFT-Eval data plus
KernelSynth. MOIRAI 2 reports a subset of GIFT-Eval histories, Chronos-derived mixup,
KernelSynth, and undisclosed internal Salesforce operational data. Neither model card
provides dataset-instance evidence sufficient to exclude ETT overlap. The paper must
not claim unseen-domain generalization or contamination-free evaluation.

Recommended limitation wording:

> Zero-Shot denotes inference without additional parameter updates on the designated
> target-data train split. Public pretraining disclosures do not permit us to exclude
> overlap with ETT or closely related benchmark series.

## Colab acceptance gate

Run each notebook in a fresh Colab Pro+ runtime. A model may advance only after the
notebook records the pinned revision, resolved packages, synthetic multivariate
shape, finite output, deterministic repeat result, ETTh1 validation-window result,
raw-unit output, elapsed time, memory, and unchanged Zero-Shot parameter hash. TTM
must additionally pass a 1–2 optimizer-step train/validation and state round trip.
MOIRAI full fine-tuning remains disabled until the user accepts an official supported
path or changes the research design. The multivariate inference blocker also requires
an official fix/version decision before a production adapter can be implemented.

The notebooks must never use the test split, download bundle data, commit caches or
weights, or print credentials. Runtime output is evidence for review, not a paper
performance result.

## Official sources

- TTM model card: <https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3>
- TTM code: <https://github.com/ibm-granite/granite-tsfm>
- MOIRAI 2 model card: <https://huggingface.co/Salesforce/moirai-2.0-R-small>
- uni2ts code: <https://github.com/SalesforceAIResearch/uni2ts>

The machine-readable candidate files and manifests contain the exact hashes, file
sizes, warnings, and full capability matrices.
