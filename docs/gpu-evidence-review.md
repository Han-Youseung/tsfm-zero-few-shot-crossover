# Returned Colab evidence and official source review

Adopted execution: `1bb93b80b0c22c990cd2b5f7d9ff2a125b21e7c5`.
The twelve records comprise TTM four FP32 train/restore conditions, MOIRAI four
eight-sample FP32 train/restore conditions, and four 100-sample inference conditions.
See `results/manifests/models/gpu_evidence_review.json` for the validated inventory.
Original archive checksums and historical records are retained. CPU and MOIRAI 2.0
records are unchanged. This review verifies records against their execution source;
it is not another GPU execution.

## Scope and limits

Official raw ETTh1, 512 observed context points, 7 channels, batch 1, horizons
96/192/336/720, native scaling and FP32. The probe reads numeric rows only through
validation end. Reading the file bytes for SHA256 and recording test boundaries
does not use test observations for inference or learning. All model windows are
the registered train/validation windows. AMP remains not_run. Research remains
unfrozen. Compatibility does not establish dataset-wide feasibility or accuracy.
TTM ran on A100 80GB and MOIRAI on T4; wall times cannot rank model speed.

## Initial TTM failure and change review

The old `129ae0397723b8fd2c9d4686af5799aac3405c54` records have three failures
at the official API stage and a successful 720 condition. Their JSON contains
only exception category/class, not the full traceback. The user-supplied runtime
traceback identifies CUDA median with indices under strict deterministic mode.
The source diff switches to `torch.use_deterministic_algorithms(True, warn_only=True)`
and records that policy; it also fixes Colab venv preparation. It does not patch
vendor libraries, reshape outputs, change weights, crop/padding rules or objectives.
Seed repeats and checkpoint prediction comparisons passed at rtol=1e-4, atol=1e-5.
This does not guarantee strict determinism or cross-GPU bitwise equality.
Old conditions remain archived and cannot supply missing new-run conditions.

## TTM channel semantics

Pinned code: [modeling_tinytimemixer.py](https://github.com/ibm-granite/granite-tsfm/blob/fe7a35697723e2a2f5246ae979474bfc554e26c0/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py).
`TinyTimeMixerLayer` enables channel feature mixing only for `mode=mix_channel`.
The selected checkpoints use `common_channel` for backbone/decoder and disable
forecast channel mixing. They support a multivariate tensor with channel-independent
processing and shared parameters. Our caller supplies one `(B,L,C)` tensor, without
an external channel loop. MOIRAI uses joint target dimensions/variate IDs and attention.
The common contract is multivariate tensor input/output, not identical cross-channel
interaction. No mixing flag is enabled to conceal this architectural difference.

## TTM actual objective and point outputs

All selected configs enable `multi_quantile_head`; `loss=mae` alone is misleading.
For decomposed 96/192/336, `_choose_loss` selects `MultiPinballLoss`. The returned
loss is `w_joint*joint + w_trend*trend + w_residual*residual`; all three weights are 1.
`forecast_loss_type=joint` does not bypass this weighted sum in this forward path.
Joint and residual terms each add `point_extra_weight=2` times point L1. The trend
term additionally includes 0.1 times the L1 error of first differences. Targets use
the context-fitted native scaler and the official trend/residual target construction.
The quantile width penalty is disabled in these checkpoint configurations.

For standard 720, the returned objective is multi-quantile pinball plus 2 times
point L1, in original units after native inverse scaling. Thus objectives differ
even among TTM horizon checkpoints. Official `prediction_outputs` is a deterministic
quantile-median-derived point output (sum of component point outputs for decomposed
models), not an independently specified MSE head.

For 192, `prediction_filter_length=192` crops inside the official heads and target
path; the caller supplies exactly 192 future points. For 720, the caller left-pads
512 zeros to reach 1024; target remains exactly 720. Zero padding participates in
the official native scaling path exactly as in the validated probe.

MOIRAI retains official `MoiraiFinetune` and `PackedNLLLoss`. Both probes used
AdamW (TTM explicitly, MOIRAI official optimizer groups); these smoke settings
are not pilot decisions. Training objectives are distinct from MAE/MSE evaluation.
Any description of both models as Adam+MSE or both as cross-channel interaction
models is obsolete. MOIRAI sample median uses `torch.median` (lower middle order
statistic for even sample counts) provisionally, matching the executed probe.

## Adapter gate

Verified compatibility permits adapter implementation. New adapter execution is
separate evidence, with its own code commit and pending GPU status until executed.
Pilot decisions remain metrics/aggregation, point statistic and sample count,
learning rate/batch size, optimizer-step budget/early stopping, precision, and
high-channel-count memory feasibility.
