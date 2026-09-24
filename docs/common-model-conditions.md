# Common TTM–MOIRAI conditions after the GPU gate

Phase 4.6 FP32 Colab evidence passed for both models. See
[Colab GPU validation](colab-gpu-validation.md) for the separate mandatory FP32 matrix,
100-sample MOIRAI inference checks, historical TTM CPU training-split caveat, and result import.
GPU compatibility does not freeze the research protocol or select its metrics/budgets.
The adopted 12 conditions and official channel/loss review are in
[GPU evidence review](gpu-evidence-review.md). Adapter GPU execution is a separate check.

The comparison unit is model–dataset–horizon. Both models use the same 60/20/20
chronological split, selected train-window manifests, validation/test origins, metrics,
original-unit evaluation, seed policy, and optimizer-step budget rule. Test data never
choose a checkpoint, scaler, point rule, patch size, or hyperparameter.

| Condition | TTM R3 | MOIRAI 1.1-R-small | CPU status |
| --- | --- | --- | --- |
| context 512 / horizon 96 | 512/96 checkpoint, direct point head | patch 64, direct probabilistic samples | both supported; both 7-channel |
| context 512 / horizon 192 | 512/336 checkpoint, crop direct output to 192 | patch 64, direct 192 samples | supported; TTM crops |
| context 512 / horizon 336 | 512/336 checkpoint, direct point head | patch 64, direct 336 samples | both supported |
| context 512 / horizon 720 | official zero-pad to 1024/720 checkpoint | patch 64, direct 720 samples | supported; TTM pads context |
| multivariate | `(batch, context, channel)`, channel-independent shared weights | joint target dimensions with `variate_id` | both validated at 7 channels; interaction differs |
| full fine-tuning | all checkpoint parameters | `MoiraiFinetune`, `finetune_pattern=full` | one CPU step each validated |
| point forecast | deterministic point head | sample median, provisional 100 samples | model-native rules differ |
| scaling | native standard scaling | native `PackedStdScaler` | external scaling disabled |

Architectures do not need identical internal scaling or output distributions. Fairness
requires each model to keep its own scaling and point rule unchanged between Zero-Shot
and every Few-Shot rate. Metrics are computed after outputs return to original units.

The TTM 192 condition uses a direct longer head followed by deterministic cropping; its
720 condition uses the official zero-padding path to a 1024-context checkpoint. MOIRAI
uses one pinned checkpoint at every horizon and right-pads incomplete prediction
patches internally. These disclosed model-native support differences are not reasons to
remove a horizon. They mean that “same context length” refers to 512 observed values
supplied by the experiment, not identical internal tokenization.

AMP, final `num_samples`, metrics/aggregation and model-specific optimizer settings
remain pending pilot decisions. Both 8- and 100-sample MOIRAI inference passed;
100 samples is still provisional. `production_adapter_allowed=true` permits
implementation, not an assertion that the new adapter has run on GPU.
