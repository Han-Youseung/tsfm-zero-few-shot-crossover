# Common TTM–MOIRAI conditions after the CPU gate

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
| multivariate | joint `(batch, context, channel)` | joint target dimensions with `variate_id` | both validated at 7 channels |
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

Still pending are GPU/AMP validation, peak GPU memory, final `num_samples`, and the
model-specific optimizer settings selected once using validation and memory only. Until
the Colab gate passes, `model_selection_gate.json` keeps
`production_adapter_allowed=false`.
