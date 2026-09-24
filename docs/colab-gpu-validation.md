# Phase 4.6: Colab GPU compatibility

## Status and scope

Preparation only. Local Windows has no accessible CUDA GPU; the inspected model environment
has torch 2.4.1+cpu. Actual Colab installation is **installation_pending**, GPU evidence is
**pending_gpu**, AMP is **not_run**, and production adapters remain blocked. No GPU timings,
memory measurements, successful predictions, training results or restores are claimed.

This gate covers official raw ETTh1, seven jointly processed channels, context 512,
batch size one, and horizons 96/192/336/720. It is not a full-dataset feasibility test,
metric benchmark, crossover analysis, hyperparameter search or protocol freeze.

The historical CPU manifests and MOIRAI 2.0 incompatibility evidence are preserved.
Review found that the historical TTM CPU probe used a validation window for its one-step
training smoke. Its old manifest is retained as historical evidence, **not** accepted as
leakage-safe GPU training evidence. The probe now uses a train window. Phase 4.6 independently
validates all four horizons, with official WindowIndex IDs and no test split access.

## Pinned model sources

| Model / horizon | Checkpoint selection name | Immutable HF revision |
| --- | --- | --- |
| TTM 96 | 512-96-dec-512-r3 | 7b070728ff280ee2bb682bd8dfbe71da21ca2c7c |
| TTM 192, 336 | 512-336-dec-512-r3 | 81f34486a7d43de21f847df1262621c53c525314 |
| TTM 720 | 1024-720-r3 | aa610521c42d25c3a1ba03ad0d3d1e637b2c8d08 |
| MOIRAI all | Salesforce/moirai-1.1-R-small | 0c24ab99db2c1a70ea2a0fc03bf113329772ac64 |

TTM code: fe7a35697723e2a2f5246ae979474bfc554e26c0, granite-tsfm 0.3.9.
Uni2TS code: cfd46d4510ed8896f263116f32928eede05b0a75, version 2.0.0.
The distinct Uni2TS release-tag commit is not substituted. Installed PEP 610 commit IDs,
package versions, resolved packages and individual downloaded config SHA256 are checked.
CPU dependency snapshots are not applied to CUDA environments. CUDA wheel dependency
resolution and imports must succeed; pending installation does not mean Colab is verified.
The notebook installs the Colab `python3.12-venv` system component before creating each
isolated environment because the pinned runtime image does not include `ensurepip` by default.

TTM uses its official prediction_outputs and loss path, including prediction_filter_length
192 for the 336-output checkpoint. The 720 condition prepends 512 zero rows to the native
1024-context checkpoint. This matches the official get_model zeropad caller contract and
ts_padding left-zero behavior; no source patch or recursive forecast is used.
Native model scaling sees the padded input, just as in that official padding route.
MOIRAI uses joint seven-channel samples, patch 64, PackedStdScaler, MoiraiFinetune and
PackedNLLLoss. Objectives differ between models; neither objective is replaced by a common
custom loss. TTM point head and MOIRAI torch sample median remain provisional.

## Execute in Colab

Open `notebooks/10_ttm_compatibility.ipynb` for TTM and
`notebooks/12_moirai1_compatibility.ipynb` for MOIRAI. Choose Runtime → Change runtime type →
GPU. Use Python 3.11 or 3.12. Run one model per fresh session (Disconnect and delete runtime
before switching models), or the separately named virtual environments/processes supplied.
No GitHub write authorization or model token is required.

Execute all cells top-to-bottom:

1. Check actual NVIDIA GPU and kernel Python; do not load kernel torch.
2. Enter the published phase-4.6 **full commit SHA**, clone read-only, check clean status,
   detach at that exact commit. No git pull during the run.
3. Choose temporary output or user-mounted Drive **before** executing conditions.
4. Create model-only venv; install its GPU requirements and run pip check.
5. Verify the same venv interpreter, actual CUDA torch and official source commit.
   The notebook kernel is only orchestration; installation and every model process use PY.
6. Download only official raw ETTh1 and verify the registered SHA256.
7. Run four FP32 conditions. Each subprocess verifies model config/revision, runs synthetic
   and validation Zero-Shot checks, reloads pretrained weights for one train step, then
   restores a new model and optimizer and compares validation predictions.
8. MOIRAI only: run four additional 100-sample inference conditions. Eight-sample success
   does not establish 100-sample feasibility.
9. Validate JSON summaries and download the small ZIP. AMP stays not_run; there is no
   automatic selection of a global precision policy.

The existing probe entry points accept --gpu-gate, --horizon, --num-samples,
--expected-commit, --data, --cache-dir and --output. No production adapter or duplicate
runner is introduced. One process per condition avoids accumulating model/optimizer VRAM.
Each training condition reloads its original pretrained checkpoint; no horizon inherits
another condition's fine-tuned weights.

## Evidence and success criteria

Zero-Shot requires CUDA parameters and input, eval/no-grad, finite expected shape, unchanged
parameter hash, seed repeat and channel permutation check. No optimizer is created until
after Zero-Shot completes. MOIRAI records samples [1,S,H,7] and point [1,H,7]. TTM records
[1,H,7]. Permutation checks use the official predictive distribution mean for MOIRAI,
not differently ordered random samples. Comparisons use rtol=1e-4, atol=1e-5, not bitwise equality.

PyTorch does not provide a strict deterministic CUDA implementation for the indices output
of `median`, although decomposed TTM consumes only `.values`. The first A100 attempt at commit
`129ae0397723b8fd2c9d4686af5799aac3405c54` therefore stopped horizons 96/192/336 at the
synthetic forward while the standard 720 model passed. The gate now uses deterministic
algorithms in `warn_only` mode and records that setting. Seeded repeat predictions must still
pass rtol=1e-4 and atol=1e-5; warnings are not treated as proof of reproducibility.

Full training requires all model parameters to have requires_grad and optimizer membership,
finite loss and gradients, an actual optimizer step and some parameter changes. Missing
gradient tensor names and changed parameter tensor counts are reported separately;
not every element must change in one step. Smoke learning rates are implementation checks
only (TTM AdamW 1e-6; MOIRAI official AdamW 5e-7), not selected experimental budgets.

Temporary trusted checkpoints contain model, optimizer, global step, identity/config,
Python/NumPy/torch CPU/all-CUDA RNG and a null GradScaler (FP32). Restore uses newly created
model and optimizer objects. Hashes, optimizer state, RNG, config, step and seeded validation
predictions must agree. These checkpoints are never exported or committed.

Inference and training measurements synchronize CUDA and reset peak allocated/reserved
memory at boundaries. Installation and downloads occur outside measurements. The synthetic
forward is disclosed as inference warm-up; the first validation shape is timed. Training
has no warm-up. Every result includes actual environment, seed/dtype, shape and condition
identity. Absolute process paths and arbitrary exception messages are excluded from portable
JSON to prevent credential/path leakage; failure stage and exception class remain available.

Failures are separated into dependency installation, GPU/runtime, data/config identity,
out of memory, official API/shape, nonfinite loss/gradient, restore and reproducibility.
Infrastructure failure is not model rejection. Failed attempts are retained; never substitute
MOIRAI 1.0, channelwise prediction, PEFT, custom loss or recursive forecasting automatically.

## Persistence, resume and return

Each completed condition is atomically written. An existing passed condition is skipped only
after complete schema and identity validation. A failed/pending prior attempt is archived as
another small JSON before a retry. A killed in-flight condition may have no JSON and must rerun.
Download results immediately when using temporary disk: **Colab temporary storage does not
survive session deletion**. For persistent progress, the user must execute the optional Drive
mount cell and select USE_DRIVE before the runs. Drive authentication is never embedded.

Download `ttm-gpu-results.zip` and `moirai1-gpu-results.zip`. Extract into ignored
`results/raw/gpu-return/`. Review installation-*.json separately. Validate/import condition
files together with the actual execution commit (not today's HEAD if different):

```text
python -m tsfm_crossover.models.gpu_gate --expected-commit FULL_EXECUTION_SHA --import-results CONDITION_JSON_1 CONDITION_JSON_2 ...
```

The importer validates immutable model/config/code/project identity, official data fingerprint,
exact registered train/validation windows and CUDA evidence. Imported evidence goes to
`results/manifests/models/gpu_runs/<execution-commit>/`; old CPU manifests and failed results
are never overwritten. It prints the aggregate decision but does not automatically mutate
the selected model or freeze research. Pass all eight required FP32 conditions from one
execution commit to open the compatibility gate; review and commit the evidence and selection
decision in a separate result-return step. The four 100-sample results have their own status.

## Still reserved for the subsequent pilot

MSE/MAE primary metric and point statistic; heterogeneous-channel and cross-dataset aggregation;
sample count and stochastic seeds; batch size, optimizer, learning rate and step/epoch budgets;
precision policy; interpretation of TTM checkpoint differences and padding/crop. Test data
must never choose any of these. Native scaling only, no external StandardScaler, original-unit
predictions, and the same Zero-Shot/fine-tuning scaling path remain in force.
