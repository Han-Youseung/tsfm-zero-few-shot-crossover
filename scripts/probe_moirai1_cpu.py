"""Manual MOIRAI 1.1 CPU selection probe using official Uni2TS APIs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

REPOSITORY = "Salesforce/moirai-1.1-R-small"
REVISION = "0c24ab99db2c1a70ea2a0fc03bf113329772ac64"
HORIZONS = (96, 192, 336, 720)
CONTEXT, CHANNELS, PATCH_SIZE, SMOKE_SAMPLES, SEED = 512, 7, 64, 8, 1729


def digest(model) -> str:
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def rss_mb() -> float:
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    process = get_process()
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL
    if not get_memory(process, ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize / (1024 * 1024)


def packed_batch(forecast, past, future):
    import torch

    converted = forecast._convert(
        PATCH_SIZE,
        past,
        torch.ones_like(past, dtype=torch.bool),
        torch.zeros(past.shape[:2], dtype=torch.bool, device=past.device),
        future_target=future,
        future_observed_target=torch.ones_like(future, dtype=torch.bool),
        future_is_pad=torch.zeros(future.shape[:2], dtype=torch.bool, device=future.device),
    )
    target, observed, sample_id, time_id, variate_id, prediction_mask = converted
    return {
        "target": target,
        "observed_mask": observed,
        "sample_id": sample_id,
        "time_id": time_id,
        "variate_id": variate_id,
        "prediction_mask": prediction_mask,
        "patch_size": torch.ones_like(time_id, dtype=torch.long) * PATCH_SIZE,
    }


def make_forecast(module, horizon, num_samples=SMOKE_SAMPLES):
    from uni2ts.model.moirai import MoiraiForecast

    return MoiraiForecast(
        prediction_length=horizon,
        target_dim=CHANNELS,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=CONTEXT,
        module=module,
        patch_size=PATCH_SIZE,
        num_samples=num_samples,
    )


def make_finetune(module, horizon=96):
    from uni2ts.loss.packed import PackedNLLLoss
    from uni2ts.model.moirai import MoiraiFinetune

    return MoiraiFinetune(
        min_patches=2,
        min_mask_ratio=0.15,
        max_mask_ratio=0.5,
        max_dim=128,
        num_training_steps=2,
        num_warmup_steps=0,
        module=module,
        num_samples=SMOKE_SAMPLES,
        loss_func=PackedNLLLoss(),
        lr=5e-7,
        weight_decay=0.1,
        context_length=CONTEXT,
        prediction_length=horizon,
        patch_size=PATCH_SIZE,
        finetune_pattern="full",
    )


def gpu_probe(args):
    import torch

    from tsfm_crossover.models.gpu_smoke import run

    def factory(i, cache, training):
        from uni2ts.model.moirai import MoiraiModule

        module = MoiraiModule.from_pretrained(
            i["repository"], revision=i["revision"], cache_dir=cache
        )
        return make_finetune(module, i["horizon"]) if training else module

    def underlying(model):
        return model.module if hasattr(model, "module") else model

    def predict(model, x, i):
        forecast = make_forecast(underlying(model), i["horizon"], i["num_samples"]).eval()
        return forecast(
            x,
            torch.ones_like(x, dtype=torch.bool),
            torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device),
        )

    def loss(model, x, y, i):
        packer = make_forecast(model.module, i["horizon"])
        model.train()
        return model.training_step(packed_batch(packer, x, y), 0)

    def channels(model, x, i):
        forecast = make_forecast(underlying(model), i["horizon"]).eval()
        perm = torch.tensor([6, 4, 2, 0, 1, 3, 5], device=x.device)

        def mean(context):
            distr = forecast._get_distr(
                PATCH_SIZE,
                context,
                torch.ones_like(context, dtype=torch.bool),
                torch.zeros(context.shape[:2], dtype=torch.bool, device=context.device),
            )
            return forecast._format_preds(PATCH_SIZE, distr.mean.unsqueeze(0), CHANNELS)

        a, b = mean(x), mean(x[..., perm])[..., torch.argsort(perm)]
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        return float((a - b).abs().max())

    return run(
        args,
        "moirai1",
        factory,
        predict,
        loss,
        lambda m: m.configure_optimizers()["optimizer"],
        channels,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/moirai1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    from tsfm_crossover.models.gpu_smoke import arguments

    arguments(parser, SMOKE_SAMPLES)
    args = parser.parse_args()
    if args.gpu_gate:
        if not args.output or not args.expected_commit:
            parser.error("--gpu-gate requires --output and --expected-commit")
        return gpu_probe(args)
    if args.device == "cuda":
        parser.error("CUDA evidence requires --gpu-gate (legacy CPU path is not a GPU gate)")

    import numpy as np
    import pandas as pd
    import torch
    import uni2ts
    from uni2ts.model.moirai import MoiraiModule

    expected_data_hash = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == expected_data_hash
    values = pd.read_csv(args.data).drop(columns=["date"]).to_numpy(dtype="float32")
    assert values.shape[1] == CHANNELS
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires an available CUDA device")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_end, validation_end = int(0.6 * len(values)), int(0.8 * len(values))
    validation_origin = train_end + CONTEXT
    assert validation_origin + max(HORIZONS) <= validation_end
    context = (
        torch.from_numpy(values[validation_origin - CONTEXT : validation_origin])
        .unsqueeze(0)
        .to(device)
    )

    module = MoiraiModule.from_pretrained(
        REPOSITORY, revision=REVISION, cache_dir=args.cache_dir
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    initial_hash = digest(module)
    horizon_results, peak_rss = [], rss_mb()
    for horizon in HORIZONS:
        model = make_forecast(module, horizon).eval()
        observed = torch.ones_like(context, dtype=torch.bool)
        pad = torch.zeros((1, CONTEXT), dtype=torch.bool, device=device)
        before, started = digest(model), time.perf_counter()
        torch.manual_seed(SEED)
        with torch.no_grad():
            samples = model(context, observed, pad)
        seconds = time.perf_counter() - started
        torch.manual_seed(SEED)
        with torch.no_grad():
            repeated = model(context, observed, pad)
        assert tuple(samples.shape) == (1, SMOKE_SAMPLES, horizon, CHANNELS)
        assert torch.isfinite(samples).all() and torch.equal(samples, repeated)
        assert before == digest(model)
        point = torch.median(samples, dim=1).values
        peak_rss = max(peak_rss, rss_mb())
        horizon_results.append(
            {
                "horizon": horizon,
                "direct_output": True,
                "recursive": False,
                "sample_shape": list(samples.shape),
                "point_shape": list(point.shape),
                "finite": True,
                "seed_repeat_equal": True,
                "parameter_hash_unchanged": True,
                "inference_seconds": seconds,
                "peak_process_rss_mb": peak_rss,
                "patch_size": PATCH_SIZE,
                "token_sequence_length": (
                    ((CONTEXT + PATCH_SIZE - 1) // PATCH_SIZE)
                    + ((horizon + PATCH_SIZE - 1) // PATCH_SIZE)
                )
                * CHANNELS,
                "padding": {
                    "context_left": (-CONTEXT) % PATCH_SIZE,
                    "prediction_right": (-horizon) % PATCH_SIZE,
                },
                "cropping": False,
            }
        )

    permutation = torch.tensor([6, 4, 2, 0, 1, 3, 5], device=device)
    inverse = torch.argsort(permutation)
    model = make_forecast(module, 96).eval()
    observed, pad = (
        torch.ones_like(context, dtype=torch.bool),
        torch.zeros((1, CONTEXT), dtype=torch.bool, device=device),
    )
    with torch.no_grad():
        original_distr = model._get_distr(PATCH_SIZE, context, observed, pad)
        permuted_distr = model._get_distr(PATCH_SIZE, context[..., permutation], observed, pad)
        original_mean = model._format_preds(PATCH_SIZE, original_distr.mean.unsqueeze(0), CHANNELS)
        permuted_mean = model._format_preds(PATCH_SIZE, permuted_distr.mean.unsqueeze(0), CHANNELS)
    permutation_difference = float((original_mean - permuted_mean[..., inverse]).abs().max())

    train_past = torch.from_numpy(values[:CONTEXT]).unsqueeze(0).to(device)
    train_future = torch.from_numpy(values[CONTEXT : CONTEXT + 96]).unsqueeze(0).to(device)
    finetune_module = MoiraiModule.from_pretrained(
        REPOSITORY, revision=REVISION, cache_dir=args.cache_dir
    ).to(device)
    finetune = make_finetune(finetune_module)
    packer = make_forecast(finetune_module, 96)
    train_batch = packed_batch(packer, train_past, train_future)
    val_future = (
        torch.from_numpy(values[validation_origin : validation_origin + 96]).unsqueeze(0).to(device)
    )
    val_batch = packed_batch(packer, context, val_future)
    optimizer = finetune.configure_optimizers()["optimizer"]
    trainable = sum(p.numel() for p in finetune.parameters() if p.requires_grad)
    assert trainable == parameter_count
    before_train, started = digest(finetune), time.perf_counter()
    loss = finetune.training_step(train_batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in finetune.parameters()
    )
    assert gradients_finite
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_seconds, after_train = time.perf_counter() - started, digest(finetune)
    assert before_train != after_train
    finetune.eval()
    with torch.no_grad():
        validation_loss = finetune.validation_step(val_batch, 0)
    assert torch.isfinite(validation_loss)
    trained_forecast = make_forecast(finetune.module, 96).eval()
    validation_observed = torch.ones_like(context, dtype=torch.bool)
    validation_pad = torch.zeros((1, CONTEXT), dtype=torch.bool, device=device)
    torch.manual_seed(SEED)
    with torch.no_grad():
        trained_validation_prediction = trained_forecast(
            context, validation_observed, validation_pad
        )
    peak_rss = max(peak_rss, rss_mb())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.cache_dir) as temporary:
        checkpoint = Path(temporary) / "state.ckpt"
        torch.save({"state_dict": finetune.state_dict()}, checkpoint)
        restored = make_finetune(
            MoiraiModule.from_pretrained(
                REPOSITORY, revision=REVISION, cache_dir=args.cache_dir
            ).to(device)
        )
        restored.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"]
        )
        assert digest(restored) == after_train
        restored_forecast = make_forecast(restored.module, 96).eval()
        torch.manual_seed(SEED)
        with torch.no_grad():
            restored_validation_prediction = restored_forecast(
                context, validation_observed, validation_pad
            )
        checkpoint_prediction_equal = torch.equal(
            trained_validation_prediction, restored_validation_prediction
        )
        assert checkpoint_prediction_equal

    payload = {
        "python": platform.python_version(),
        "uni2ts": uni2ts.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "repository": REPOSITORY,
        "revision": REVISION,
        "device": str(device),
        "dtype": "torch.float32",
        "source_variant": "ETTh1__official_raw",
        "zero_shot_split": "validation",
        "finetune_split": "train",
        "test_split_used": False,
        "target_dim": CHANNELS,
        "batch_size": 1,
        "context_length": CONTEXT,
        "patch_size": PATCH_SIZE,
        "patch_size_selection": "official small LSF setting; not test-selected",
        "num_samples_production_candidate": 100,
        "num_samples_cpu_smoke": SMOKE_SAMPLES,
        "point_forecast": "sample median",
        "parameter_count": parameter_count,
        "initial_parameter_hash": initial_hash,
        "horizons": horizon_results,
        "channel_permutation_statistic": "official predictive distribution mean",
        "channel_permutation_max_absolute_difference": permutation_difference,
        "channel_permutation_equivariant": permutation_difference <= 1e-5,
        "finetune": {
            "wrapper": "uni2ts.model.moirai.MoiraiFinetune",
            "objective": "uni2ts.loss.packed.PackedNLLLoss",
            "pattern": "full",
            "optimizer": "official AdamW from configure_optimizers",
            "optimizer_steps": 1,
            "train_loss": float(loss.detach()),
            "validation_loss": float(validation_loss.detach()),
            "finite_gradients": gradients_finite,
            "parameter_updated": True,
            "checkpoint_round_trip": True,
            "checkpoint_validation_prediction_equal": checkpoint_prediction_equal,
            "trainable_parameters": trainable,
            "total_parameters": parameter_count,
            "train_seconds": train_seconds,
        },
        "scaling": {
            "model_native": "PackedStdScaler",
            "external_scaler": False,
            "output_rescaled_by_distribution": True,
        },
        "peak_process_rss_mb": peak_rss,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else None,
            "amp_available": device.type == "cuda",
            "amp_executed": False,
        },
        "gpu_validated": device.type == "cuda",
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
