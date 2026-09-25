"""Bounded MOIRAI memory experiment; no vendor patches or automatic protocol promotion."""

import argparse
import gc
import json
import os
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.experiments.pilot import measure, prepare_condition
from tsfm_crossover.experiments.pilot_config import PilotConfig
from tsfm_crossover.models.adapters import create_adapter
from tsfm_crossover.models.contract import parameter_hash
from tsfm_crossover.models.gpu_smoke import equal_tree, rng_state
from tsfm_crossover.tracking.atomic import write_json_atomic

PROFILES = ("fp32", "fp32_cpu_saved", "bf16", "bf16_cpu_saved")


def availability(profile, cuda, bf16, available_ram_gib):
    if profile not in PROFILES:
        raise ValueError("unknown memory profile")
    if not cuda:
        return "pending_gpu"
    if profile.startswith("bf16") and not bf16:
        return "bf16_not_supported"
    if profile.endswith("cpu_saved") and available_ram_gib < 64:
        return "insufficient_host_ram_for_conservative_offload_gate"
    return None


def amp_context(profile):
    import torch

    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if profile.startswith("bf16")
        else nullcontext()
    )


def training_step(adapter, item, profile):
    import torch

    offload = (
        torch.autograd.graph.save_on_cpu(pin_memory=False)
        if profile.endswith("cpu_saved")
        else nullcontext()
    )
    with offload:
        return adapter.train_step(item, forward_context=amp_context(profile))


def run(root, dataset, profile, output, commit):
    import psutil
    import torch

    prepared = json.loads(
        (root / "results/manifests/pilot/prepared_primary.json").read_text(encoding="utf-8")
    )
    identity = {
        "commit": commit,
        "dataset": dataset,
        "profile": profile,
        "fingerprint": prepared[dataset]["qc"]["sha256"],
        "context": 512,
        "horizon": 720,
        "samples": 100,
        "batch": 1,
        "seed": 1729,
        "steps": 2,
        "family": "moirai1",
    }
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise ValueError("refusing to mix runs")
        return old  # Preserve failures as well as successes; retry in a new directory.
    cuda = torch.cuda.is_available()
    host_ram = psutil.virtual_memory().available / 2**30
    bf16 = cuda and torch.cuda.is_bf16_supported()
    record = {
        "identity": identity,
        "status": "running",
        "stage": "preflight",
        "test_evaluation": False,
        "protocol_frozen": False,
        "main_experiment_allowed": False,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if cuda else None,
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else None,
            "available_host_ram_gib": host_ram,
            "bf16_supported": bf16,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip()
            if cuda
            else None,
        },
        "precision": {
            "parameters": "float32",
            "autocast": "bfloat16" if profile.startswith("bf16") else None,
            "grad_scaler": None,
            "saved_tensors_on_cpu": profile.endswith("cpu_saved"),
            "pin_memory": False,
            "deterministic_warn_only": True,
            "tf32": False,
        },
        "checks": {},
        "timing": {},
    }
    reason = availability(profile, cuda, bf16, host_ram)
    if reason:
        record.update(status="not_run", reason=reason)
        write_json_atomic(output, record)
        return record
    write_json_atomic(output, record)
    adapter = fresh = None
    try:
        config = PilotConfig(max_steps=2, learning_rates={"ttm": [1e-4], "moirai1": [5e-6]})
        row = {"family": "moirai1", "horizon": 720, "id": stable_hash(identity)}
        settings, train, valid, batch, _, data = prepare_condition(
            config, row, prepared[dataset], root, commit
        )
        record["data"] = data
        adapter = create_adapter(settings, root=root, device="cuda")
        adapter.load_model()
        record["metadata"] = adapter.execution_metadata()
        record["checks"]["cuda_parameters"] = all(
            p.device.type == "cuda" for p in adapter.model.parameters()
        )
        record["checks"]["cuda_input"] = (
            adapter.prepare_batch(batch(valid[:1])).past.device.type == "cuda"
        )
        before = parameter_hash(adapter.parameter_state())

        def predict(model):
            with amp_context(profile):
                return model.predict(batch(valid[:1]))

        record["stage"] = "zero_shot"
        write_json_atomic(output, record)
        point, timing = measure(lambda: predict(adapter), "cuda")
        record["prediction"] = {
            "shape": list(point.shape),
            "device": str(point.device),
            "dtype": str(point.dtype),
        }
        record["timing"]["inference"] = {**timing, "warm_up": False}
        record["checks"]["finite_shape"] = list(point.shape) == [
            1,
            720,
            len(settings.channel_names),
        ] and bool(torch.isfinite(point).all())
        repeated = predict(adapter)
        record["checks"]["seed_repeat"] = bool(
            torch.allclose(point, repeated, atol=1e-5, rtol=1e-4)
        )
        record["checks"]["zero_shot_unchanged"] = (
            before == parameter_hash(adapter.parameter_state()) and adapter.optimizer is None
        )
        if dataset == "ETTh1" and profile.startswith("bf16"):
            reference = adapter.predict(batch(valid[:1]))
            difference = (point.float() - reference.float()).abs()
            record["bf16_vs_fp32_prediction"] = {
                "max_absolute_error": float(difference.max()),
                "mean_absolute_error": float(difference.mean()),
                "interpretation": "diagnostic only; no precision policy promotion",
            }
            del reference, difference
        del point, repeated
        adapter.configure_finetuning()
        training_before = parameter_hash(adapter.parameter_state())
        if dataset == "ETTh1" and profile.startswith("bf16"):
            reference_loss = adapter.validation_step(batch(valid[:1]))
            with amp_context(profile):
                mixed_loss = adapter.validation_step(batch(valid[:1]))
            record["bf16_vs_fp32_official_validation_loss"] = {
                "fp32": reference_loss,
                "bf16": mixed_loss,
                "absolute_difference": abs(reference_loss - mixed_loss),
            }
        record["stage"] = "training"
        write_json_atomic(output, record)
        record["losses"] = []
        for step in range(2):
            loss, timing = measure(
                lambda: training_step(adapter, batch(train[:1]), profile), "cuda"
            )
            record["losses"].append(loss)
            record["timing"][f"training_{step + 1}"] = timing
            record["host_rss_gib"] = psutil.Process().memory_info().rss / 2**30
            write_json_atomic(output, record)
        record["checks"]["update"] = training_before != parameter_hash(adapter.parameter_state())
        record["checks"]["steps"] = adapter.global_step == 2
        record["checks"]["full_parameter_coverage"] = all(
            p.requires_grad for p in adapter.model.parameters()
        ) and {id(p) for p in adapter.model.parameters()} == {
            id(p) for g in adapter.optimizer.param_groups for p in g["params"]
        }
        record["checks"]["finite_gradients"] = any(
            p.grad is not None for p in adapter.model.parameters()
        ) and all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in adapter.model.parameters()
        )
        record["missing_gradient_parameters"] = adapter.last_gradient_missing
        record["metadata"] = adapter.execution_metadata()
        record["stage"] = "restore"
        write_json_atomic(output, record)
        expected = predict(adapter).detach().cpu()
        record["checks"]["validation_after_training_finite"] = bool(torch.isfinite(expected).all())
        expected_hash = parameter_hash(adapter.parameter_state())
        with tempfile.TemporaryDirectory(prefix="traffic-memory-") as directory:
            checkpoint = Path(directory) / "state.pt"
            adapter.save_training_state(checkpoint, loop_state={"memory_profile": profile})
            adapter.cleanup()
            adapter = None
            gc.collect()
            torch.cuda.empty_cache()
            fresh = create_adapter(settings, root=root, device="cuda")
            fresh.load_model()
            fresh.load_training_state(checkpoint)
            record["checks"]["restore_memory_profile"] = fresh.loaded_loop_state == {
                "memory_profile": profile
            }
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            record["checks"]["restore_rng"] = equal_tree(saved["rng"], rng_state())
            record["checks"]["restore_optimizer"] = equal_tree(
                saved["optimizer"], fresh.optimizer.state_dict()
            )
            record["checks"]["restore_hash"] = (
                parameter_hash(fresh.parameter_state()) == expected_hash
            )
            record["checks"]["restore_step"] = fresh.global_step == 2
            restored = predict(fresh).detach().cpu()
            record["restore_max_absolute_error"] = float((expected - restored).abs().max())
            record["restore_tolerance"] = {"atol": 1e-5, "rtol": 1e-4}
            record["checks"]["restore_prediction"] = bool(
                torch.allclose(expected, restored, atol=1e-5, rtol=1e-4)
            )
        record["status"] = "passed" if all(record["checks"].values()) else "failed"
        record["stage"] = "complete"
    except Exception as exc:
        record.update(
            status="failed",
            error={
                "type": type(exc).__name__,
                "message": str(exc)[:700],
                "category": "out_of_memory"
                if isinstance(exc, torch.OutOfMemoryError)
                else "runtime_or_api",
            },
        )
    finally:
        for model in (adapter, fresh):
            if model is not None:
                model.cleanup()
        write_json_atomic(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ETTh1", "Traffic"], required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        commit != args.expected_commit
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ):
        raise ValueError("clean pinned execution commit required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(run(Path.cwd(), args.dataset, args.profile, args.output, commit)["status"])


if __name__ == "__main__":
    main()
