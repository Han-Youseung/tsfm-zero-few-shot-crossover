"""Bounded, validation-only pilot over existing model/data interfaces."""

import argparse
import json
import math
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.coverage import compute_coverage
from tsfm_crossover.data.missing import POLICY, causal_context, eligible_windows
from tsfm_crossover.data.pilot_data import load_pilot_values
from tsfm_crossover.data.sampling import build_sampling_manifest
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.evaluation.metrics import StreamingMetrics, relative_improvement, train_scale
from tsfm_crossover.experiments.pilot_config import PilotConfig, execution_plan, token_risk
from tsfm_crossover.experiments.pilot_state import SamplerCursor, check_completed, condition_lock
from tsfm_crossover.models.adapter_config import AdapterConfig
from tsfm_crossover.models.adapters import create_adapter
from tsfm_crossover.models.contract import parameter_hash
from tsfm_crossover.models.gpu_smoke import equal_tree, rng_state
from tsfm_crossover.models.window_batch import make_window_batch, select_train_windows
from tsfm_crossover.tracking.atomic import write_csv_atomic, write_json_atomic


def validation_subset(windows, count):
    """Deterministic evenly spaced origin strata, never selected by observed scores."""
    count = min(count, len(windows))
    if count < 1:
        raise ValueError("no validation windows")
    indices = [
        min(len(windows) - 1, (2 * i + 1) * len(windows) // (2 * count)) for i in range(count)
    ]
    return [windows[i] for i in indices]


def prepare_condition(config, row, entry, root, commit):
    values, channels, split = load_pilot_values(root, entry)
    fingerprint = entry["qc"]["sha256"]
    candidates = generate_train_windows(
        entry["variant"], fingerprint, split.train, 512, row["horizon"]
    )
    unfiltered_train_count = len(candidates)
    context = causal_context(values) if entry.get("missing_policy") == POLICY else values
    if entry.get("missing_policy") == POLICY:
        candidates = eligible_windows(candidates, values, context, training=True)
    manifest = build_sampling_manifest(
        candidates,
        dataset_sha256=fingerprint,
        split_config_hash=split.split_config_hash,
        seed=config.seed,
        code_commit_sha=commit,
        rates=[config.sampling_rate],
    )
    selected = select_train_windows(candidates, manifest, config.sampling_rate)
    validating = validation_subset(
        eligible_windows(
            generate_rolling_windows(
                entry["variant"], fingerprint, "validation", split.validation, 512, row["horizon"]
            ),
            values,
            context,
            training=False,
        ),
        config.validation_windows,
    )

    def batch(windows):
        return make_window_batch(
            values,
            windows,
            splits=split,
            channel_names=channels,
            fingerprint=fingerprint,
            sampling_manifest_hash=manifest.manifest_hash,
            context_values=context,
        )

    settings = AdapterConfig(
        family=row["family"],
        condition_id=row["id"],
        dataset_fingerprint=fingerprint,
        sampling_manifest_hash=manifest.manifest_hash,
        channel_names=channels,
        horizon=row["horizon"],
        num_samples=1 if row["family"] == "ttm" else config.num_samples,
        prediction_seed=config.seed,
        training_seed=config.seed,
        learning_rate=row.get("learning_rate", config.learning_rates[row["family"]][0]),
        weight_decay=0.01 if row["family"] == "ttm" else 0.1,
        max_optimizer_steps=config.max_steps,
        point_statistic="prediction_outputs" if row["family"] == "ttm" else "torch_sample_median",
    )
    scale = train_scale(values, split.train.end)
    provenance = {
        "variant": entry["variant"],
        "fingerprint": fingerprint,
        "split": split.as_dict(),
        "sampling_manifest": manifest.as_dict(),
        "validation_windows": [w.as_dict() for w in validating],
        "validation_window_hash": stable_hash([w.window_id for w in validating]),
        "coverage": compute_coverage(selected, (split.train.start, split.train.end)).as_dict(),
        "selected_train_windows": len(selected),
        "total_train_windows": len(candidates),
        "unfiltered_train_windows": unfiltered_train_count,
        "missing_policy": entry.get("missing_policy", "complete_observations"),
        "requested_sampling_rate": config.sampling_rate,
        "effective_sampling_rate": len(selected) / len(candidates),
        "test_targets_read": False,
        "metric_train_scale": asdict(scale),
    }
    return settings, selected, validating, batch, scale, provenance


def measure(function, device):
    import torch

    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = function()
    if cuda:
        torch.cuda.synchronize()
    timing = {
        "seconds": time.perf_counter() - start,
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
    }
    return value, timing


def evaluate(adapter, windows, batch, scale):
    # Canonical window-at-a-time prediction fixes RNG assignment irrespective of
    # throughput-calibration batches. Both Zero-Shot and Few-Shot use this same rule.
    metrics = StreamingMetrics(scale)
    before = rng_state()
    for window in windows:
        item = batch([window])
        prediction = adapter.predict(item).detach().cpu().double().reshape(-1, len(scale.std))
        targets = item.future[0]
        metrics.update(prediction.tolist(), targets)
    if not equal_tree(before, rng_state()):
        raise RuntimeError("validation advanced training RNG")
    return metrics.compute()


def stability(
    adapter, config, selected, validating, batch, scale, directory, identity, provenance, device
):
    """A checkpoint atomically contains both model and loop/sampler state."""
    last = directory / "last.pt"
    if last.exists():
        adapter.load_training_state(last)
        state = adapter.loaded_loop_state
        if state is None or state["identity"] != identity:
            raise ValueError("pilot checkpoint identity mismatch")
        sampler = SamplerCursor(**state["sampler"])
        if state["step"] != adapter.global_step:
            raise ValueError("sampler/model step transaction mismatch")
    else:
        original = parameter_hash(adapter.parameter_state())
        zero, timing = measure(lambda: evaluate(adapter, validating, batch, scale), device)
        if original != parameter_hash(adapter.parameter_state()) or adapter.optimizer is not None:
            raise RuntimeError("Zero-Shot changed parameters or created optimizer")
        adapter.configure_finetuning()
        sampler = SamplerCursor(len(selected), config.seed)
        state = {
            "identity": identity,
            "step": 0,
            "sampler": sampler.state(),
            "zero_shot": zero,
            "initial_inference": timing,
            "best_metric": None,
            "best_step": None,
            "best_checkpoint": None,
            "bad_evals": 0,
            "history": [],
            "loss_history": [],
            "visited_train_indices": [],
            "train_seconds": 0.0,
            "inference_seconds": timing["seconds"],
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }
        adapter.save_training_state(last, loop_state=state)
    while adapter.global_step < config.max_steps and state["bad_evals"] < config.patience_evals:
        indices = sampler.next(config.training_batch)
        loss, timing = measure(
            lambda indices=indices: adapter.train_step(batch([selected[i] for i in indices])),
            device,
        )
        state["step"] = adapter.global_step
        state["sampler"] = sampler.state()
        state["visited_train_indices"] = sorted(set(state["visited_train_indices"]) | set(indices))
        state["train_seconds"] += timing["seconds"]
        state["loss_history"].append(
            {"step": adapter.global_step, "official_loss": loss, "actual_batch_size": len(indices)}
        )
        for key in ("allocated", "reserved"):
            state[f"peak_{key}_mb"] = max(
                state[f"peak_{key}_mb"], timing[f"max_memory_{key}_mb"] or 0
            )
        if adapter.global_step % config.eval_every == 0 or adapter.global_step == config.max_steps:
            metrics, timing = measure(lambda: evaluate(adapter, validating, batch, scale), device)
            diagnostic = next(
                (
                    w
                    for w in validating
                    if all(math.isfinite(v) for row in batch([w]).future[0] for v in row)
                ),
                None,
            )
            objective = adapter.validation_step(batch([diagnostic])) if diagnostic else None
            score = metrics["macro"][config.selection_metric]
            if score is None or not math.isfinite(score):
                raise ValueError("selection metric undefined; no checkpoint chosen")
            state["inference_seconds"] += timing["seconds"]
            state["history"].append(
                {
                    "step": adapter.global_step,
                    "metrics": metrics,
                    "official_validation_loss_first_window": objective,
                    "timing": timing,
                }
            )
            if state["best_metric"] is None or score < state["best_metric"]:
                state["best_metric"], state["best_step"], state["bad_evals"] = (
                    score,
                    adapter.global_step,
                    0,
                )
                state["best_checkpoint"] = f"best-step-{adapter.global_step}.pt"
                adapter.save_training_state(directory / state["best_checkpoint"], loop_state=state)
            else:
                state["bad_evals"] += 1
        if (
            adapter.global_step % config.checkpoint_every == 0
            or adapter.global_step % config.eval_every == 0
            or adapter.global_step == config.max_steps
        ):
            adapter.save_training_state(last, loop_state=state)
    best = next((r["metrics"] for r in state["history"] if r["step"] == state["best_step"]), None)
    return {
        "status": "completed",
        "identity": identity,
        "data": provenance,
        "loop": state,
        "best_validation": best,
        "relative_improvement": {
            k: relative_improvement(state["zero_shot"]["macro"][k], v)
            for k, v in best["macro"].items()
        }
        if best
        else None,
        "equivalent_epochs": adapter.window_exposures / len(selected),
        "average_window_exposures": adapter.window_exposures / len(selected),
        "actual_unique_train_windows": len(state["visited_train_indices"]),
        "stopping_step": adapter.global_step,
        "best_validation_step": state["best_step"],
        "convergence": "not_established; bounded pilot only",
        "metadata": adapter.execution_metadata(),
        "microbatch": config.training_batch,
        "accumulation": 1,
        "nominal_effective_batch": config.training_batch,
        "drop_last": False,
        "protocol_frozen": False,
        "amp": "not_run",
        "test_evaluation": False,
    }


def feasibility(factory, config, selected, validating, batch, directory, identity, device):
    import torch

    if device != "cuda":
        raise ValueError("throughput feasibility requires CUDA; CPU is not GPU evidence")

    attempts = []
    for operation in ("inference", "training"):
        for size in config.batch_candidates:
            dest = directory / f"{operation}-b{size}.json"
            if dest.exists():
                result = json.loads(dest.read_text())
                if result["identity"] != identity:
                    raise ValueError("batch evidence identity mismatch")
            else:
                adapter = factory()
                point = single = None
                result = {
                    "identity": identity,
                    "batch_size": size,
                    "operation": operation,
                    "dtype": "float32",
                    "status": "failed",
                }
                try:
                    adapter.load_model()
                    windows = selected if operation == "training" else validating
                    if len(windows) < size:
                        result.update(status="not_run", reason="insufficient distinct windows")
                    else:
                        item = batch(windows[:size])
                        # Warm-up only forward, never an unrecorded optimizer update.
                        adapter.predict(batch(validating[:1]))
                        if operation == "training":
                            adapter.configure_finetuning()
                            _, timing = measure(lambda a=adapter, b=item: a.train_step(b), device)
                        else:
                            point, timing = measure(lambda a=adapter, b=item: a.predict(b), device)
                            single = adapter.predict(batch(windows[:1]))
                            result["batching_first_window_max_abs_difference"] = float(
                                (point[:1] - single).abs().max()
                            )
                        result.update(
                            status="passed",
                            timing=timing,
                            warmup="one validation forward",
                            windows_per_second=size / timing["seconds"],
                            metadata=adapter.execution_metadata(),
                        )
                        total = torch.cuda.get_device_properties(0).total_memory / 2**20
                        result["memory_headroom_ok"] = (
                            timing["max_memory_reserved_mb"] <= config.memory_fraction * total
                        )
                except Exception as exc:
                    result["error"] = {
                        "category": error_category(exc),
                        "type": type(exc).__name__,
                        "message": str(exc)[:500],
                    }
                finally:
                    point = single = None
                    adapter.cleanup()
                    write_json_atomic(dest, result)
            attempts.append(result)
            if result["status"] != "passed" or not result.get("memory_headroom_ok", False):
                break
    recommendations = {}
    for operation in ("training", "inference"):
        eligible = [
            a
            for a in attempts
            if a["operation"] == operation
            and a["status"] == "passed"
            and a.get("memory_headroom_ok")
        ]
        recommendations[operation] = (
            max(eligible, key=lambda a: a["windows_per_second"])["batch_size"] if eligible else None
        )
    # Optional bf16 probes do not change the required FP32 route or precision policy.
    amp = []
    if config.amp_probe and torch.cuda.is_bf16_supported():
        for operation in ("inference", "training"):
            amp_path = directory / f"bf16-{operation}.json"
            if amp_path.exists():
                record = json.loads(amp_path.read_text())
                if record.get("identity") != identity:
                    raise ValueError("AMP evidence identity mismatch")
                amp.append(record)
                continue
            adapter = factory()
            reference = value = None
            record = {
                "identity": identity,
                "operation": operation,
                "dtype": "bfloat16",
                "grad_scaler": False,
            }
            try:
                adapter.load_model()
                reference = adapter.predict(batch(validating[:1]))
                if operation == "training":
                    adapter.configure_finetuning()

                def probe(adapter=adapter, operation=operation):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return (
                            adapter.train_step(batch(selected[:1]))
                            if operation == "training"
                            else adapter.predict(batch(validating[:1]))
                        )

                value, timing = measure(probe, device)
                record.update(status="passed", timing=timing)
                if operation == "inference":
                    record["fp32_max_absolute_error"] = float((value - reference).abs().max())
            except Exception as exc:
                record.update(status="failed", error=type(exc).__name__)
            finally:
                reference = value = None
                adapter.cleanup()
                write_json_atomic(amp_path, record)
            amp.append(record)
    return {
        "identity": identity,
        "status": "completed",
        "attempts": attempts,
        "fp32_batch1_supported": all(
            any(
                a["operation"] == op and a["batch_size"] == 1 and a["status"] == "passed"
                for a in attempts
            )
            for op in ("inference", "training")
        ),
        "recommended_batches_not_applied": recommendations,
        "amp": amp or "not_run",
        "protocol_frozen": False,
        "test_evaluation": False,
    }


def error_category(exc):
    message = str(exc).lower()
    if "out of memory" in message:
        return "out_of_memory"
    if "nonfinite" in message:
        return "nonfinite_loss_gradient_prediction"
    if "checkpoint" in message or "resume" in message:
        return "checkpoint_resume"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "installation_dependency"
    return "runtime_api_or_data"


def run_condition(config, row, prepared, root, commit, output, device):
    import torch

    directory = output / row["id"]
    identity = {
        "commit": commit,
        "device": device,
        "condition": row,
        "config": config.model_dump(mode="json"),
        "data_sha256": prepared[row["dataset"]]["qc"]["sha256"],
    }
    with condition_lock(directory):
        if check_completed(directory / "result.json", identity):
            return "skipped_completed"
        if device == "cuda" and not torch.cuda.is_available():
            write_json_atomic(
                directory / "pending.json",
                {"identity": identity, "status": "pending_gpu"},
                overwrite=(directory / "pending.json").exists(),
            )
            return "pending_gpu"
        adapter = None
        try:
            settings, selected, validating, batch, scale, provenance = prepare_condition(
                config, row, prepared[row["dataset"]], root, commit
            )

            def factory():
                return create_adapter(
                    settings, root=root, device=device, cache_dir=root / ".cache" / row["family"]
                )

            if row["kind"] == "stability":
                adapter = factory()
                adapter.load_model()
                result = stability(
                    adapter,
                    config,
                    selected,
                    validating,
                    batch,
                    scale,
                    directory,
                    identity,
                    provenance,
                    device,
                )
            else:
                result = feasibility(
                    factory, config, selected, validating, batch, directory, identity, device
                )
                result["data"] = provenance
            result["risk"] = token_risk(len(settings.channel_names), settings.horizon)
            result["environment"] = {
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "vram_mb": torch.cuda.get_device_properties(0).total_memory / 2**20
                if device == "cuda"
                else None,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "driver": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                ).strip()
                if device == "cuda"
                else None,
            }
            write_json_atomic(directory / "result.json", result)
            return result["status"]
        except Exception as exc:
            write_json_atomic(
                directory / f"failure-{uuid.uuid4().hex}.json",
                {
                    "identity": identity,
                    "status": "failed",
                    "category": error_category(exc),
                    "type": type(exc).__name__,
                    "summary": str(exc)[:500],
                },
            )
            raise
        finally:
            if adapter:
                adapter.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/pilot/a100_validation.yaml"))
    p.add_argument(
        "--prepared", type=Path, default=Path("results/manifests/pilot/prepared_data.json")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--condition-id", help="Execute exactly one row from the printed plan")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    config = PilotConfig.model_validate(yaml.safe_load(args.config.read_text()))
    prepared = json.loads(args.prepared.read_text())
    root = Path.cwd()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != args.expected_commit:
        raise ValueError("execution commit mismatch")
    plan = execution_plan(config, prepared, commit)
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / "plan.json"
    if dest.exists() and json.loads(dest.read_text()) != plan:
        raise ValueError("plan identity changed; use a separate run directory")
    if not dest.exists():
        write_json_atomic(dest, plan)
    print("planned groups:", plan["planned_groups"], "blocked:", plan["blocked_groups"], flush=True)
    if not args.condition_id:
        return
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("pilot execution requires clean tracked source")
    row = next(r for r in plan["conditions"] if r["id"] == args.condition_id)
    if row["status"] != "planned":
        raise ValueError("dataset not ready; no automatic fallback")
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(run_condition(config, row, prepared, root, commit, args.output, args.device))
    summaries = []
    for file in args.output.glob("*/result.json"):
        r = json.loads(file.read_text())
        condition = r["identity"]["condition"]
        summaries.append(
            {
                "id": condition["id"],
                "dataset": condition["dataset"],
                "model": condition["family"],
                "kind": condition["kind"],
                "status": r["status"],
                "best_metric": (r.get("best_validation") or {})
                .get("macro", {})
                .get("normalized_mae"),
            }
        )
    if summaries:
        write_csv_atomic(
            args.output / "summary.csv", summaries, fieldnames=list(summaries[0]), overwrite=True
        )


if __name__ == "__main__":
    main()
