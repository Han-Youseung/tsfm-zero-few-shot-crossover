"""One authorized main condition per process; train/selection/test transactions."""

import argparse
import json
import math
import os
import subprocess
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.missing import POLICY, causal_context, eligible_windows
from tsfm_crossover.data.pilot_data import digest, load_final_values
from tsfm_crossover.data.windows import generate_rolling_windows
from tsfm_crossover.evaluation.metrics import StreamingMetrics
from tsfm_crossover.experiments.main_plan import (
    grid,
    identity_for,
    load_plan,
    resource_status,
    verify_runtime,
)
from tsfm_crossover.experiments.pilot import error_category, measure, prepare_condition, stability
from tsfm_crossover.experiments.pilot_state import check_completed, condition_lock
from tsfm_crossover.models.adapters import create_adapter
from tsfm_crossover.models.contract import parameter_hash
from tsfm_crossover.models.window_batch import make_window_batch
from tsfm_crossover.tracking.atomic import write_json_atomic


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def immutable(path, record):
    """Idempotent exact identity check, never replace existing evidence."""
    if path.exists():
        if read(path) != json.loads(json.dumps(record)):
            raise ValueError(f"immutable record differs: {path.name}")
    else:
        write_json_atomic(path, record)


def loop_config(plan, row):
    return SimpleNamespace(
        seed=row["seed"],
        sampling_rate=row["rate"],
        num_samples=plan.models[row["family"]].samples,
        learning_rates={k: [v.learning_rate] for k, v in plan.models.items()},
        max_steps=plan.max_optimizer_steps,
        eval_every=plan.eval_every_steps,
        patience_evals=plan.patience_evals,
        validation_windows=plan.validation_windows,
        training_batch=1,
        checkpoint_every=plan.checkpoint_every_steps,
        selection_metric="normalized_mae",
    )


def final_windows(root, entry, settings, selection):
    if selection.get("status") != "selection_locked" or not selection.get("parameter_hash"):
        raise ValueError("checkpoint must be locked before opening test values")
    values, channels, splits = load_final_values(root, entry)
    context = causal_context(values) if entry.get("missing_policy") == POLICY else values
    nominal = generate_rolling_windows(
        entry["variant"], entry["qc"]["sha256"], "test", splits.test, 512, settings.horizon
    )
    windows = eligible_windows(nominal, values, context, training=False)
    if not windows:
        raise ValueError("no eligible test windows")

    def batch(items):
        return make_window_batch(
            values,
            items,
            splits=splits,
            channel_names=channels,
            fingerprint=settings.dataset_fingerprint,
            sampling_manifest_hash=settings.sampling_manifest_hash,
            context_values=context,
            final_test=True,
        )

    return (
        windows,
        batch,
        {"nominal": len(nominal), "target_counts": target_counts(values, windows)},
    )


def target_counts(values, windows):
    """Count repeated observed targets in O(rows*channels + windows), not O(N*H*C)."""
    events = {}
    for window in windows:
        events[window.target_start] = events.get(window.target_start, 0) + 1
        events[window.target_end] = events.get(window.target_end, 0) - 1
    counts = [0] * len(values[0])
    active = 0
    for t in range(min(events), max(events)):
        active += events.get(t, 0)
        for c, value in enumerate(values[t]):
            if math.isfinite(value):
                counts[c] += active
    return counts


def evaluate_final(
    adapter, windows, batch, scale, directory, identity, selection, *, every=100, device="cuda"
):
    """Resume paired rolling-origin sums, not arrays; never double-count saved origins."""
    model_hash = parameter_hash(adapter.parameter_state())
    if selection["identity"] != identity or selection["parameter_hash"] != model_hash:
        raise ValueError("selected checkpoint/identity changed before test")
    signature = {
        "identity": identity,
        "selection_hash": stable_hash(selection),
        "window_hash": stable_hash([w.window_id for w in windows]),
    }
    progress = directory / "test-progress.json"
    metric = StreamingMetrics(scale)
    state = {
        "signature": signature,
        "status": "running",
        "next_window": 0,
        "inference_seconds": 0.0,
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
    }
    if progress.exists():
        state = read(progress)
        if state["signature"] != signature or not 0 <= state["next_window"] <= len(windows):
            raise ValueError("test resume identity/window mismatch")
        metric.restore(state["metrics"])
    steps = adapter.global_step
    backward = adapter.backward_calls
    for i in range(state["next_window"], len(windows)):
        item = batch([windows[i]])
        # Test labels never enter adapter.prepare_batch or any loss function.
        point, timing = measure(
            lambda item=item: adapter.predict_test(replace(item, future=None)), device
        )
        metric.update_array(point.detach().cpu().double().numpy()[0], item.future[0])
        state["next_window"] = i + 1
        state["inference_seconds"] += timing["seconds"]
        for key in ("allocated", "reserved"):
            state[f"peak_{key}_mb"] = max(
                state[f"peak_{key}_mb"], timing[f"max_memory_{key}_mb"] or 0
            )
        if (i + 1) % every == 0 or i + 1 == len(windows):
            state["metrics"] = metric.state()
            write_json_atomic(progress, state, overwrite=progress.exists())
            print(f"test windows {i + 1}/{len(windows)}", flush=True)
    if (
        model_hash != parameter_hash(adapter.parameter_state())
        or steps != adapter.global_step
        or backward != adapter.backward_calls
    ):
        raise RuntimeError("test changed model/optimizer state")
    return {
        "metrics": metric.compute(),
        "window_hash": signature["window_hash"],
        "number_of_test_windows": len(windows),
        "stride": 1,
        "no_update_verified": True,
        "parameter_hash": model_hash,
        "timing": {
            k: state[k] for k in ("inference_seconds", "peak_allocated_mb", "peak_reserved_mb")
        },
        "warmup": "none; first final-test forward included",
    }


def compact_training(training):
    if training is None:
        return {
            "actual_optimizer_steps": 0,
            "train_seconds": 0.0,
            "equivalent_epochs": 0.0,
            "average_window_exposures": 0.0,
            "actual_unique_train_windows": 0,
            "best_validation_step": None,
            "stopping_step": 0,
            "validation_history": [],
        }
    state = training["loop"]
    return {
        **{
            k: training[k]
            for k in (
                "equivalent_epochs",
                "average_window_exposures",
                "actual_unique_train_windows",
                "best_validation_step",
                "stopping_step",
            )
        },
        "actual_optimizer_steps": training["stopping_step"],
        **{
            k: state[k]
            for k in (
                "train_seconds",
                "inference_seconds",
                "peak_allocated_mb",
                "peak_reserved_mb",
                "visited_train_indices",
            )
        },
        "validation_history": [
            {"step": h["step"], "macro": h["metrics"]["macro"]} for h in state["history"]
        ],
        "zero_shot_validation": state["zero_shot"]["macro"],
        "best_validation": training["best_validation"]["macro"],
        "stop_reason": "early_stopping" if state["bad_evals"] == 3 else "maximum_steps",
        "scheduler": "none",
        "objective": "official model-specific loss",
        "convergence": "not_established; bounded-compute main study",
    }


def run(plan, row, prepared, root, commit, output):
    import torch

    identity = identity_for(plan, prepared, root, row, commit)
    directory = output / row["id"]
    with condition_lock(directory):
        if check_completed(directory / "result.json", identity):
            return "skipped_completed"
        cuda = torch.cuda.is_available()
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0
        pending = resource_status(row, cuda, vram)
        if pending != "ready":
            write_json_atomic(
                directory / f"pending-{uuid.uuid4().hex}.json",
                {"identity": identity, "status": pending, "actual_vram_gib": vram},
            )
            return pending
        adapter = None
        try:
            start_path = directory / "started.json"
            if not start_path.exists():
                write_json_atomic(
                    start_path, {"identity": identity, "started_at": datetime.now(UTC).isoformat()}
                )
            started = read(start_path)
            if started["identity"] != identity:
                raise ValueError("start record identity mismatch")
            cfg = loop_config(plan, row)
            settings, selected, validating, batch, scale, provenance = prepare_condition(
                cfg,
                row,
                prepared[row["dataset"]],
                root,
                commit,
                sampling_rates=(0.0, *plan.sampling_rates),
                protocol_frozen=True,
            )
            manifest = provenance.pop("sampling_manifest")
            manifest_path = output / "sampling" / f"{manifest['manifest_hash']}.json"
            immutable(manifest_path, manifest)
            provenance.update(
                sampling_manifest_hash=manifest["manifest_hash"],
                sampling_manifest_file=manifest_path.relative_to(output).as_posix(),
                sampling_seed=row["seed"],
            )
            # All stages refer to the same initial train/validation-only provenance.
            immutable(directory / "provenance.json", provenance)

            def factory():
                a = create_adapter(
                    settings, root=root, device="cuda", cache_dir=root / ".cache" / row["family"]
                )
                a.load_model()
                return a

            adapter = factory()
            metadata = adapter.execution_metadata()
            runtime = {
                k: metadata[k]
                for k in (
                    "python",
                    "torch",
                    "torch_cuda",
                    "packages",
                    "official_source",
                    "gpu",
                    "device",
                    "deterministic_algorithms",
                    "deterministic_warn_only",
                    "tf32",
                    "cudnn_benchmark",
                )
            }
            runtime.update(
                vram_gib=vram,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                driver=subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                ).strip(),
            )
            verify_runtime(runtime, row["family"], root)
            # Do not silently combine a partial run across different hardware/software.
            immutable(directory / "runtime.json", runtime)
            training = None
            checkpoint = None
            if row["kind"] == "few":
                trained = directory / "training.json"
                if check_completed(trained, identity):
                    training = read(trained)
                else:
                    training = stability(
                        adapter,
                        cfg,
                        selected,
                        validating,
                        batch,
                        scale,
                        directory,
                        identity,
                        provenance,
                        "cuda",
                    )
                    write_json_atomic(trained, training)
                checkpoint = directory / training["loop"]["best_checkpoint"]
                adapter.cleanup()
                adapter = factory()
                adapter.load_training_state(checkpoint)
                if adapter.global_step != training["best_validation_step"]:
                    raise ValueError("best validation checkpoint step mismatch")
                # Restoring optimizer state is verified by adapter. Release it for test.
                adapter.optimizer = None
                torch.cuda.empty_cache()
            selection = {
                "identity": identity,
                "status": "selection_locked",
                "selected_by": "validation_only" if training else "pretrained_zero_shot",
                "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
                "best_validation_step": training["best_validation_step"] if training else 0,
                "parameter_hash": parameter_hash(adapter.parameter_state()),
            }
            immutable(directory / "selection.json", selection)
            # Release the large train/validation closure before opening final test.
            del batch, selected, validating
            windows, test_batch, inventory = final_windows(
                root, prepared[row["dataset"]], settings, selection
            )
            evaluated = evaluate_final(
                adapter,
                windows,
                test_batch,
                scale,
                directory,
                identity,
                selection,
                every=plan.test_checkpoint_every_windows,
            )
            if [c["count"] for c in evaluated["metrics"]["per_channel"]] != inventory[
                "target_counts"
            ]:
                raise ValueError("final metric mask inventory mismatch")
            result = {
                "status": "completed",
                "identity": identity,
                "protocol_frozen": True,
                "test_evaluation": True,
                "started_at": started["started_at"],
                "completed_at": datetime.now(UTC).isoformat(),
                "provenance": provenance,
                "runtime_hash": stable_hash(runtime),
                "runtime": runtime,
                "selection": selection,
                "training": compact_training(training),
                "test": evaluated,
                "nominal_test_windows": inventory["nominal"],
                "excluded_test_windows": inventory["nominal"] - len(windows),
                "adapter_settings": settings.model_dump(mode="json"),
                "trainable_parameters": adapter.count_parameters()[0],
                "total_parameters": adapter.count_parameters()[1],
                "precision": "float32",
                "amp": "not_run",
                "external_scaler": False,
            }
            write_json_atomic(directory / "result.json", result)
            return "completed"
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
    p.add_argument("--config", type=Path, default=Path("configs/study/main.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--condition-id", help="one planned condition; omit for CPU-only planning")
    args = p.parse_args()
    root = Path.cwd()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != args.expected_commit:
        raise ValueError("execution commit mismatch")
    plan, prepared = load_plan(root, args.config)
    rows = grid(plan)
    immutable(
        args.output / "plan.json",
        {
            "commit": commit,
            "config": plan.model_dump(mode="json"),
            "conditions": rows,
            "prepared_hash": stable_hash(prepared),
        },
    )
    print(f"planned conditions: {len(rows)}; test values not opened by planning", flush=True)
    if not args.condition_id:
        return
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("main execution requires clean tracked source")
    row = next(r for r in rows if r["id"] == args.condition_id)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    print(run(plan, row, prepared, root, commit, args.output), flush=True)
    print("process wall seconds:", time.perf_counter() - start, flush=True)


if __name__ == "__main__":
    main()
