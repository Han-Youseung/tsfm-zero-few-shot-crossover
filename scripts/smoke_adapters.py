"""Small train/validation adapter integration; no experiments or test evaluation."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.sampling import build_sampling_manifest
from tsfm_crossover.data.splits import chronological_split
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.models.adapter_config import AdapterConfig
from tsfm_crossover.models.adapters import create_adapter
from tsfm_crossover.models.contract import parameter_hash
from tsfm_crossover.models.gpu_gate import DATA_SHA
from tsfm_crossover.models.gpu_smoke import equal_tree, measured, rng_state
from tsfm_crossover.models.window_batch import make_window_batch, select_train_windows
from tsfm_crossover.tracking.atomic import write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--horizon", type=int, choices=(96, 192, 336, 720))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="CPU development only")
    args = parser.parse_args()
    root = Path.cwd()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if commit != args.expected_commit or (
        dirty and not (args.allow_dirty and args.device == "cpu")
    ):
        raise ValueError("exact clean execution commit required (CPU development override only)")
    settings = yaml.safe_load(args.config.read_text())
    if args.horizon:
        settings["horizon"] = args.horizon
    batch_size = settings.pop("batch_size")
    rate = settings.pop("sampling_rate")
    sampling_seed = settings.pop("sampling_seed")
    if not 1 <= batch_size <= 4 or settings["max_optimizer_steps"] not in (1, 2):
        raise ValueError("smoke limited to batch 1..4 and 1..2 steps")
    split = chronological_split(17420)
    candidates = generate_train_windows(
        "ETTh1__official_raw", DATA_SHA, split.train, settings["context"], settings["horizon"]
    )
    manifest = build_sampling_manifest(
        candidates,
        dataset_sha256=DATA_SHA,
        split_config_hash=split.split_config_hash,
        seed=sampling_seed,
        code_commit_sha=commit,
        rates=[rate],
    )
    selected = select_train_windows(candidates, manifest, rate)
    config = AdapterConfig(
        **settings,
        condition_id=stable_hash([settings, manifest.manifest_hash]),
        dataset_fingerprint=DATA_SHA,
        sampling_manifest_hash=manifest.manifest_hash,
    )
    source_hash = stable_hash(
        {
            str(p.relative_to(root).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src").rglob("*.py"))
        }
        | {"scripts/smoke_adapters.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    )
    run_identity = {
        "execution_commit": commit,
        "source_sha256": source_hash,
        "config": config.model_dump(mode="json"),
        "device": args.device,
        "batch_size": batch_size,
        "sampling_rate": rate,
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("identity") != run_identity:
            raise ValueError("existing output belongs to a different adapter execution")
        if previous.get("status") == "passed":
            print("resume: completed adapter smoke", args.output.name)
            return 0
        raise ValueError("preserve failed attempt; use a new output filename")
    result = {
        "kind": "adapter_integration",
        "identity": run_identity,
        "dirty_worktree": dirty,
        "status": "pending_gpu",
        "protocol_frozen": False,
        "amp": "not_run",
        "test_split_used": False,
        "checks": {},
    }
    adapter = None
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            result["reason"] = "CUDA unavailable; no GPU execution attempted"
            return 2
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hashlib.sha256(args.data.read_bytes()).hexdigest() != DATA_SHA:
            raise ValueError("official raw ETTh1 fingerprint mismatch")
        with args.data.open(encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            if next(rows) != ["date", *config.channel_names]:
                raise ValueError("ETTh1 channel order mismatch")
            values = [
                [float(x) for x in row[1:]] for row in itertools.islice(rows, split.validation.end)
            ]
        validation = generate_rolling_windows(
            "ETTh1__official_raw",
            DATA_SHA,
            "validation",
            split.validation,
            config.context,
            config.horizon,
        )[:batch_size]

        def batch(windows):
            return make_window_batch(
                values,
                windows,
                splits=split,
                channel_names=config.channel_names,
                fingerprint=DATA_SHA,
                sampling_manifest_hash=manifest.manifest_hash,
            )

        training = batch(selected[:batch_size])
        validating = batch(validation)
        result["data"] = {
            "fingerprint": DATA_SHA,
            "sampling_manifest": manifest.as_dict(),
            "train_windows": [w.as_dict() for w in selected[:batch_size]],
            "validation_windows": [w.as_dict() for w in validation],
            "selected_train_windows": len(selected),
            "executed_train_windows": len(selected[:batch_size]),
            "drop_last": False,
        }
        kwargs = dict(
            root=root,
            device=args.device,
            cache_dir=root / ".cache" / config.family,
            local_files_only=args.local_files_only,
        )

        def measure(fn):
            if args.device == "cuda":
                return measured(fn)
            start = time.perf_counter()
            value = fn()
            return value, {
                "wall_seconds": time.perf_counter() - start,
                "max_memory_allocated_mb": None,
                "max_memory_reserved_mb": None,
            }

        adapter = create_adapter(config, **kwargs)
        adapter.load_model()
        original = parameter_hash(adapter.parameter_state())
        initial_rng = rng_state()
        prediction, result["inference"] = measure(lambda: adapter.zero_shot_predict(validating))
        torch.testing.assert_close(
            prediction, adapter.zero_shot_predict(validating), rtol=1e-4, atol=1e-5
        )
        result["checks"]["prediction_rng_preserved"] = equal_tree(initial_rng, rng_state())
        result["checks"]["zero_shot_no_update"] = original == parameter_hash(
            adapter.parameter_state()
        )
        result["checks"]["shape_finite_repeat"] = True
        result["point_shape"] = list(prediction.shape)
        adapter.configure_finetuning()
        result["checks"]["full_optimizer_coverage"] = True
        # MOIRAI adds the official wrapper's state-key prefix when configured.
        # Compare within that representation so wrapper creation is not an update.
        before_training = parameter_hash(adapter.parameter_state())
        losses = []
        for _ in range(config.max_optimizer_steps):
            loss, timing = measure(lambda: adapter.train_step(training))
            losses.append(loss)
            result.setdefault("training_steps", []).append(timing)
        result["losses"] = losses
        result["checks"]["finite_loss_gradients"] = True
        result["checks"]["parameters_updated"] = before_training != parameter_hash(
            adapter.parameter_state()
        )
        result["validation_objective"] = adapter.validation_step(validating)
        expected = adapter.predict(validating).cpu()
        trained = parameter_hash(adapter.parameter_state())
        result["metadata"] = adapter.execution_metadata()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapter.pt"
            adapter.save_training_state(checkpoint)
            saved_rng = rng_state()
            optimizer_state = adapter.optimizer.state_dict()
            adapter.cleanup()
            adapter = create_adapter(config, **kwargs)
            adapter.load_model()
            result["checks"]["independent_pretrained"] = (
                parameter_hash(adapter.parameter_state()) == original
            )
            adapter.load_training_state(checkpoint)
            result["checks"]["restore_rng"] = equal_tree(saved_rng, rng_state())
            result["checks"]["restore_optimizer"] = equal_tree(
                optimizer_state, adapter.optimizer.state_dict()
            )
            result["checks"]["restore_hash"] = parameter_hash(adapter.parameter_state()) == trained
            result["checks"]["restore_step"] = adapter.global_step == config.max_optimizer_steps
            restored = adapter.predict(validating).cpu()
            torch.testing.assert_close(expected, restored, rtol=1e-4, atol=1e-5)
            result["restore_max_absolute_error"] = float((expected - restored).abs().max())
            result["checks"]["restore_prediction"] = True
        if not all(result["checks"].values()):
            raise ValueError("adapter integration check failed")
        result["status"] = "passed"
        result["adapter_gpu_status"] = "passed" if args.device == "cuda" else "pending_gpu"
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        raise
    finally:
        if adapter:
            adapter.cleanup()
        write_json_atomic(args.output, result)
        print(result["status"], args.output.name)


if __name__ == "__main__":
    raise SystemExit(main())
