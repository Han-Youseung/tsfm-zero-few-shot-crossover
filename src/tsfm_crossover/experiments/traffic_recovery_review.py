"""Verify the returned FP32 Traffic recovery without loading model weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.experiments.pilot import prepare_condition
from tsfm_crossover.experiments.pilot_config import PilotConfig
from tsfm_crossover.experiments.pilot_review import read_archive, require, validate_metadata
from tsfm_crossover.tracking.atomic import write_json_atomic

COMMIT = "a4386f2cdceb1e552cd73d0cd3c89fa31228fd6c"
CHECKS = {
    "cuda_input",
    "cuda_parameters",
    "finite_gradients",
    "finite_shape",
    "full_parameter_coverage",
    "restore_hash",
    "restore_memory_profile",
    "restore_optimizer",
    "restore_prediction",
    "restore_rng",
    "restore_step",
    "seed_repeat",
    "steps",
    "update",
    "validation_after_training_finite",
    "zero_shot_unchanged",
}


def validate_envelope(record, name, entry):
    identity = {
        "batch": 1,
        "commit": COMMIT,
        "context": 512,
        "dataset": name,
        "family": "moirai1",
        "fingerprint": entry["qc"]["sha256"],
        "horizon": 720,
        "profile": "fp32",
        "samples": 100,
        "seed": 1729,
        "steps": 2,
    }
    require(record["identity"] == identity, "identity mismatch")
    require(record["status"] == "passed" and record["stage"] == "complete", "incomplete probe")
    require(
        record["checks"].keys() >= CHECKS and all(record["checks"][k] is True for k in CHECKS),
        "required checks missing or failed",
    )
    require(
        not record["test_evaluation"]
        and not record["protocol_frozen"]
        and not record["main_experiment_allowed"],
        "research gate violation",
    )
    require(
        record["precision"]["parameters"] == "float32"
        and record["precision"]["autocast"] is None
        and record["precision"]["saved_tensors_on_cpu"] is False,
        "not plain FP32 evidence",
    )
    require(
        record["prediction"]["shape"] == [1, 720, entry["qc"]["channels"]]
        and record["prediction"]["device"].startswith("cuda")
        and record["prediction"]["dtype"] == "torch.float32",
        "prediction mismatch",
    )
    require(
        record["metadata"]["actual_optimizer_steps"] == 2
        and len(record["losses"]) == 2
        and all(math.isfinite(v) for v in record["losses"]),
        "training steps/loss mismatch",
    )
    require(
        record["restore_tolerance"] == {"atol": 1e-5, "rtol": 1e-4}
        and math.isfinite(record["restore_max_absolute_error"]),
        "restore evidence invalid",
    )
    for stage in ("inference", "training_1", "training_2"):
        for key in ("seconds", "max_memory_allocated_mb", "max_memory_reserved_mb"):
            value = record["timing"][stage][key]
            require(math.isfinite(value) and value > 0, "timing/memory invalid")
    env = record["environment"]
    require(env["gpu"] and env["torch_cuda"] and env["vram_gib"] > 0, "GPU evidence missing")
    require(
        record["metadata"]["torch"] == env["torch"]
        and record["metadata"]["torch_cuda"] == env["torch_cuda"],
        "environment mismatch",
    )
    return identity


def review(path, root):
    docs = read_archive(path)
    require(set(docs) == {"ETTh1-fp32.json", "Traffic-fp32.json"}, "unexpected/missing probe files")
    prepared = json.loads(
        (root / "results/manifests/pilot/prepared_primary.json").read_text(encoding="utf-8")
    )
    cpu = json.loads((root / "results/manifests/models/moirai1_compatibility.json").read_text())
    config = PilotConfig(max_steps=2, learning_rates={"ttm": [1e-4], "moirai1": [5e-6]})
    conditions = []
    for name in ("ETTh1", "Traffic"):
        record = docs[f"{name}-fp32.json"]
        identity = validate_envelope(record, name, prepared[name])
        row = {"family": "moirai1", "horizon": 720, "id": stable_hash(identity)}
        validate_metadata(record["metadata"], row, config, prepared[name], cpu)
        data = prepare_condition(config, row, prepared[name], root, COMMIT)[-1]
        data["sampling_manifest"]["generated_at"] = record["data"]["sampling_manifest"][
            "generated_at"
        ]
        require(stable_hash(data) == stable_hash(record["data"]), "data/window/sampling mismatch")
        conditions.append(
            {
                "record_hash": stable_hash(record),
                **{k: v for k, v in record.items() if k != "data"},
                "data_hash": stable_hash(record["data"]),
                "sampling_manifest_hash": data["sampling_manifest"]["manifest_hash"],
                "validation_window_hash": data["validation_window_hash"],
            }
        )
    return {
        "status": "validated_fp32_gpu_recovery",
        "execution_commit": COMMIT,
        "archive": {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "conditions": conditions,
        "traffic": "include_with_gpu_resource_constraint",
        "cpu_offload": "not_run",
        "bf16": "not_run",
        "budget_confirmation": "may_resume",
        "test_evaluation": False,
        "protocol_frozen": False,
        "main_experiment_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/traffic_recovery_review.json")
    )
    args = parser.parse_args()
    result = review(args.archive, Path.cwd())
    if args.output.exists():
        require(
            json.loads(args.output.read_text(encoding="utf-8")) == result,
            "previous evidence differs",
        )
    else:
        write_json_atomic(args.output, result)
    from tsfm_crossover.experiments.primary_selection import PRIMARY

    source = json.loads(
        Path("results/manifests/pilot/prepared_primary.json").read_text(encoding="utf-8")
    )
    primary = {name: source[name] for name in (*PRIMARY, "Traffic")}
    selection_path = args.output.parent / "prepared_primary9.json"
    if selection_path.exists():
        require(
            json.loads(selection_path.read_text(encoding="utf-8")) == primary,
            "existing primary selection differs",
        )
    else:
        write_json_atomic(selection_path, primary)
    print(
        "Two FP32 probes verified; Traffic recovered with resource constraint; protocol not frozen"
    )


if __name__ == "__main__":
    main()
