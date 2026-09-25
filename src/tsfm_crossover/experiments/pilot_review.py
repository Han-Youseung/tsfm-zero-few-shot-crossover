"""Read-only, bounded import of phase-6 evidence; never enables test execution."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.sampling import build_sampling_manifest
from tsfm_crossover.data.splits import chronological_split
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.experiments.pilot import validation_subset
from tsfm_crossover.experiments.pilot_config import PilotConfig, execution_plan
from tsfm_crossover.tracking.atomic import write_json_atomic

EXECUTION_COMMIT = "486fb7c5151773a372b9371d433b9c05d578b1e5"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_archive(path):
    """No extraction, pickle, checkpoint loading, or execution of archive contents."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) <= 2000, "duplicate/excessive ZIP members")
        require(sum(i.file_size for i in archive.infolist()) <= 80_000_000, "ZIP too large")
        documents = {}
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            require(
                not name.is_absolute()
                and ".." not in name.parts
                and "\\" not in info.orig_filename
                and ":" not in info.filename
                and name.suffix in {".json", ".csv"}
                and info.file_size <= 2_000_000,
                "unsafe ZIP member",
            )
            payload = archive.read(info)  # Includes ZIP CRC verification.
            if name.suffix == ".json":
                documents[info.filename] = json.loads(payload)
    return documents


def validate_metadata(meta, row, config, entry, cpu):
    model = meta["model"]
    selected = (
        cpu["selected_model_revisions"][
            0 if row["horizon"] == 96 else 2 if row["horizon"] == 720 else 1
        ]
        if row["family"] == "ttm"
        else cpu
    )
    require(model["revision"] == selected["revision"], "model revision mismatch")
    require(model["repository"] == cpu["repository"], "model repository mismatch")
    require(model["code_commit"] == cpu["code_commit"], "official code mismatch")
    require(
        meta["official_source"]["code_commit"] == cpu["code_commit"], "installed source mismatch"
    )
    require(
        meta["official_source"]["version"] == ("0.3.9" if row["family"] == "ttm" else "2.0.0"),
        "package version mismatch",
    )
    require(meta["device"].startswith("cuda") and meta["torch_cuda"], "CPU is not GPU evidence")
    cfg = meta["config"]
    require(
        cfg["family"] == row["family"]
        and cfg["condition_id"] == row["id"]
        and cfg["horizon"] == row["horizon"]
        and cfg["context"] == config.context
        and cfg["channel_names"] == entry["qc"]["channel_names"]
        and cfg["dataset_fingerprint"] == entry["qc"]["sha256"]
        and cfg["num_samples"] == (1 if row["family"] == "ttm" else 100)
        and cfg["dtype"] == "float32"
        and cfg["external_scaler"] is False,
        "adapter configuration mismatch",
    )
    require(cfg["prediction_seed"] == cfg["training_seed"] == config.seed, "seed mismatch")
    require(
        cfg["learning_rate"] == row.get("learning_rate", config.learning_rates[row["family"]][0]),
        "learning rate mismatch",
    )
    require(
        meta["total_parameters"] == meta["trainable_parameters"] > 0,
        "full parameter coverage mismatch",
    )


def validate_record(record, row, config, entry, cpu):
    identity = {
        "commit": EXECUTION_COMMIT,
        "condition": row,
        "config": config.model_dump(mode="json"),
        "data_sha256": entry["qc"]["sha256"],
        "device": "cuda",
    }
    require(record["identity"] == identity, "condition/commit/config/fingerprint mismatch")
    require(record["status"] == "completed", "incomplete condition")
    require(record["test_evaluation"] is False, "test evaluation prohibited")
    require(record["protocol_frozen"] is False, "pilot cannot freeze protocol")
    data = record["data"]
    require(data["test_targets_read"] is False, "test targets prohibited")
    split = chronological_split(entry["qc"]["rows"])
    require(stable_hash(data["split"]) == stable_hash(split.as_dict()), "split mismatch")
    fingerprint = entry["qc"]["sha256"]
    require(data["fingerprint"] == fingerprint, "data fingerprint mismatch")
    candidates = generate_train_windows(
        entry["variant"], fingerprint, split.train, 512, row["horizon"]
    )
    expected = build_sampling_manifest(
        candidates,
        dataset_sha256=fingerprint,
        split_config_hash=split.split_config_hash,
        seed=config.seed,
        code_commit_sha=EXECUTION_COMMIT,
        rates=[config.sampling_rate],
        generated_at=data["sampling_manifest"]["generated_at"],
    )
    require(
        stable_hash(expected.as_dict()) == stable_hash(data["sampling_manifest"]),
        "sampling mismatch",
    )
    windows = validation_subset(
        generate_rolling_windows(
            entry["variant"], fingerprint, "validation", split.validation, 512, row["horizon"]
        ),
        config.validation_windows,
    )
    require(
        [w.as_dict() for w in windows] == data["validation_windows"],
        "validation/test window mismatch",
    )
    require(
        data["validation_window_hash"] == stable_hash([w.window_id for w in windows]),
        "window hash mismatch",
    )
    require(
        record["environment"]["gpu"] and record["environment"]["vram_mb"] > 0,
        "GPU environment missing",
    )
    if row["kind"] == "feasibility":
        seen = set()
        for attempt in record["attempts"]:
            key = (attempt["operation"], attempt["batch_size"])
            require(key not in seen, "duplicate batch attempt")
            seen.add(key)
            require(attempt["identity"] == identity, "attempt identity mismatch")
            require(attempt["status"] in {"passed", "failed", "not_run"}, "invalid attempt status")
            if attempt["status"] == "passed":
                validate_metadata(attempt["metadata"], row, config, entry, cpu)
                require(
                    attempt["metadata"]["actual_optimizer_steps"] == (key[0] == "training"),
                    "optimizer step mismatch",
                )
                require(
                    all(math.isfinite(v) and v >= 0 for v in attempt["timing"].values()),
                    "invalid timing",
                )
        supported = all(
            any(
                a["operation"] == op and a["batch_size"] == 1 and a["status"] == "passed"
                for a in record["attempts"]
            )
            for op in ("training", "inference")
        )
        require(record["fp32_batch1_supported"] is supported, "batch-1 support mismatch")
    else:
        validate_metadata(record["metadata"], row, config, entry, cpu)
        loop = record["loop"]
        require(loop["identity"] == identity, "loop identity mismatch")
        require(
            0
            < record["stopping_step"]
            == loop["step"]
            == record["metadata"]["actual_optimizer_steps"]
            <= config.max_steps,
            "step mismatch",
        )
        history = loop["history"]
        best = min(history, key=lambda x: x["metrics"]["macro"][config.selection_metric])
        require(
            best["step"] == record["best_validation_step"]
            and best["metrics"] == record["best_validation"],
            "best validation mismatch",
        )
        require(
            all(math.isfinite(x["official_loss"]) for x in loop["loss_history"]), "nonfinite loss"
        )
    return identity


def compact(record):
    row = record["identity"]["condition"]
    result = {
        "condition": row,
        "environment": record["environment"],
        "amp": record["amp"],
        "sampling_manifest_hash": record["data"]["sampling_manifest"]["manifest_hash"],
        "validation_window_hash": record["data"]["validation_window_hash"],
    }
    if row["kind"] == "feasibility":
        result.update(
            fp32_batch1_supported=record["fp32_batch1_supported"],
            recommended_batches_not_applied=record["recommended_batches_not_applied"],
            attempts=[
                {k: v for k, v in a.items() if k not in {"identity", "metadata"}}
                for a in record["attempts"]
            ],
        )
    else:
        result.update(
            stopping_step=record["stopping_step"],
            best_validation_step=record["best_validation_step"],
            zero_shot=record["loop"]["zero_shot"]["macro"],
            best_validation=record["best_validation"]["macro"],
            relative_improvement=record["relative_improvement"],
            history=[
                {"step": h["step"], "metrics": h["metrics"]["macro"]}
                for h in record["loop"]["history"]
            ],
            train_seconds=record["loop"]["train_seconds"],
            inference_seconds=record["loop"]["inference_seconds"],
            peak_allocated_mb=record["loop"]["peak_allocated_mb"],
            convergence=record["convergence"],
            actual_unique_train_windows=record["actual_unique_train_windows"],
        )
    return result


def review(archives, root):
    prepared = json.loads((root / "results/manifests/pilot/prepared_data.json").read_text())
    config = PilotConfig.model_validate(
        yaml.safe_load((root / "configs/pilot/a100_validation.yaml").read_text())
    )
    plan = execution_plan(config, prepared, EXECUTION_COMMIT)
    expected = {r["id"]: r for r in plan["conditions"] if r["status"] == "planned"}
    found, sources, environments = {}, [], {}
    for path in archives:
        documents = read_archive(path)
        require(documents["plan.json"] == plan, "exported plan mismatch")
        sources.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for name, record in documents.items():
            if not name.endswith("/result.json"):
                continue
            key = name.split("/")[0]
            require(key in expected and key not in found, "unexpected/duplicate condition")
            row = expected[key]
            cpu = json.loads(
                (root / f"results/manifests/models/{row['family']}_compatibility.json").read_text()
            )
            validate_record(record, row, config, prepared[row["dataset"]], cpu)
            for attempt in record.get("attempts", []):
                require(
                    documents[f"{key}/{attempt['operation']}-b{attempt['batch_size']}.json"]
                    == attempt,
                    "attempt file mismatch",
                )
            metadata = record.get("metadata") or next(
                a["metadata"] for a in record["attempts"] if a["status"] == "passed"
            )
            environments[row["family"]] = {
                k: metadata[k]
                for k in ("python", "torch", "torch_cuda", "official_source", "packages")
            }
            found[key] = {
                "source_archive": path.name,
                "source_member": name,
                "record_hash": stable_hash(record),
                **compact(record),
            }
    require(set(found) == set(expected), "required pilot conditions missing")
    return {
        "execution_commit": EXECUTION_COMMIT,
        "status": "validated_returned_evidence",
        "archives": sources,
        "environments": environments,
        "conditions": [found[k] for k in sorted(found)],
        "validated_conditions": len(found),
        "remaining_blocked_groups": plan["blocked_groups"],
        "test_evaluation": False,
        "protocol_frozen": False,
        "main_experiment_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=2, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/a100_review.json")
    )
    args = parser.parse_args()
    result = review(args.archives, Path.cwd())
    if args.output.exists():
        require(json.loads(args.output.read_text()) == result, "refusing to replace prior evidence")
    else:
        write_json_atomic(args.output, result)
    print(f"Validated {result['validated_conditions']} conditions; protocol remains unfrozen")


if __name__ == "__main__":
    main()
