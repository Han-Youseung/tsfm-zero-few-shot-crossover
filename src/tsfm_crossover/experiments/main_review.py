"""CPU-only import of small final result ZIPs, with canonical provenance checks."""

import argparse
import math
from pathlib import Path

from tsfm_crossover.analysis.crossover import summarize
from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.pilot_data import digest
from tsfm_crossover.experiments.main_plan import identity_for, load_plan, verify_runtime
from tsfm_crossover.experiments.main_study import final_windows, loop_config
from tsfm_crossover.experiments.pilot import prepare_condition
from tsfm_crossover.experiments.pilot_review import read_archive, require
from tsfm_crossover.tracking.atomic import write_json_atomic


def validate_result(record, expected_identity, settings, provenance, test_signature, plan, root):
    require(record["status"] == "completed", "incomplete main result")
    require(record["identity"] == expected_identity, "commit/revision/config/fingerprint mismatch")
    require(record["adapter_settings"] == settings.model_dump(mode="json"), "adapter mismatch")
    require(
        stable_hash(record["provenance"]) == stable_hash(provenance), "train provenance mismatch"
    )
    require(
        record["test_evaluation"] is True and record["protocol_frozen"] is True,
        "historical pilot is not a main result",
    )
    require(
        record["precision"] == "float32" and record["external_scaler"] is False,
        "precision/scaling mismatch",
    )
    runtime = record["runtime"]
    require(stable_hash(runtime) == record["runtime_hash"], "runtime hash mismatch")
    verify_runtime(runtime, expected_identity["condition"]["family"], root)
    require(
        bool(runtime["gpu"]) and runtime["vram_gib"] > 0 and runtime["device"].startswith("cuda"),
        "CPU result is not GPU evidence",
    )
    selection, test = record["selection"], record["test"]
    require(
        selection["identity"] == expected_identity and selection["status"] == "selection_locked",
        "checkpoint selection identity mismatch",
    )
    require(
        test["parameter_hash"] == selection["parameter_hash"] and test["no_update_verified"],
        "test parameter no-update check absent",
    )
    require(
        test["stride"] == 1
        and test["number_of_test_windows"] == test_signature["count"]
        and test["window_hash"] == test_signature["hash"],
        "test windows mismatch",
    )
    per_channel = test["metrics"]["per_channel"]
    require(
        [c["count"] for c in per_channel] == test_signature["target_counts"],
        "test observation mask/count mismatch",
    )
    require(
        set(test["metrics"]["macro"]) == {"mae", "mse", "normalized_mae", "normalized_mse"},
        "required metrics missing",
    )
    scale = provenance["metric_train_scale"]
    for c, sd in zip(per_channel, scale["std"], strict=True):
        require(
            all(
                v is None or (math.isfinite(v) and v >= 0)
                for k, v in c.items()
                if k != "normalization_eligible"
            ),
            "nonfinite metric",
        )
        eligible = sd is not None and sd > scale["near_constant_threshold"]
        require(c["normalization_eligible"] == eligible, "constant channel handling mismatch")
        for raw, normalized, power in (("mae", "normalized_mae", 1), ("mse", "normalized_mse", 2)):
            expected = c[raw] / sd**power if eligible and c["count"] else None
            require(
                c[normalized] == expected
                or (
                    c[normalized] is not None
                    and expected is not None
                    and math.isclose(c[normalized], expected, rel_tol=1e-12)
                ),
                "normalized metric scale mismatch",
            )
    for key, value in test["metrics"]["macro"].items():
        values = [c[key] for c in per_channel if c[key] is not None]
        expected = math.fsum(values) / len(values) if values else None
        require(
            value == expected
            or (
                value is not None
                and expected is not None
                and math.isclose(value, expected, rel_tol=1e-12)
            ),
            "channel macro metric mismatch",
        )
    require(
        record["trainable_parameters"] == record["total_parameters"] > 0,
        "not full-parameter training",
    )
    train = record["training"]
    if expected_identity["condition"]["kind"] == "zero":
        require(
            train["actual_optimizer_steps"] == 0
            and not train["validation_history"]
            and selection["checkpoint_sha256"] is None
            and selection["selected_by"] == "pretrained_zero_shot",
            "Zero-Shot updated model",
        )
        return
    history = train["validation_history"]
    stop = train["stopping_step"]
    require(
        stop == train["actual_optimizer_steps"] <= plan.max_optimizer_steps
        and [h["step"] for h in history] == list(range(100, stop + 1, 100)),
        "step budget/cadence mismatch",
    )
    best, best_step, bad = None, None, 0
    for h in history:
        require(bad < 3, "continued after early stopping")
        score = h["macro"]["normalized_mae"]
        require(math.isfinite(score) and score >= 0, "invalid validation score")
        if best is None or score < best:
            best, best_step, bad = score, h["step"], 0
        else:
            bad += 1
    require(
        best_step is not None
        and best_step == train["best_validation_step"] == selection["best_validation_step"]
        and selection["selected_by"] == "validation_only"
        and selection["checkpoint_sha256"] is not None,
        "validation selection mismatch",
    )
    require(stop == 1000 or bad == 3, "premature stopping")
    count = provenance["selected_train_windows"]
    require(
        train["equivalent_epochs"] == train["average_window_exposures"] == stop / count,
        "exposure count mismatch",
    )
    indices = train["visited_train_indices"]
    require(
        indices == sorted(set(indices))
        and all(0 <= i < count for i in indices)
        and len(indices) == train["actual_unique_train_windows"] <= stop,
        "visited training window mismatch",
    )


def review(archives, root, commit, plan, prepared):
    results, test_cache = {}, {}
    for archive in archives:
        docs = read_archive(archive)
        exported_plan = docs["plan.json"]
        require(
            exported_plan["commit"] == commit
            and exported_plan["config"] == plan.model_dump(mode="json")
            and exported_plan["prepared_hash"] == stable_hash(prepared),
            "archive plan mismatch",
        )
        for member, record in docs.items():
            if not member.endswith("/result.json"):
                continue
            row = record["identity"]["condition"]
            require(member == row["id"] + "/result.json", "result member identity mismatch")
            if row["id"] in results:
                require(results[row["id"]] == record, "conflicting duplicate result")
                continue
            expected_identity = identity_for(plan, prepared, root, row, commit)
            settings, selected, validating, batch, scale, provenance = prepare_condition(
                loop_config(plan, row),
                row,
                prepared[row["dataset"]],
                root,
                commit,
                sampling_rates=(0.0, *plan.sampling_rates),
                protocol_frozen=True,
            )
            manifest = provenance.pop("sampling_manifest")
            provenance.update(
                sampling_manifest_hash=manifest["manifest_hash"],
                sampling_manifest_file=f"sampling/{manifest['manifest_hash']}.json",
                sampling_seed=row["seed"],
            )
            del selected, validating, batch, scale, manifest
            key = (row["dataset"], row["horizon"])
            if key not in test_cache:
                # Import never selects a checkpoint using these labels. It checks the
                # already-completed final evaluation's canonical origin/mask inventory.
                windows, batch, inventory = final_windows(
                    root, prepared[row["dataset"]], settings, record["selection"]
                )
                test_cache[key] = {
                    "count": len(windows),
                    "hash": stable_hash([w.window_id for w in windows]),
                    "target_counts": inventory["target_counts"],
                }
                del batch, windows
            validate_result(
                record, expected_identity, settings, provenance, test_cache[key], plan, root
            )
            results[row["id"]] = record
    require(bool(results), "no completed final results in archives")
    return {
        "status": "validated_partial" if len(results) < 1944 else "validated_complete",
        "execution_commit": commit,
        "validated_conditions": len(results),
        "planned_conditions": 1944,
        "archives": [{"name": p.name, "sha256": digest(p)} for p in archives],
        "conditions": [compact_result(r) for _, r in sorted(results.items())],
        "test_metrics_recomputed_from_predictions": False,
        "validation": "identities, canonical windows/masks, package pins and stopping replay",
        "analysis": summarize(list(results.values()), plan.sampling_rates, plan.seeds),
    }


def compact_result(record):
    """Keep small per-condition metrics in Git; arrays/checkpoints stay outside it."""
    data, train = record["provenance"], record["training"]
    return {
        "identity": record["identity"],
        "record_hash": stable_hash(record),
        "sampling_manifest_hash": data["sampling_manifest_hash"],
        "total_train_windows": data["total_train_windows"],
        "selected_train_windows": data["selected_train_windows"],
        "effective_sampling_rate": data["effective_sampling_rate"],
        "coverage": data["coverage"],
        "actual_optimizer_steps": train["actual_optimizer_steps"],
        "actual_unique_train_windows": train["actual_unique_train_windows"],
        "equivalent_epochs": train["equivalent_epochs"],
        "best_validation_step": train["best_validation_step"],
        "stopping_step": train["stopping_step"],
        "train_seconds": train["train_seconds"],
        "test_timing": record["test"]["timing"],
        "test_window_hash": record["test"]["window_hash"],
        "test_windows": record["test"]["number_of_test_windows"],
        "test_metrics": record["test"]["metrics"]["macro"],
        "runtime_hash": record["runtime_hash"],
        "gpu": record["runtime"]["gpu"],
        "vram_gib": record["runtime"]["vram_gib"],
        "trainable_parameters": record["trainable_parameters"],
        "total_parameters": record["total_parameters"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    plan, prepared = load_plan(root, root / "configs/study/main.yaml")
    result = review(args.archives, root, args.expected_commit, plan, prepared)
    write_json_atomic(args.output, result)
    print(result["status"], result["validated_conditions"])


if __name__ == "__main__":
    main()
