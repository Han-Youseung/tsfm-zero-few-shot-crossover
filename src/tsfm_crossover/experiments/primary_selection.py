"""Evidence-gated primary selection. Produces plans only, never evaluates test."""

import argparse
import hashlib
import json
from pathlib import Path

from tsfm_crossover.tracking.atomic import write_json_atomic

PRIMARY = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Solar", "Weather", "Tetouan")
HORIZONS = (96, 192, 336, 720)


def select(prepared, reviews):
    evidence = {}
    for review in reviews:
        if review["test_evaluation"] or review["protocol_frozen"]:
            raise ValueError("invalid pilot scope")
        for result in review["conditions"]:
            row = result["condition"]
            if row["kind"] != "feasibility":
                continue
            key = (row["dataset"], row["family"], row["horizon"])
            if key in evidence:
                raise ValueError("ambiguous repeated condition evidence")
            evidence[key] = result
    selected = {}
    for name in PRIMARY:
        entry = prepared[name]
        if entry["status"] != "ready_with_warnings":
            raise ValueError("primary data not ready")
        for family in ("ttm", "moirai1"):
            for horizon in HORIZONS:
                result = evidence.get((name, family, horizon))
                if not result or result["fp32_batch1_supported"] is not True:
                    raise ValueError(f"missing common feasibility: {name}/{family}/{horizon}")
        selected[name] = entry
    return selected


def build(root):
    paths = [
        root / "results/manifests/pilot" / n
        for n in ("prepared_primary.json", "a100_review.json", "new_data_review.json")
    ]
    prepared, *reviews = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if [r["execution_commit"] for r in reviews] != [
        "486fb7c5151773a372b9371d433b9c05d578b1e5",
        "28e8151931560143aa0955f267f9d8e787549b89",
    ] or any(r["status"] != "validated_returned_evidence" for r in reviews):
        raise ValueError("unexpected source evidence")
    historical = root / "results/manifests/pilot/prepared_data.json"
    original = json.loads(historical.read_text(encoding="utf-8"))
    for name in PRIMARY[:5]:
        if prepared[name]["qc"]["sha256"] != original[name]["qc"]["sha256"]:
            raise ValueError("historical fingerprint mismatch")
    for result in reviews[1]["conditions"]:
        name = result["condition"]["dataset"]
        if (
            name in PRIMARY
            and result["eligibility"]["fingerprint"] != prepared[name]["qc"]["sha256"]
        ):
            raise ValueError("new data fingerprint mismatch")
    selected = select(prepared, reviews)
    return selected, {
        "status": "primary_scope_selected_budget_pending",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "primary_datasets": list(PRIMARY),
        "verified_common_feasibility_conditions": 64,
        "traffic": {
            "primary": False,
            "reason": "MOIRAI FP32 H720 batch-1 training OOM on A100 40GB",
            "applies_to_both_models_and_all_horizons": True,
            "retry_80gb": {
                "status": "not_run_gpu_not_allocated",
                "evidence_type": "user_report_not_imported_gpu_result",
                "reported_gpu": "NVIDIA A100-SXM4-40GB",
                "reported_vram_gib": 39.4935,
                "required_min_vram_gib": 75,
                "experiment_started": False,
                "archive_created": False,
            },
        },
        "dry_run_design": {
            "sampling_rates": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            "seeds": [1729, 2718, 31415],
            "horizons": list(HORIZONS),
            "model_dataset_horizon_units": 64,
            "zero_shot_seed_evaluations": 192,
            "few_shot_training_runs": 1536,
            "total_conditions": 1728,
            "execution_status": "not_run",
            "note": "Planning counts only; runtime and final optimizer budget not established",
        },
        "test_evaluation": False,
        "protocol_frozen": False,
        "main_experiment_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/manifests/pilot"))
    args = parser.parse_args()
    selected, decision = build(Path.cwd())
    for name, value in (("prepared_primary8.json", selected), ("primary8_decision.json", decision)):
        path = args.output / name
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError("refusing to replace prior decision")
        else:
            write_json_atomic(path, value)
    print("Primary 8 selected; 64 feasibility conditions; main experiment remains blocked")


if __name__ == "__main__":
    main()
