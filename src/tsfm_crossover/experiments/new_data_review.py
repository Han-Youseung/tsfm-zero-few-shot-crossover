"""Verify the returned new-data pilot, including causal eligibility regenerated locally."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.experiments.pilot import prepare_condition
from tsfm_crossover.experiments.pilot_config import PilotConfig, execution_plan
from tsfm_crossover.experiments.pilot_review import compact, read_archive, require, validate_record
from tsfm_crossover.tracking.atomic import write_json_atomic

COMMIT = "28e8151931560143aa0955f267f9d8e787549b89"
DATASETS = {"Solar", "Traffic", "Weather", "Tetouan"}


def review(archives, root):
    prepared = json.loads(
        (root / "results/manifests/pilot/prepared_primary.json").read_text(encoding="utf-8")
    )
    config = PilotConfig.model_validate(
        yaml.safe_load((root / "configs/pilot/a100_validation.yaml").read_text())
    )
    plan = execution_plan(config, prepared, COMMIT)
    expected = {
        r["id"]: r
        for r in plan["conditions"]
        if r["dataset"] in DATASETS and r["kind"] == "feasibility" and r["status"] == "planned"
    }
    found, sources, provenance = {}, [], {}
    for path in archives:
        docs = read_archive(path)
        require(docs["plan.json"] == plan, "plan/commit mismatch")
        sources.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for name, record in docs.items():
            if not name.endswith("/result.json"):
                continue
            key = name.split("/")[0]
            require(key in expected and key not in found, "unexpected/duplicate condition")
            row = expected[key]
            entry = prepared[row["dataset"]]
            cache_key = (row["dataset"], row["horizon"])
            if cache_key not in provenance:
                provenance[cache_key] = prepare_condition(config, row, entry, root, COMMIT)[-1]
            cpu = json.loads(
                (root / f"results/manifests/models/{row['family']}_compatibility.json").read_text()
            )
            validate_record(
                record,
                row,
                config,
                entry,
                cpu,
                commit=COMMIT,
                expected_data=provenance[cache_key],
            )
            for attempt in record["attempts"]:
                require(
                    docs[f"{key}/{attempt['operation']}-b{attempt['batch_size']}.json"] == attempt,
                    "attempt file mismatch",
                )
            metadata = next(
                (a["metadata"] for a in record["attempts"] if a["status"] == "passed"), None
            )
            found[key] = {
                "source_archive": path.name,
                "source_member": name,
                "record_hash": stable_hash(record),
                "verified_metadata": metadata,
                "eligibility": {
                    k: record["data"][k]
                    for k in (
                        "unfiltered_train_windows",
                        "total_train_windows",
                        "selected_train_windows",
                        "missing_policy",
                        "fingerprint",
                    )
                },
                **compact(record),
            }
            print(
                row["family"],
                row["dataset"],
                row["horizon"],
                record["fp32_batch1_supported"],
                flush=True,
            )
    require(set(found) == set(expected), "required conditions missing")
    return {
        "execution_commit": COMMIT,
        "status": "validated_returned_evidence",
        "archives": sources,
        "conditions": [found[k] for k in sorted(found)],
        "validated_conditions": len(found),
        "fp32_batch1_supported_conditions": sum(r["fp32_batch1_supported"] for r in found.values()),
        "test_evaluation": False,
        "protocol_frozen": False,
        "main_experiment_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=2, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/new_data_review.json")
    )
    args = parser.parse_args()
    result = review(args.archives, Path.cwd())
    if args.output.exists():
        require(
            json.loads(args.output.read_text(encoding="utf-8")) == result,
            "refusing to replace evidence",
        )
    else:
        write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
