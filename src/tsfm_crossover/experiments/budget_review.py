"""Verify bounded budget evidence and replay early stopping without test evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.experiments.pilot import prepare_condition
from tsfm_crossover.experiments.pilot_config import PilotConfig, execution_plan
from tsfm_crossover.experiments.pilot_review import compact, read_archive, require, validate_record
from tsfm_crossover.tracking.atomic import write_json_atomic

COMMIT = "d6c63f812b36d6c17df56016e6b9415558e11cdc"


def replay_stopping(record, config):
    loop = record["loop"]
    stop = record["stopping_step"]
    require(
        [x["step"] for x in loop["history"]]
        == list(range(config.eval_every, stop + 1, config.eval_every)),
        "evaluation cadence mismatch",
    )
    require(
        [x["step"] for x in loop["loss_history"]] == list(range(1, stop + 1)), "loss steps mismatch"
    )
    best, best_step, bad = None, None, 0
    for item in loop["history"]:
        require(bad < config.patience_evals, "continued after early stopping")
        score = item["metrics"]["macro"][config.selection_metric]
        require(math.isfinite(score), "nonfinite validation")
        if best is None or score < best:
            best, best_step, bad = score, item["step"], 0
        else:
            bad += 1
    require(
        best == loop["best_metric"]
        and best_step == record["best_validation_step"]
        and bad == loop["bad_evals"],
        "best/early-stopping state mismatch",
    )
    require(stop == config.max_steps or bad == config.patience_evals, "premature stopping")
    exposures = sum(x["actual_batch_size"] for x in loop["loss_history"])
    require(exposures == record["metadata"]["window_exposures"], "exposure mismatch")
    require(all(x["actual_batch_size"] == 1 for x in loop["loss_history"]), "batch mismatch")
    require(
        record["equivalent_epochs"] == exposures / record["data"]["selected_train_windows"],
        "equivalent epoch mismatch",
    )
    return "early_stopping" if bad == config.patience_evals else "maximum_steps"


def review(archives, root):
    prepared = json.loads(
        (root / "results/manifests/pilot/prepared_primary9.json").read_text(encoding="utf-8")
    )
    config = PilotConfig.model_validate(
        yaml.safe_load((root / "configs/pilot/budget_confirmation.yaml").read_text())
    )
    plan = execution_plan(config, prepared, COMMIT)
    expected = {r["id"]: r for r in plan["conditions"]}
    found, sources = {}, []
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
            data = prepare_condition(config, row, entry, root, COMMIT)[-1]
            cpu = json.loads(
                (root / f"results/manifests/models/{row['family']}_compatibility.json").read_text()
            )
            validate_record(record, row, config, entry, cpu, commit=COMMIT, expected_data=data)
            reason = replay_stopping(record, config)
            found[key] = {
                "source_archive": path.name,
                "record_hash": stable_hash(record),
                "source_member": name,
                "metadata": record["metadata"],
                "stopping_reason": reason,
                "selected_train_windows": record["data"]["selected_train_windows"],
                "average_window_exposures": record["average_window_exposures"],
                **compact(record),
            }
    require(set(found) == set(expected), "required budget conditions missing")
    return {
        "execution_commit": COMMIT,
        "status": "validated_returned_evidence",
        "archives": sources,
        "conditions": [found[k] for k in sorted(found)],
        "validated_conditions": 4,
        "convergence": "not_established",
        "decision": "bounded_compute_max1000_eval100_patience3",
        "test_evaluation": False,
        "protocol_frozen": False,
        "main_experiment_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=2, type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/budget_review.json")
    )
    args = parser.parse_args()
    result = review(args.archives, Path.cwd())
    if args.output.exists():
        require(
            json.loads(args.output.read_text(encoding="utf-8")) == result, "prior evidence differs"
        )
    else:
        write_json_atomic(args.output, result)
    print("Four budget conditions verified; convergence not established; no main experiment")


if __name__ == "__main__":
    main()
