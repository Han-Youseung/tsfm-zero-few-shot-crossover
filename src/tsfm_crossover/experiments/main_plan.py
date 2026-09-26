"""Approved bounded-compute study, separate from historical pilot gates."""

import json
import re
from pathlib import Path
from typing import Literal

import yaml

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.experiments.study_preflight import StudyPlan

DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Solar",
    "Weather",
    "Tetouan",
    "Traffic",
)


class MainPlan(StudyPlan):
    test_evaluation: Literal[True]
    protocol_frozen: Literal[True]
    main_experiment_allowed: Literal[True]
    analysis_rule: Literal["paired_seed_descriptive_v1"]
    checkpoint_every_steps: Literal[25]
    test_checkpoint_every_windows: Literal[100]


def load_plan(root: Path, config: Path):
    plan = MainPlan.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    prepared = json.loads((root / plan.prepared_manifest).read_text(encoding="utf-8"))
    if set(prepared) != set(DATASETS) or any(
        v["status"] != "ready_with_warnings" or "__bundle" in v["variant"]
        for v in prepared.values()
    ):
        raise ValueError("approved primary9 required")
    return plan, prepared


def grid(plan):
    rows = []
    for family in plan.models:
        for dataset in DATASETS:
            for horizon in plan.horizons:
                for seed in plan.seeds:
                    for rate in (0.0, *plan.sampling_rates):
                        row = dict(
                            family=family,
                            dataset=dataset,
                            horizon=horizon,
                            seed=seed,
                            rate=rate,
                            kind="zero" if rate == 0 else "few",
                        )
                        row["id"] = f"{family}-{dataset}-h{horizon}-s{seed}-r{rate:g}"
                        rows.append(row)
    return rows


def model_pin(root, row):
    cpu = json.loads(
        (root / "results/manifests/models" / f"{row['family']}_compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    chosen = cpu
    if row["family"] == "ttm":
        chosen = cpu["selected_model_revisions"][
            0 if row["horizon"] == 96 else 2 if row["horizon"] == 720 else 1
        ]
    return {
        "repository": cpu["repository"],
        "revision": chosen["revision"],
        "config_sha256": chosen["config_sha256"],
        "code_commit": cpu["code_commit"],
    }


def identity_for(plan, prepared, root, row, commit):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("immutable execution commit required")
    if row not in grid(plan):
        raise ValueError("condition outside approved study grid")
    return {
        "schema_version": 1,
        "commit": commit,
        "condition": row,
        "protocol_hash": stable_hash(plan.model_dump(mode="json")),
        "prepared_hash": stable_hash(prepared),
        "data_sha256": prepared[row["dataset"]]["qc"]["sha256"],
        "model_pin": model_pin(root, row),
    }


def resource_status(row, cuda, vram_gib):
    if not cuda:
        return "pending_gpu"
    if (row["family"], row["dataset"], row["horizon"], row["kind"]) == (
        "moirai1",
        "Traffic",
        720,
        "few",
    ) and vram_gib < 75:
        return "pending_80gb_class_gpu"
    return "ready"  # actual OOM is retained; this is not a guarantee of fit


def verify_runtime(runtime, family, root):
    evidence = json.loads((root / "results/manifests/pilot/budget_review.json").read_text())
    expected = next(
        c["metadata"] for c in evidence["conditions"] if c["condition"]["family"] == family
    )
    if (runtime["torch"], runtime["torch_cuda"]) != (expected["torch"], expected["torch_cuda"]):
        raise ValueError("main CUDA torch build differs from budget evidence")
    if runtime["python"].split(".")[:2] != ["3", "12"]:
        raise ValueError("main runtime requires pinned Python 3.12 minor series")
    exempt = {"pip", "setuptools", "wheel", "tsfm-crossover"}
    for name, version in expected["packages"].items():
        if name not in exempt and runtime["packages"].get(name) != version:
            raise ValueError(f"main package differs from budget evidence: {name}")
    if runtime["official_source"] != expected["official_source"]:
        raise ValueError("official source identity differs from budget evidence")
