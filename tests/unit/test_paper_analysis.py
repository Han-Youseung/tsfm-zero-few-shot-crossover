"""Synthetic arithmetic fixtures only; never represent measured model performance."""

import copy
import itertools
import json
from pathlib import Path

import pytest

from tsfm_crossover.analysis import paper
from tsfm_crossover.analysis.paper_report import render, table

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evidence():
    scope = paper.load_scope(ROOT / "configs/analysis/primary5_h96.yaml")
    records = []
    for family, dataset, seed, rate in itertools.product(
        scope["families"], scope["datasets"], scope["seeds"], scope["rates"]
    ):
        metric = {m: 1.0 if rate == 0 else 0.9 for m in paper.METRICS}
        records.append(
            {
                "identity": {
                    "condition": {
                        "family": family,
                        "dataset": dataset,
                        "seed": seed,
                        "rate": rate,
                        "horizon": 96,
                    },
                    "commit": scope["execution_commit"],
                    "data_sha256": dataset,
                    "prepared_hash": "prepared",
                    "protocol_hash": "protocol",
                },
                "status": "completed",
                "test_evaluation": True,
                "protocol_frozen": True,
                "external_scaler": False,
                "selection": {"parameter_hash": "selected"},
                "test": {
                    "no_update_verified": True,
                    "parameter_hash": "selected",
                    "stride": 1,
                    "window_hash": dataset,
                    "number_of_test_windows": 5,
                    "metrics": {"macro": metric.copy(), "per_channel": [{**metric, "count": 5}]},
                },
                "provenance": {
                    "metric_train_scale": [2.0],
                    "sampling_manifest_hash": f"{dataset}-{seed}",
                    "selected_train_windows": max(1, int(rate * 10000)),
                    "total_train_windows": 10000,
                },
                "training": {
                    "actual_optimizer_steps": 1000,
                    "actual_unique_train_windows": 100,
                    "equivalent_epochs": 0.1,
                    "best_validation_step": 900,
                },
                "adapter_settings": {"channel_names": ["observed"]},
            }
        )
    descriptors = {
        "descriptors": [
            {
                "dataset": d,
                "dataset_sha256": d,
                "channels": 1,
                "train_rows": 12000,
                "frequency_seconds": 3600,
                "channel_macro_lag1_correlation": 0.5,
                "channel_macro_absolute_linear_trend_in_train_std": 0.1,
            }
            for d in scope["datasets"]
        ]
    }
    return scope, records, descriptors


def test_analysis_preserves_primary_and_marks_posthoc(evidence):
    scope, records, descriptors = evidence
    result = paper.analyze(records, scope, descriptors)
    assert result["conditions"] == 270 and len(result["groups"]) == 10
    assert result["scope"]["primary_metric"] == "normalized_mae"
    assert not result["new_gpu_execution"]
    assert not result["test_metrics_recomputed_from_predictions"]
    assert not result["iid_window_bootstrap_used"]
    assert "posthoc" in result["additional_analyses"]
    group = result["groups"][0]
    assert len(group["posthoc_leave_one_seed_out"]) == 3
    assert group["metrics"]["normalized_mae"]["curve"][0]["mean_delta"] == pytest.approx(0.1)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_family"])
def test_exact_grid_required(evidence, change):
    scope, records, _ = evidence
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[0]["identity"]["condition"]["family"] = "timer"
    with pytest.raises(ValueError):
        paper.validate_records(records, scope)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("identity", "commit"), "different"),
        (("identity", "condition", "horizon"), 192),
        (("test_evaluation",), False),
        (("status",), "running"),
        (("protocol_frozen",), False),
        (("external_scaler",), True),
        (("test", "no_update_verified"), False),
        (("test", "parameter_hash"), "changed"),
        (("test", "stride"), 2),
        (("test", "window_hash"), "other"),
        (("provenance", "sampling_manifest_hash"), "other"),
        (("provenance", "metric_train_scale"), [99.0]),
        (("test", "metrics", "macro", "normalized_mae"), float("nan")),
    ],
)
def test_untrusted_record_rejected(evidence, path, value):
    scope, records, _ = evidence
    item = records[0]
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = value
    with pytest.raises(ValueError):
        paper.validate_records(records, scope)


def test_zero_baseline_not_divided(evidence):
    scope, records, descriptors = evidence
    records[0]["test"]["metrics"]["macro"]["normalized_mae"] = 0
    with pytest.raises(ValueError, match="undefined baseline"):
        paper.analyze(records, scope, descriptors)


def test_selected_pool_is_not_unique_exposure(evidence):
    scope, records, descriptors = evidence
    budget = paper.analyze(records, scope, descriptors)["groups"][0]["budget"]
    assert budget["full_pool_conditions"][0]["selected_pool"] == 10000
    assert budget["full_pool_unique_fraction"]["mean"] == 0.01
    assert budget["reached_max_steps"] == 24
    assert budget["best_step_at_budget_cap"] == 0
    assert budget["convergence_established"] is False


def test_mean_and_all_seed_sustained_are_distinct():
    curve = [
        {"rate": r, "mean_delta": 1, "seed_min_relative_improvement": v}
        for r, v in zip([0.005, 0.01, 1.0], [0.1, -0.1, 0.1], strict=True)
    ]
    result = paper.boundaries(curve)
    assert result["mean"]["sustained_improvement"]["upper_inclusive"] == 0.005
    assert result["all_seeds"]["sustained_improvement"]["upper_inclusive"] == 1.0
    assert result["all_seeds"]["nonmonotone_sign"]


def test_source_groups_are_not_five_independent_samples(evidence):
    scope, records, descriptors = evidence
    result = paper.analyze(records, scope, descriptors)
    for group in result["groups"]:
        for point in group["metrics"]["normalized_mae"]["curve"]:
            point["mean_relative_improvement"] = 0.5 if group["dataset"] == "Tetouan" else 0
    row = paper.source_aggregate(result["groups"], scope)[0]
    assert row["dataset_equal_weight_mean"] == pytest.approx(0.1)
    assert row["source_equal_weight_mean"] == pytest.approx(0.5 / 3)
    assert len(row["leave_one_source_out_means"]) == 3
    assert not row["is_confidence_interval"]


def test_descriptor_must_match_dataset(evidence):
    scope, records, descriptors = evidence
    descriptors["descriptors"][0]["dataset_sha256"] = "wrong"
    with pytest.raises(ValueError, match="descriptor data"):
        paper.analyze(records, scope, descriptors)


def test_pinned_review_and_archive_inventory():
    scope = paper.load_scope(ROOT / "configs/analysis/primary5_h96.yaml")
    with pytest.raises(ValueError, match="source ZIP inventory"):
        paper.load_verified(scope, ROOT, [])
    scope["reviews"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="review bytes changed"):
        paper.load_verified(scope, ROOT, [])


def test_changed_archive_bytes_rejected(tmp_path):
    scope = paper.load_scope(ROOT / "configs/analysis/primary5_h96.yaml")
    archives = []
    for ref in scope["reviews"]:
        for item in paper.read(ROOT / ref["path"])["archives"]:
            path = tmp_path / item["name"]
            path.write_bytes(b"not-the-original-archive")
            archives.append(path)
    with pytest.raises(ValueError, match="source ZIP bytes changed"):
        paper.load_verified(scope, ROOT, archives)


def test_report_outputs_are_descriptive_and_escape_html(evidence, tmp_path):
    scope, records, descriptors = evidence
    result = paper.analyze(records, scope, descriptors)
    render(result, tmp_path)
    text = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "{{" not in text and text.count("<svg ") == 5
    assert len(list(tmp_path.glob("*.svg"))) == 5
    assert "&lt;script&gt;" in table(["<script>"], [["<script>"]])
    assert "사전등록" in (tmp_path / "manuscript_sections.txt").read_text(encoding="utf-8")
    json.dumps(result, allow_nan=False)
