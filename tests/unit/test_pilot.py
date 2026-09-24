import csv
import json

import pytest
from scripts.import_adapter_results import validate

from tsfm_crossover.data.pilot_data import digest, load_pilot_values
from tsfm_crossover.evaluation.metrics import (
    StreamingMetrics,
    TrainScale,
    dataset_macro,
    relative_improvement,
    train_scale,
)
from tsfm_crossover.experiments.pilot import validation_subset
from tsfm_crossover.experiments.pilot_config import PilotConfig, execution_plan, token_risk
from tsfm_crossover.experiments.pilot_state import SamplerCursor, check_completed, condition_lock


def test_metrics_tail_count_constant_missing_and_zero():
    scale = TrainScale((2.0, 0.0, None), (10, 10, 0))
    one, tail = StreamingMetrics(scale), StreamingMetrics(scale)
    pred = [[2, 1, 4], [4, 1, 6], [8, 1, 8]]
    target = [[0, 0, float("nan")]] * 3
    one.update(pred, target)
    tail.update(pred[:2], target[:2])
    tail.update(pred[2:], target[2:])
    assert one.compute() == tail.compute()
    out = one.compute()
    assert out["per_channel"][0]["count"] == 3
    assert out["macro"]["normalized_mae"] == pytest.approx(7 / 3)
    assert out["macro"]["normalized_mse"] == 7
    assert out["excluded_normalized_channels"] == 2
    assert relative_improvement(0, 1) is None
    assert relative_improvement(10, 8) == 0.2
    assert dataset_macro([out, out]) == out["macro"]
    with pytest.raises(ValueError, match="nonfinite prediction"):
        one.update([[float("nan"), 0, 0]], [[0, 0, 0]])


def test_train_statistics_do_not_read_validation_or_test():
    class Guarded(list):
        def __getitem__(self, item):
            if item >= 2:
                raise AssertionError("future read")
            return super().__getitem__(item)

    assert train_scale(Guarded([[0], [2], [999]]), 2).std == (1.0,)
    with pytest.raises(ValueError):
        train_scale([[1], [2]], 1, split="validation")
    assert train_scale([[1e8], [1e8 + 2]], 2).std == (1.0,)


def test_pilot_loader_stops_before_test_numeric_values(tmp_path):
    path = tmp_path / "data.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "channel"])
        for i in range(10):
            writer.writerow([str(i), str(i) if i < 8 else "TEST_VALUE_MUST_NOT_PARSE"])
    entry = {
        "path": "data.csv",
        "variant": "fixture__raw",
        "status": "ready_with_warnings",
        "qc": {"rows": 10, "sha256": digest(path), "channel_names": ["channel"]},
    }
    values, _, split = load_pilot_values(tmp_path, entry)
    assert len(values) == split.validation.end == 8
    entry["variant"] = "fixture__bundle_long"
    with pytest.raises(ValueError, match="blocked"):
        load_pilot_values(tmp_path, entry)


def test_sampler_resume_includes_tail_and_does_not_touch_global_rng():
    import random

    cursor = SamplerCursor(5, 7)
    state = random.getstate()
    assert len(cursor.next(3)) == 3
    resumed = SamplerCursor(**cursor.state())
    assert cursor.next(3) == resumed.next(3)
    assert cursor.position == 5
    assert cursor.next(3) == resumed.next(3)
    assert cursor.epoch == 1 and random.getstate() == state
    with pytest.raises(ValueError):
        SamplerCursor(2, 7, order=[0, 0])


def test_duplicate_lock_and_completed_identity(tmp_path):
    with (
        condition_lock(tmp_path),
        pytest.raises(RuntimeError, match="already owned"),
        condition_lock(tmp_path),
    ):
        pass
    assert not (tmp_path / "running.lock").exists()
    dest = tmp_path / "result.json"
    dest.write_text(json.dumps({"identity": {"id": "a"}, "status": "completed"}))
    assert check_completed(dest, {"id": "a"})
    with pytest.raises(ValueError):
        check_completed(dest, {"id": "b"})


def test_plan_ratio_and_budget_separate_and_blocked_datasets_retained():
    a = PilotConfig()
    b = PilotConfig(max_steps=50)
    assert a.sampling_rate == b.sampling_rate == 0.05
    plan = execution_plan(a, {"ETTh1": {"status": "ready_with_warnings"}}, "x")
    assert len(plan["conditions"]) == 124
    assert plan["planned_groups"] == 14  # 8 feasibility + 6 learning-rate conditions
    assert plan["blocked_groups"] == 110
    assert len({r["id"] for r in plan["conditions"]}) == 124
    assert validation_subset(list(range(100)), 4) == [12, 37, 62, 87]
    assert token_risk(370, 720)["moirai_joint_tokens"] == 7400
    assert token_risk(862, 720)["hard_joint_512_token_limit"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"test_evaluation": True},
        {"accumulation": 2},
        {"max_steps": 201},
        {"evaluation_batch": 2},
        {"validation_windows": 65},
        {"learning_rates": {"ttm": [1, 2, 3, 4], "moirai1": [1]}},
    ],
)
def test_pilot_limits(changes):
    with pytest.raises(ValueError):
        PilotConfig(**changes)


def test_adapter_evidence_checks_missing_or_duplicate_rejected():
    from pathlib import Path

    records = [
        json.loads(p.read_text()) for p in Path("results/manifests/adapters/gpu").glob("*-h*.json")
    ]
    validate(records)
    with pytest.raises(ValueError, match="duplicate"):
        validate(records + [records[0]])
    del records[0]["checks"]["restore_rng"]
    with pytest.raises(ValueError, match="incomplete"):
        validate(records)


def test_model_independent_pilot_sampling_and_validation(tmp_path):
    from tsfm_crossover.experiments.pilot import prepare_condition

    path = tmp_path / "fixture.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "a", "b"])
        writer.writerows((i, i % 17, i % 7) for i in range(2000))
    entry = {
        "path": "fixture.csv",
        "variant": "ETTh1__fixture_raw",
        "status": "ready_with_warnings",
        "qc": {"rows": 2000, "sha256": digest(path), "channel_names": ["a", "b"]},
    }
    outputs = []
    for family in ("ttm", "moirai1"):
        result = prepare_condition(
            PilotConfig(),
            {"family": family, "id": family, "horizon": 96},
            entry,
            tmp_path,
            "a" * 40,
        )
        outputs.append(result)
    assert outputs[0][1] == outputs[1][1]
    assert outputs[0][2] == outputs[1][2]
    assert outputs[0][0].sampling_manifest_hash == outputs[1][0].sampling_manifest_hash
    assert outputs[0][5]["metric_train_scale"]["counts"] == (1200, 1200)
    assert len(outputs[0][2]) == 64
    assert outputs[0][5]["selected_train_windows"] == 29
