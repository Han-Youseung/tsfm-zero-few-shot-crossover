import copy
import json
from pathlib import Path

import pytest

from tsfm_crossover.analysis.crossover import interval, source_group, summarize
from tsfm_crossover.analysis.train_features import features
from tsfm_crossover.evaluation.metrics import StreamingMetrics, TrainScale
from tsfm_crossover.experiments.main_plan import (
    MainPlan,
    grid,
    identity_for,
    load_plan,
    resource_status,
    verify_runtime,
)
from tsfm_crossover.experiments.main_study import final_windows, immutable

ROOT = Path(__file__).resolve().parents[2]


def test_main_grid_authorized_but_historical_pilot_still_closed():
    plan, prepared = load_plan(ROOT, ROOT / "configs/study/main.yaml")
    rows = grid(plan)
    assert len(rows) == len({r["id"] for r in rows}) == 1944
    assert sum(r["kind"] == "zero" for r in rows) == 216
    raw = plan.model_dump(mode="json")
    for key in ("protocol_frozen", "main_experiment_allowed", "test_evaluation"):
        wrong = copy.deepcopy(raw)
        wrong[key] = False
        with pytest.raises(ValueError):
            MainPlan.model_validate(wrong)
    first = identity_for(plan, prepared, ROOT, rows[0], "a" * 40)
    wrong = copy.deepcopy(prepared)
    wrong["ETTh1"]["qc"]["sha256"] = "b" * 64
    assert first != identity_for(plan, wrong, ROOT, rows[0], "a" * 40)
    with pytest.raises(ValueError):
        identity_for(plan, prepared, ROOT, rows[0], "main")


@pytest.mark.parametrize(
    "cuda,vram,kind,expected",
    [
        (False, 0, "few", "pending_gpu"),
        (True, 40, "few", "pending_80gb_class_gpu"),
        (True, 79, "few", "ready"),
        (True, 40, "zero", "ready"),
    ],
)
def test_traffic_resource_gate(cuda, vram, kind, expected):
    row = dict(family="moirai1", dataset="Traffic", horizon=720, kind=kind)
    assert resource_status(row, cuda, vram) == expected


def test_no_test_values_before_selection(monkeypatch):
    from tsfm_crossover.experiments import main_study

    monkeypatch.setattr(main_study, "load_final_values", lambda *a: pytest.fail("test opened"))
    with pytest.raises(ValueError, match="locked"):
        final_windows(ROOT, {}, None, {"status": "running"})


def test_metric_resume_and_scale_guard():
    metric = StreamingMetrics(TrainScale((2.0, 0.0), (8, 8)))
    metric.update([[4, 2]], [[2, float("nan")]])
    other = StreamingMetrics(metric.scale)
    other.restore(json.loads(json.dumps(metric.state())))
    assert other.compute() == metric.compute()
    state = metric.state()
    state["absolute"][0] = -1
    with pytest.raises(ValueError):
        other.restore(state)
    with pytest.raises(ValueError, match="scale"):
        StreamingMetrics(TrainScale((3.0, 0.0), (8, 8))).restore(metric.state())


def test_immutable_selection(tmp_path):
    path = tmp_path / "selection.json"
    immutable(path, {"checkpoint": 100})
    immutable(path, {"checkpoint": 100})
    with pytest.raises(ValueError):
        immutable(path, {"checkpoint": 200})


def test_all_rates_and_both_models_share_one_nested_manifest(monkeypatch):
    from tsfm_crossover.data.splits import chronological_split
    from tsfm_crossover.experiments import pilot
    from tsfm_crossover.experiments.main_study import loop_config

    plan, _ = load_plan(ROOT, ROOT / "configs/study/main.yaml")
    split = chronological_split(2500)
    values = [[float(i), float(i + 1)] for i in range(split.validation.end)]
    monkeypatch.setattr(pilot, "load_pilot_values", lambda *a: (values, ("a", "b"), split))
    manifests, selected_ids = [], []
    for family, rate in (("ttm", 0.0), ("ttm", 0.005), ("ttm", 0.01), ("moirai1", 0.01)):
        row = dict(family=family, rate=rate, seed=1729, horizon=96, id="fixture")
        prepared = pilot.prepare_condition(
            loop_config(plan, row),
            row,
            {"variant": "fixture", "qc": {"sha256": "a" * 64}},
            ROOT,
            "b" * 40,
            sampling_rates=(0.0, *plan.sampling_rates),
        )
        manifests.append(prepared[-1]["sampling_manifest"])
        selected_ids.append({w.window_id for w in prepared[1]})
    assert all(m == manifests[0] for m in manifests)
    assert selected_ids[0] == set()
    assert selected_ids[1] <= selected_ids[2] == selected_ids[3]
    assert len(selected_ids[1]) == max(1, int(0.005 * manifests[0]["total_train_windows"]))


def test_rolling_target_counts_do_not_count_missing_targets():
    from types import SimpleNamespace

    from tsfm_crossover.experiments.main_study import target_counts

    values = [[1.0, 1.0], [1.0, float("nan")], [1.0, 1.0], [1.0, 1.0]]
    windows = [
        SimpleNamespace(target_start=0, target_end=3),
        SimpleNamespace(target_start=1, target_end=4),
    ]
    assert target_counts(values, windows) == [6, 4]


def test_preflight_json_roundtrip_is_idempotent(tmp_path, monkeypatch):
    from tsfm_crossover.experiments import study_preflight

    record = {"split": {"ratios": (0.6, 0.2, 0.2)}, "test_values_read": False}
    dest = tmp_path / "preflight.json"
    original = json.dumps(record)
    dest.write_text(original)
    monkeypatch.setattr(study_preflight, "preflight", lambda *a: record)
    monkeypatch.setattr("sys.argv", ["preflight", "--output", str(dest)])
    study_preflight.main()
    assert dest.read_text() == original


def test_crossover_nonmonotone_missing_and_source_clusters():
    out = interval([0.005, 0.01, 0.02, 0.05], [False, True, False, True])
    assert out["first_improvement"] == {"lower_exclusive": 0.005, "upper_inclusive": 0.01}
    assert out["sustained_improvement"]["lower_exclusive"] == 0.02
    assert out["nonmonotone_sign"]
    assert interval([0.1], [False])["status"] == "not_observed_on_grid"
    assert source_group("ETTh1") == source_group("ETTm1")
    assert source_group("ETTh2") != source_group("ETTm1")
    assert summarize([], [0.01], [1, 2, 3])["groups"] == []


def test_paired_seed_crossover_no_iid_ci():
    records = []
    for seed in (1, 2, 3):
        for rate in (0.0, 0.01, 0.1):
            records.append(
                {
                    "identity": {
                        "condition": {
                            "family": "ttm",
                            "dataset": "ETTh1",
                            "horizon": 96,
                            "seed": seed,
                            "rate": rate,
                        }
                    },
                    "test": {
                        "window_hash": "common",
                        "metrics": {
                            "macro": {"normalized_mae": 1 - rate},
                            "per_channel": [{"count": 10}],
                        },
                    },
                }
            )
    out = summarize(records, [0.01, 0.1], [1, 2, 3])
    assert out["groups"][0]["all_three_seeds"]["status"] == "observed_on_grid"
    assert not out["source_group_aggregates"]
    assert summarize(records[:-1], [0.01, 0.1], [1, 2, 3])["groups"][0]["status"] == "incomplete"
    records[-1]["test"]["window_hash"] = "wrong"
    with pytest.raises(ValueError, match="unpaired"):
        summarize(records, [0.01, 0.1], [1, 2, 3])


def test_features_never_use_future():
    values = [[float(i), 1.0] for i in range(20)]
    baseline = features(values, 10)
    assert baseline["constant_channels"] == 1
    assert baseline["channel_macro_lag1_correlation"] == pytest.approx(1)
    values[10:] = [[1e20, float("nan")]] * 10
    assert features(values, 10) == baseline


def test_runtime_is_pinned_to_gpu_budget_not_cpu_versions():
    evidence = json.loads((ROOT / "results/manifests/pilot/budget_review.json").read_text())
    for condition in evidence["conditions"]:
        runtime = copy.deepcopy(condition["metadata"])
        family = condition["condition"]["family"]
        verify_runtime(runtime, family, ROOT)
        runtime["torch"] = runtime["torch"].split("+")[0] + "+cpu"
        with pytest.raises(ValueError):
            verify_runtime(runtime, family, ROOT)


def test_main_notebook_is_output_free_python_and_no_hidden_execution():
    import ast

    notebook = json.loads((ROOT / "notebooks/40_main_study.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            source = "".join(cell["source"])
            ast.parse(source)
            assert '"pull"' not in source
    assert [c["id"] for c in notebook["cells"]] == [
        "intro",
        "runtime",
        "repository",
        "drive",
        "installation",
        "plan",
        "data",
        "run",
        "export",
        "next",
    ]


def test_main_requirements_match_returned_gpu_versions():
    evidence = json.loads((ROOT / "results/manifests/pilot/budget_review.json").read_text())
    exempt = {"pip", "setuptools", "wheel", "tsfm-crossover", "granite-tsfm", "uni2ts"}
    for family in ("ttm", "moirai1"):
        packages = next(
            c["metadata"]["packages"]
            for c in evidence["conditions"]
            if c["condition"]["family"] == family
        )
        lines = (ROOT / f"requirements/{family}-main.txt").read_text().splitlines()
        pins = dict(line.split("==") for line in lines if "==" in line)
        assert pins == {k: v for k, v in packages.items() if k not in exempt}


def test_final_import_rejects_revision_commit_fingerprint_and_mask_tampering():
    from types import SimpleNamespace

    from tests.unit.test_adapters import adapter_config

    from tsfm_crossover.data.common import stable_hash
    from tsfm_crossover.experiments.main_review import validate_result

    plan, prepared = load_plan(ROOT, ROOT / "configs/study/main.yaml")
    row = grid(plan)[0]
    identity = identity_for(plan, prepared, ROOT, row, "a" * 40)
    settings = adapter_config()
    metric = StreamingMetrics(TrainScale((1.0, 2.0, 0.0), (10, 10, 10)))
    metric.update([[1, 2, 3]], [[0, 0, 0]])
    provenance = {"metric_train_scale": metric.state()["scale"]}
    evidence = json.loads((ROOT / "results/manifests/pilot/budget_review.json").read_text())
    runtime = copy.deepcopy(
        next(c["metadata"] for c in evidence["conditions"] if c["condition"]["family"] == "ttm")
    )
    runtime["vram_gib"] = 40
    record = {
        "status": "completed",
        "identity": identity,
        "adapter_settings": settings.model_dump(mode="json"),
        "provenance": provenance,
        "test_evaluation": True,
        "protocol_frozen": True,
        "precision": "float32",
        "external_scaler": False,
        "runtime": runtime,
        "runtime_hash": stable_hash(runtime),
        "selection": {
            "identity": identity,
            "status": "selection_locked",
            "parameter_hash": "c" * 64,
            "checkpoint_sha256": None,
            "selected_by": "pretrained_zero_shot",
        },
        "test": {
            "parameter_hash": "c" * 64,
            "no_update_verified": True,
            "stride": 1,
            "number_of_test_windows": 1,
            "window_hash": "d" * 64,
            "metrics": metric.compute(),
        },
        "trainable_parameters": 3,
        "total_parameters": 3,
        "training": {"actual_optimizer_steps": 0, "validation_history": []},
    }
    args = (
        identity,
        settings,
        provenance,
        {"count": 1, "hash": "d" * 64, "target_counts": [1, 1, 1]},
        SimpleNamespace(max_optimizer_steps=1000),
        ROOT,
    )
    validate_result(record, *args)
    for key in ("commit", "data_sha256", "model_pin"):
        wrong = copy.deepcopy(record)
        wrong["identity"][key] = "wrong"
        with pytest.raises(ValueError, match="mismatch"):
            validate_result(wrong, *args)
    wrong = copy.deepcopy(record)
    wrong["test"]["metrics"]["per_channel"][0]["count"] = 2
    with pytest.raises(ValueError, match="mask"):
        validate_result(wrong, *args)
    wrong = copy.deepcopy(record)
    wrong["runtime"]["device"] = "cpu"
    wrong["runtime_hash"] = stable_hash(wrong["runtime"])
    with pytest.raises(ValueError, match="CPU"):
        validate_result(wrong, *args)
