"""CPU toy only: no real model, GPU claims or weight downloads."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.integration.test_adapter_tensors import batch, toy  # noqa: E402

from tsfm_crossover.evaluation.metrics import StreamingMetrics, TrainScale  # noqa: E402
from tsfm_crossover.experiments.main_plan import grid, load_plan  # noqa: E402
from tsfm_crossover.experiments.main_study import evaluate_final, run  # noqa: E402
from tsfm_crossover.models.contract import parameter_hash  # noqa: E402


def test_test_prediction_is_explicit_and_target_free():
    adapter = toy()
    item = batch(adapter, "test")
    with pytest.raises(ValueError):
        adapter.predict(item)
    with pytest.raises(ValueError, match="target-free"):
        adapter.predict_test(item)
    before = parameter_hash(adapter.parameter_state())
    point = adapter.predict_test(dataclasses.replace(item, future=None))
    assert point.shape == (2, 96, 3)
    assert before == parameter_hash(adapter.parameter_state())
    assert adapter.optimizer is None
    adapter.configure_finetuning()
    with pytest.raises(ValueError, match="test split"):
        adapter.train_step(item)
    with pytest.raises(ValueError, match="test split"):
        adapter.validation_step(item)


def test_final_resume_equals_uninterrupted_and_does_not_update(tmp_path):
    windows = [SimpleNamespace(window_id=str(i)) for i in range(5)]
    identity = {"condition": "fixture"}
    scale = TrainScale((2.0, 1.0, 0.0), (10, 10, 10))

    def make(items):
        i = int(items[0].window_id)
        item = batch(toy(), "test")
        targets = item.future[:1].tolist()
        targets[0][0][1] = float("nan")
        return dataclasses.replace(
            item, past=item.past[:1] * (i + 1), future=targets, window_ids=(str(i),)
        )

    def evaluate(adapter, dest):
        dest.mkdir(exist_ok=True)
        selected = {
            "identity": identity,
            "parameter_hash": parameter_hash(adapter.parameter_state()),
        }
        return evaluate_final(
            adapter, windows, make, scale, dest, identity, selected, every=2, device="cpu"
        )

    full = evaluate(toy(), tmp_path / "full")
    interrupted = toy()
    original = interrupted.predict_test
    calls = []

    def fail(item):
        calls.append(item.window_ids[0])
        if item.window_ids == ("3",):
            raise RuntimeError("interrupted test")
        return original(item)

    interrupted.predict_test = fail
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate(interrupted, tmp_path / "resume")
    assert calls == ["0", "1", "2", "3"]
    assert json.loads((tmp_path / "resume/test-progress.json").read_text())["next_window"] == 2
    restored = toy()
    original = restored.predict_test
    calls.clear()

    def record(item):
        calls.append(item.window_ids[0])
        return original(item)

    restored.predict_test = record
    resumed = evaluate(restored, tmp_path / "resume")
    assert calls == ["2", "3", "4"]
    assert resumed["metrics"] == full["metrics"]
    assert restored.global_step == restored.backward_calls == 0
    assert resumed["no_update_verified"]
    calls.clear()
    assert evaluate(restored, tmp_path / "resume")["metrics"] == full["metrics"]
    assert calls == []


def test_vectorized_masked_metrics_match_reference():
    import numpy as np

    rng = np.random.default_rng(3)
    pred, target = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    target[::2, 1] = np.nan
    scale = TrainScale((0.0, 2.0, 3.0), (10, 10, 10))
    first, second = StreamingMetrics(scale), StreamingMetrics(scale)
    first.update(pred, target)
    second.update_array(pred, target)
    assert first.count == second.count
    assert first.compute()["macro"] == pytest.approx(second.compute()["macro"])


def test_main_cuda_absent_never_opens_data(tmp_path, monkeypatch):
    from tsfm_crossover.experiments import main_study

    root = Path.cwd()
    plan, prepared = load_plan(root, root / "configs/study/main.yaml")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        main_study, "prepare_condition", lambda *a, **kw: pytest.fail("data opened")
    )
    row = grid(plan)[0]
    assert run(plan, row, prepared, root, "b" * 40, tmp_path) == "pending_gpu"
    assert not (tmp_path / row["id"] / "result.json").exists()


def test_main_training_selection_then_test_and_completed_skip(tmp_path, monkeypatch):
    from tests.unit.test_adapters import adapter_config

    from tsfm_crossover.experiments import main_study, pilot

    root = Path.cwd()
    plan, prepared = load_plan(root, root / "configs/study/main.yaml")
    row = grid(plan)[1]  # few-shot, no actual external execution
    settings = adapter_config(max_optimizer_steps=3)
    cfg = SimpleNamespace(
        seed=11,
        max_steps=3,
        eval_every=1,
        checkpoint_every=1,
        patience_evals=3,
        training_batch=1,
        selection_metric="normalized_mae",
    )
    monkeypatch.setattr(main_study, "loop_config", lambda *a: cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=80 * 2**30)
    )
    monkeypatch.setattr(main_study, "verify_runtime", lambda *a: None)
    monkeypatch.setattr(main_study.subprocess, "check_output", lambda *a, **kw: "fixture driver")
    original_measure = pilot.measure
    monkeypatch.setattr(main_study, "measure", lambda f, d: original_measure(f, "cpu"))
    monkeypatch.setattr(pilot, "measure", lambda f, d: original_measure(f, "cpu"))
    stages, trained_adapters = [], []

    def factory(*a, **kw):
        adapter = toy(settings)
        adapter.load_model = lambda: None
        # GPU is simulated for control-flow testing; never exported as evidence.
        trained_adapters.append(adapter)
        return adapter

    monkeypatch.setattr(main_study, "create_adapter", factory)

    def train_batch(ids):
        item = batch(toy(settings), "train" if ids[0] < 10 else "validation")
        return dataclasses.replace(
            item, past=item.past[:1], future=item.future[:1].tolist(), window_ids=(str(ids[0]),)
        )

    def prepare(*a, **kw):
        stages.append("train_validation_only")
        return (
            settings,
            [0, 1],
            [10],
            train_batch,
            TrainScale((1, 1, 1), (10, 10, 10)),
            {"sampling_manifest": {"manifest_hash": "b" * 64}, "selected_train_windows": 2},
        )

    monkeypatch.setattr(main_study, "prepare_condition", prepare)

    def final(*a):
        directory = tmp_path / row["id"]
        assert (directory / "training.json").exists()
        assert (directory / "selection.json").exists()
        assert a[-1]["selected_by"] == "validation_only"
        stages.append("test_after_selection")
        item = train_batch([10])
        return (
            [SimpleNamespace(window_id="test-1")],
            lambda _: dataclasses.replace(item, split="test", window_ids=("test-1",)),
            {"nominal": 1, "target_counts": [96, 96, 96]},
        )

    monkeypatch.setattr(main_study, "final_windows", final)
    assert run(plan, row, prepared, root, "b" * 40, tmp_path) == "completed"
    assert stages == ["train_validation_only", "test_after_selection"]
    assert len(trained_adapters) == 2  # separate fresh pretrained model before restore
    result = json.loads((tmp_path / row["id"] / "result.json").read_text())
    assert result["training"]["actual_optimizer_steps"] == 3
    assert result["test"]["no_update_verified"]
    assert run(plan, row, prepared, root, "b" * 40, tmp_path) == "skipped_completed"
    assert len(stages) == 2
