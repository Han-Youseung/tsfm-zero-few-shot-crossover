"""Optional CPU tensor test: interrupted pilot equals uninterrupted execution."""

import dataclasses
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.integration.test_adapter_tensors import batch, toy  # noqa: E402
from tests.unit.test_adapters import adapter_config  # noqa: E402

from tsfm_crossover.evaluation.metrics import TrainScale  # noqa: E402
from tsfm_crossover.experiments.pilot import run_condition, stability  # noqa: E402
from tsfm_crossover.experiments.pilot_config import PilotConfig  # noqa: E402
from tsfm_crossover.models.contract import parameter_hash  # noqa: E402


def test_atomic_loop_sampler_and_optimizer_resume(tmp_path):
    config = PilotConfig(max_steps=3, eval_every=1, checkpoint_every=1, training_batch=2)
    selected = list(range(5))
    identity = {"id": "fixture"}
    reference = toy(adapter_config(max_optimizer_steps=3))

    def run(adapter, dest):
        dest.mkdir(exist_ok=True)

        def make(indices):
            # Fixed tensor fixture with actual last-batch size; no data downloads.
            item = batch(adapter, "train" if indices[0] < 5 else "validation")
            return dataclasses.replace(
                item,
                past=item.past[:1].repeat(len(indices), 1, 1),
                future=item.future[:1].repeat(len(indices), 1, 1),
                window_ids=tuple(str(i) for i in indices),
            )

        return stability(
            adapter,
            config,
            selected,
            [10, 11],
            make,
            TrainScale((1, 1, 1), (5, 5, 5)),
            dest,
            identity,
            {},
            "cpu",
        )

    full = run(reference, tmp_path / "full")
    interrupted = toy(adapter_config(max_optimizer_steps=3))
    original = interrupted.train_step

    def fail_second(item):
        if interrupted.global_step == 1:
            raise RuntimeError("simulated interruption")
        return original(item)

    interrupted.train_step = fail_second
    with pytest.raises(RuntimeError, match="interruption"):
        run(interrupted, tmp_path / "resume")
    restored = toy(adapter_config(max_optimizer_steps=3))
    resumed = run(restored, tmp_path / "resume")
    assert parameter_hash(reference.parameter_state()) == parameter_hash(restored.parameter_state())
    assert full["loop"]["sampler"] == resumed["loop"]["sampler"]
    assert full["loop"]["loss_history"] == resumed["loop"]["loss_history"]
    assert full["best_validation"] == resumed["best_validation"]
    assert resumed["loop"]["sampler"]["position"] == 5
    assert resumed["metadata"]["window_exposures"] == 5
    assert (tmp_path / "resume" / resumed["loop"]["best_checkpoint"]).exists()


def test_cuda_absent_is_pending_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    row = {"id": "no-gpu", "dataset": "ETTh1"}
    status = run_condition(
        PilotConfig(),
        row,
        {"ETTh1": {"qc": {"sha256": "a" * 64}}},
        Path.cwd(),
        "b" * 40,
        tmp_path,
        "cuda",
    )
    assert status == "pending_gpu"
    assert not (tmp_path / "no-gpu/result.json").exists()


def test_batch_ladders_stop_on_failure_and_resume_without_reexecution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tsfm_crossover.experiments import pilot

    # Fake measurement only in the test; no synthetic GPU evidence is published.
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=1000 * 2**20)
    )
    monkeypatch.setattr(
        pilot,
        "measure",
        lambda fn, device: (
            fn(),
            {"seconds": 1.0, "max_memory_allocated_mb": 10, "max_memory_reserved_mb": 20},
        ),
    )
    loaded = []

    def factory():
        adapter = toy()
        adapter.load_model = lambda: loaded.append(True)
        original_train = adapter.train_step
        original_predict = adapter.predict

        def train(item):
            if len(item.window_ids) >= 4:
                raise RuntimeError("out of memory fixture")
            return original_train(item)

        def predict(item):
            if len(item.window_ids) >= 4:
                raise RuntimeError("out of memory fixture")
            return original_predict(item)

        adapter.train_step, adapter.predict = train, predict
        return adapter

    template = toy()

    def make(indices):
        item = batch(template, "train" if indices[0] < 64 else "validation")
        return dataclasses.replace(
            item,
            past=item.past[:1].repeat(len(indices), 1, 1),
            future=item.future[:1].repeat(len(indices), 1, 1),
            window_ids=tuple(str(i) for i in indices),
        )

    args = (
        factory,
        PilotConfig(),
        list(range(64)),
        list(range(64, 128)),
        make,
        tmp_path,
        {"id": "fixture"},
        "cuda",
    )
    result = pilot.feasibility(*args)
    assert result["fp32_batch1_supported"]
    assert len(result["attempts"]) == 6
    assert [r["batch_size"] for r in result["attempts"]] == [1, 2, 4, 1, 2, 4]
    assert len(loaded) == 6
    repeated = pilot.feasibility(*args)
    assert repeated == result and len(loaded) == 6
