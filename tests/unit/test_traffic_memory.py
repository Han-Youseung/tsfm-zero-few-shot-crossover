import json

import pytest

from tsfm_crossover.experiments.traffic_memory import availability, persist_progress


def test_no_gpu_is_not_model_failure():
    assert availability("fp32", False, False, 100) == "pending_gpu"


def test_gpu_name_is_not_a_gate():
    assert availability("fp32", True, False, 8) is None
    assert availability("fp32_cpu_saved", True, False, 80) is None
    assert availability("bf16_cpu_saved", True, True, 80) is None
    assert availability("bf16", True, True, 16) is None


def test_resources_and_precision_not_silently_changed():
    assert availability("bf16_cpu_saved", True, False, 80) == "bf16_not_supported"
    assert "host_ram" in availability("fp32_cpu_saved", True, True, 32)
    with pytest.raises(ValueError):
        availability("fp16", True, True, 80)


@pytest.mark.parametrize("terminal", ["passed", "failed", "not_run"])
def test_atomic_progress_updates_then_preserves_terminal(tmp_path, terminal):
    path = tmp_path / "probe.json"
    record = {"identity": {"commit": "fixture", "profile": "fp32"}, "status": "running"}
    for stage in ("preflight", "zero_shot", "training", "restore"):
        record["stage"] = stage
        persist_progress(path, record)
        assert json.loads(path.read_text())["stage"] == stage
    record["status"] = terminal
    persist_progress(path, record)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="terminal"):
        persist_progress(path, record)
    assert path.read_bytes() == original


def test_progress_different_attempt_cannot_overwrite(tmp_path):
    path = tmp_path / "probe.json"
    persist_progress(path, {"identity": {"commit": "old"}, "status": "running"})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        persist_progress(path, {"identity": {"commit": "new"}, "status": "failed"})
    assert path.read_bytes() == original
