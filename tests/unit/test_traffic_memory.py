import pytest

from tsfm_crossover.experiments.traffic_memory import availability


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
