from tsfm_crossover.config.identifiers import (
    ExperimentIdentity,
    build_experiment_id,
    build_protocol_id,
    build_protocol_id_from_config,
)
from tsfm_crossover.config.schema import ExperimentConfig


def test_protocol_hash_is_key_order_independent():
    left = {"split": {"train": 0.6, "test": 0.2}, "sampling": "nested"}
    right = {"sampling": "nested", "split": {"test": 0.2, "train": 0.6}}
    assert build_protocol_id("primary", left) == build_protocol_id("primary", right)


def test_protocol_hash_changes_with_configuration():
    assert build_protocol_id("primary", {"stride": 1}) != build_protocol_id(
        "primary", {"stride": 2}
    )


def test_protocol_id_from_config_changes_with_point_rule(valid_config_payload):
    first = ExperimentConfig.model_validate(valid_config_payload)
    valid_config_payload["evaluation"]["point_forecast_rule"] = "another-rule"
    second = ExperimentConfig.model_validate(valid_config_payload)
    assert build_protocol_id_from_config(first) != build_protocol_id_from_config(second)


def _identity(attempt: int = 1) -> ExperimentIdentity:
    return ExperimentIdentity(
        protocol_id="primary__abc",
        model_family="ttm",
        model_repository="example/model",
        model_revision=None,
        model_state="provisional",
        dataset="fixture",
        context_length=256,
        horizon=96,
        sampling_rate=0.05,
        selected_train_windows=5,
        sampling_seed=7,
        training_seed=11,
        run_attempt=attempt,
    )


def test_experiment_id_is_stable_and_attempt_is_separate():
    base_one, run_one = build_experiment_id(_identity(1))
    base_two, run_two = build_experiment_id(_identity(2))
    assert base_one == base_two
    assert run_one != run_two
    assert "attempt-01" in run_one
    assert "attempt-02" in run_two
