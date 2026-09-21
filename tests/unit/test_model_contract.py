from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tsfm_crossover.models.contract import (
    AdapterContract,
    CandidateState,
    CapabilityReport,
    ModelMetadata,
    parameter_hash,
    require_official_raw_variant,
    require_smoke_split,
    validate_prediction,
)


class MockAdapter(AdapterContract):
    def __init__(self) -> None:
        self.weights = {"weight": [1.0, 2.0]}
        self.optimizer_created = False
        self.backward_calls = 0
        self.evaluation = False

    def load_model(self) -> None:
        pass

    def get_model_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            family="mock",
            repository="official/mock",
            revision="1234567",
            official_code_repository="https://example.invalid/mock",
            code_commit="7654321",
            state=CandidateState.candidate,
            point_forecast_rule="direct",
        )

    def validate_capabilities(self) -> CapabilityReport:
        return CapabilityReport(
            zero_shot=True,
            full_parameter_finetuning=True,
            multivariate=True,
            missing_values=False,
            context_lengths=[256],
            prediction_lengths=[96],
        )

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def predict(self, batch: Any) -> Any:
        return batch

    def point_forecast(self, prediction: Any) -> Any:
        return prediction

    def configure_finetuning(self) -> None:
        self.optimizer_created = True

    def train_step(self, batch: Any) -> float:
        self.backward_calls += 1
        return 1.0

    def validation_step(self, batch: Any) -> float:
        return 1.0

    def save_training_state(self, path: Path) -> None:
        pass

    def load_training_state(self, path: Path) -> None:
        pass

    def count_parameters(self) -> tuple[int, int]:
        return 2, 2

    def parameter_state(self) -> dict[str, Any]:
        return self.weights

    def set_evaluation_mode(self) -> None:
        self.evaluation = True

    def cleanup(self) -> None:
        pass


def test_zero_shot_uses_eval_without_optimizer_or_backward() -> None:
    adapter = MockAdapter()
    before = parameter_hash(adapter.parameter_state())
    assert adapter.zero_shot_predict([[[1.0]]]) == [[[1.0]]]
    assert adapter.evaluation
    assert not adapter.optimizer_created
    assert adapter.backward_calls == 0
    assert parameter_hash(adapter.parameter_state()) == before


def test_zero_shot_rejects_parameter_mutation() -> None:
    adapter = MockAdapter()

    def mutating_predict(batch: Any) -> Any:
        adapter.weights["weight"].append(3.0)
        return batch

    adapter.predict = mutating_predict  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="changed model parameters"):
        adapter.zero_shot_predict([1.0])


def test_parameter_hash_is_order_independent_and_change_sensitive() -> None:
    assert parameter_hash({"a": [1], "b": [2]}) == parameter_hash({"b": [2], "a": [1]})
    assert parameter_hash({"a": [1]}) != parameter_hash({"a": [2]})


def test_prediction_shape_horizon_channel_and_finite_guards() -> None:
    values = [[[1.0, 2.0], [3.0, 4.0]]]
    validate_prediction(
        values, shape=(1, 2, 2), expected_batch=1, expected_horizon=2, expected_channels=2
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_prediction(
            values, shape=(1, 1, 2), expected_batch=1, expected_horizon=2, expected_channels=2
        )
    with pytest.raises(ValueError, match="NaN"):
        validate_prediction(
            [[[float("nan")]]],
            shape=(1, 1, 1),
            expected_batch=1,
            expected_horizon=1,
            expected_channels=1,
        )


def test_compatibility_data_and_split_guards() -> None:
    require_official_raw_variant("ETTh1__official_raw", "A", "ready_with_warnings")
    require_smoke_split("train", training=True)
    require_smoke_split("validation", training=False)
    with pytest.raises(ValueError, match="reject bundle"):
        require_official_raw_variant("ETTh1__bundle_long", "D", "blocked")
    with pytest.raises(ValueError, match="compatibility smoke"):
        require_smoke_split("test", training=True)


def test_candidate_cannot_be_frozen_at_compatibility_stage() -> None:
    payload = MockAdapter().get_model_metadata().model_dump()
    payload["state"] = "frozen"
    with pytest.raises(ValueError, match="cannot be marked frozen"):
        ModelMetadata.model_validate(payload)
