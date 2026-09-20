import pytest
from pydantic import ValidationError

from tsfm_crossover.tracking.result_schema import ExperimentResult


def test_completed_result_requires_metrics(valid_result_payload):
    valid_result_payload["mae"] = None
    with pytest.raises(ValidationError, match="mae and mse"):
        ExperimentResult.model_validate(valid_result_payload)


def test_failed_result_requires_error_information(valid_result_payload):
    valid_result_payload["status"] = "failed"
    valid_result_payload["mae"] = None
    valid_result_payload["mse"] = None
    valid_result_payload["completed_at"] = None
    with pytest.raises(ValidationError, match="error_type or error_message"):
        ExperimentResult.model_validate(valid_result_payload)
