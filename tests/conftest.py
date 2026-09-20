from copy import deepcopy
from datetime import UTC, datetime

import pytest


@pytest.fixture
def valid_config_payload():
    return {
        "dataset": {
            "name": "fixture",
            "path": "tests/fixtures/fake.csv",
            "timestamp_column": "date",
        },
        "split": {"method": "chronological_ratio", "train": 0.6, "validation": 0.2, "test": 0.2},
        "window": {"context_length": 256, "horizons": [96], "drop_last": False},
        "sampling": {
            "strategy": "nested_temporally_stratified",
            "requested_rate": 0.05,
            "seed": 7,
            "source_split": "train",
        },
        "model": {
            "family": "fixture-model",
            "repository": "example/fixture",
            "revision": None,
            "state": "provisional",
        },
        "fine_tuning": {
            "adaptation_mode": "few_shot",
            "budget_type": "max_optimizer_steps",
            "optimizer": None,
            "learning_rate": None,
            "max_optimizer_steps": None,
            "batch_size": None,
            "gradient_accumulation": None,
            "amp": None,
            "trainable_parameters": None,
        },
        "evaluation": {
            "strategy": "rolling_origin",
            "rolling_stride": 1,
            "normalization": "train_fitted_standard_scaler",
            "point_forecast_rule": "pilot_pending",
            "metrics": ["mae", "mse"],
        },
        "reproducibility": {"seeds": [7], "deterministic_requested": False},
        "storage": {
            "results_root": "results",
            "checkpoints_root": "checkpoints",
            "cache_root": ".cache/huggingface",
            "logs_root": "logs",
        },
        "runtime": {"device": "cpu", "num_workers": 0},
        "protocol_metadata": {
            "name": "primary_nested_stepbudget_rolling",
            "state": "provisional",
        },
    }


@pytest.fixture
def valid_result_payload():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return {
        "experiment_id": "fixture__abc123",
        "protocol_id": "primary__abc123",
        "status": "completed",
        "model_family": "fixture-model",
        "model_repository": "example/fixture",
        "model_revision": None,
        "code_commit_sha": "abc123",
        "dataset": "fixture",
        "train_start": "0",
        "train_end": "59",
        "validation_start": "60",
        "validation_end": "79",
        "test_start": "80",
        "test_end": "99",
        "context_length": 16,
        "horizon": 4,
        "adaptation_mode": "few_shot",
        "requested_sampling_rate": 0.1,
        "total_train_windows": 40,
        "selected_train_windows": 4,
        "effective_sampling_rate": 0.1,
        "sampling_strategy": "nested_temporally_stratified",
        "sampling_seed": 7,
        "sampling_manifest_hash": "manifest123",
        "unique_observed_time_points": 20,
        "training_seed": 11,
        "optimizer": "fixture-optimizer",
        "learning_rate": 0.001,
        "max_optimizer_steps": 10,
        "actual_optimizer_steps": 8,
        "equivalent_epochs": 2.0,
        "average_window_exposures": 2.0,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "best_validation_step": 6,
        "stopping_step": 8,
        "trainable_parameters": 10,
        "total_parameters": 12,
        "mae": 1.0,
        "mse": 2.0,
        "evaluation_strategy": "rolling_origin",
        "rolling_stride": 1,
        "number_of_test_windows": 10,
        "point_forecast_rule": "fixture-rule",
        "started_at": now,
        "completed_at": now,
        "train_seconds": 1.0,
        "inference_seconds": 0.5,
        "peak_gpu_memory_mb": None,
        "python_version": "fixture",
        "pytorch_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "error_type": None,
        "error_message": None,
    }


@pytest.fixture
def clone_payload():
    return deepcopy
