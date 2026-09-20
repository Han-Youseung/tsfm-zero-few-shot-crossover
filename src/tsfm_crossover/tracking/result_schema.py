"""Common result record shared by every model adapter."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResultStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    experiment_id: str
    protocol_id: str
    status: ResultStatus
    model_family: str
    model_repository: str
    model_revision: str | None
    code_commit_sha: str | None

    dataset: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    context_length: int = Field(gt=0)
    horizon: int = Field(gt=0)

    adaptation_mode: str
    requested_sampling_rate: float = Field(ge=0, le=1)
    total_train_windows: int = Field(ge=0)
    selected_train_windows: int = Field(ge=0)
    effective_sampling_rate: float = Field(ge=0, le=1)
    sampling_strategy: str
    sampling_seed: int
    sampling_manifest_hash: str | None
    unique_observed_time_points: int = Field(ge=0)

    training_seed: int
    optimizer: str | None
    learning_rate: float | None = Field(default=None, gt=0)
    max_optimizer_steps: int | None = Field(default=None, gt=0)
    actual_optimizer_steps: int = Field(ge=0)
    equivalent_epochs: float = Field(ge=0)
    average_window_exposures: float = Field(ge=0)
    batch_size: int | None = Field(default=None, gt=0)
    gradient_accumulation: int | None = Field(default=None, gt=0)
    best_validation_step: int | None = Field(default=None, ge=0)
    stopping_step: int | None = Field(default=None, ge=0)
    trainable_parameters: int | None = Field(default=None, ge=0)
    total_parameters: int | None = Field(default=None, ge=0)

    mae: float | None = Field(default=None, ge=0)
    mse: float | None = Field(default=None, ge=0)
    evaluation_strategy: str
    rolling_stride: int = Field(gt=0)
    number_of_test_windows: int = Field(ge=0)
    point_forecast_rule: str

    started_at: datetime | None
    completed_at: datetime | None
    train_seconds: float | None = Field(default=None, ge=0)
    inference_seconds: float | None = Field(default=None, ge=0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)
    python_version: str
    pytorch_version: str | None
    cuda_version: str | None
    gpu_name: str | None

    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_status_requirements(self) -> ExperimentResult:
        if self.status == ResultStatus.completed:
            if self.mae is None or self.mse is None:
                raise ValueError("completed results require both mae and mse")
            if self.completed_at is None:
                raise ValueError("completed results require completed_at")
        if self.status == ResultStatus.failed and not (self.error_type or self.error_message):
            raise ValueError("failed results require error_type or error_message")
        if self.selected_train_windows > self.total_train_windows:
            raise ValueError("selected_train_windows cannot exceed total_train_windows")
        return self
