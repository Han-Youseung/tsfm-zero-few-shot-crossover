"""Pydantic schemas for the common experiment configuration."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FreezeState(StrEnum):
    provisional = "provisional"
    pilot_validated = "pilot_validated"
    frozen = "frozen"


class AdaptationMode(StrEnum):
    zero_shot = "zero_shot"
    few_shot = "few_shot"


class DatasetConfig(StrictModel):
    name: str = Field(min_length=1)
    source_variant: str = Field(min_length=1)
    path: Path
    timestamp_column: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_variant_name(self) -> DatasetConfig:
        if not self.source_variant.startswith(f"{self.name}__"):
            raise ValueError("source_variant must be namespaced by dataset name")
        return self


class SplitConfig(StrictModel):
    method: str = "chronological_ratio"
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_total(self) -> SplitConfig:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("train, validation, and test ratios must sum to 1")
        return self


class WindowConfig(StrictModel):
    context_length: int = Field(gt=0)
    horizons: list[int] = Field(min_length=1)
    drop_last: bool = False

    @model_validator(mode="after")
    def validate_horizons(self) -> WindowConfig:
        if any(value <= 0 for value in self.horizons):
            raise ValueError("every horizon must be a positive integer")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must not contain duplicates")
        if self.drop_last:
            raise ValueError("drop_last must remain false for few-shot comparability")
        return self


class SamplingConfig(StrictModel):
    strategy: str
    requested_rate: float = Field(ge=0, le=1)
    seed: int
    source_split: str = "train"

    @model_validator(mode="after")
    def reject_non_train_source(self) -> SamplingConfig:
        if self.source_split != "train":
            raise ValueError("training samples may only come from the train split")
        return self


class ModelConfig(StrictModel):
    family: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str | None = None
    state: FreezeState

    @model_validator(mode="after")
    def require_frozen_revision(self) -> ModelConfig:
        if self.state == FreezeState.frozen and not self.revision:
            raise ValueError("a frozen model requires an explicit revision")
        return self


class FineTuningConfig(StrictModel):
    adaptation_mode: AdaptationMode
    budget_type: str = "max_optimizer_steps"
    optimizer: str | None = None
    learning_rate: float | None = Field(default=None, gt=0)
    max_optimizer_steps: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    gradient_accumulation: int | None = Field(default=None, gt=0)
    amp: bool | None = None
    trainable_parameters: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> FineTuningConfig:
        train_fields = (
            self.optimizer,
            self.learning_rate,
            self.max_optimizer_steps,
            self.batch_size,
            self.gradient_accumulation,
            self.amp,
            self.trainable_parameters,
        )
        if self.adaptation_mode == AdaptationMode.zero_shot and any(
            value is not None for value in train_fields
        ):
            raise ValueError("zero-shot configuration must disable every training setting")
        return self


class EvaluationConfig(StrictModel):
    strategy: str
    rolling_stride: int = Field(gt=0)
    normalization: str
    point_forecast_rule: str
    metrics: list[str] = Field(min_length=1)


class ReproducibilityConfig(StrictModel):
    seeds: list[int] = Field(min_length=1)
    deterministic_requested: bool = False


class StorageConfig(StrictModel):
    results_root: Path
    checkpoints_root: Path
    cache_root: Path
    logs_root: Path

    @model_validator(mode="after")
    def validate_distinct_roots(self) -> StorageConfig:
        roots = {
            "results": self.results_root.resolve(),
            "checkpoints": self.checkpoints_root.resolve(),
            "cache": self.cache_root.resolve(),
        }
        names = list(roots)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                left, right = roots[left_name], roots[right_name]
                if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError(f"{left_name} and {right_name} storage roots must not overlap")
        return self


class RuntimeConfig(StrictModel):
    device: str = "auto"
    num_workers: int = Field(default=0, ge=0)


class ProtocolMetadata(StrictModel):
    name: str = Field(min_length=1)
    state: FreezeState


class ExperimentConfig(StrictModel):
    dataset: DatasetConfig
    split: SplitConfig
    window: WindowConfig
    sampling: SamplingConfig
    model: ModelConfig
    fine_tuning: FineTuningConfig
    evaluation: EvaluationConfig
    reproducibility: ReproducibilityConfig
    storage: StorageConfig
    runtime: RuntimeConfig
    protocol_metadata: ProtocolMetadata

    @model_validator(mode="after")
    def validate_adaptation_sampling(self) -> ExperimentConfig:
        mode = self.fine_tuning.adaptation_mode
        rate = self.sampling.requested_rate
        if mode == AdaptationMode.zero_shot and rate != 0:
            raise ValueError("zero-shot requires sampling.requested_rate = 0")
        if mode == AdaptationMode.few_shot and rate <= 0:
            raise ValueError("few-shot requires sampling.requested_rate > 0")
        return self
