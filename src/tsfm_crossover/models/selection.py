"""Safety gate for selecting a MOIRAI research candidate.

This is deliberately not a production adapter.  It records and validates evidence
without importing torch, Uni2TS, or downloading model weights in the default suite.
"""

from __future__ import annotations

from enum import StrEnum
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REQUIRED_HORIZONS = (96, 192, 336, 720)
CANDIDATE_PRIORITY = (
    "Salesforce/moirai-1.1-R-small",
    "Salesforce/moirai-1.0-R-small",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DecisionStatus(StrEnum):
    investigating = "investigating"
    rejected = "rejected"
    selected_cpu = "selected_cpu"
    pending_gpu = "pending_gpu"
    frozen = "frozen"


class RequiredCapabilities(StrictModel):
    official_salesforce_weights: bool
    official_uni2ts_load: bool
    joint_multivariate_input: bool
    joint_multivariate_output: bool
    full_parameter_finetuning: bool
    official_objective: bool
    context_512: bool
    direct_horizons: dict[int, bool]
    no_recursive_forecasting: bool
    same_pretrained_checkpoint: bool
    original_unit_point_forecast: bool
    colab_installable: bool
    licenses_confirmed: bool
    revision_pinned: bool

    @model_validator(mode="after")
    def require_four_horizons(self) -> RequiredCapabilities:
        if set(self.direct_horizons) != set(REQUIRED_HORIZONS):
            raise ValueError(f"direct_horizons must be exactly {REQUIRED_HORIZONS}")
        return self

    def all_satisfied(self) -> bool:
        values = self.model_dump(exclude={"direct_horizons"}).values()
        return all(values) and all(self.direct_horizons.values())


class MultivariateEvidence(StrictModel):
    semantics: Literal["joint_target_dimensions"]
    batch_size: int = Field(ge=1)
    target_dim: Literal[7]
    input_shape: tuple[int, int, Literal[7]]
    output_shape: tuple[int, int, Literal[7]]
    channel_order_preserved: bool
    permutation_equivariant: bool
    channelwise_loop: Literal[False] = False

    @model_validator(mode="after")
    def dimensions_are_not_batches(self) -> MultivariateEvidence:
        if self.input_shape[0] != self.batch_size or self.output_shape[0] != self.batch_size:
            raise ValueError("batch and channel dimensions must remain distinct")
        if self.input_shape[1] != 512:
            raise ValueError("selection gate requires context length 512")
        return self


class TrainingEvidence(StrictModel):
    wrapper: str
    objective: str
    finetune_pattern: Literal["full"]
    custom_loss: Literal[False] = False
    peft: Literal[False] = False
    optimizer_steps: int = Field(ge=1, le=2)
    finite_loss: bool
    finite_gradients: bool
    parameter_update: bool
    checkpoint_round_trip: bool
    train_split_only: bool
    test_split_used: Literal[False] = False


class ScalingEvidence(StrictModel):
    model_native: bool
    external_scaler: bool
    original_unit_output: bool

    @model_validator(mode="after")
    def block_double_scaling(self) -> ScalingEvidence:
        if self.model_native and self.external_scaler:
            raise ValueError("model-native and external scaling cannot both be enabled")
        return self


class RevisionEvidence(StrictModel):
    huggingface_repository: str
    huggingface_commit: str = Field(min_length=40, max_length=40)
    config_sha256: str = Field(min_length=64, max_length=64)
    code_repository: str
    code_ref: str
    release: str
    release_tag_commit: str = Field(min_length=40, max_length=40)
    code_commit: str = Field(min_length=40, max_length=40)
    code_license: str
    weight_license: str


class CandidateDecision(StrictModel):
    candidate: str
    priority: int = Field(ge=1)
    required_capabilities: RequiredCapabilities
    capability_results: dict[str, object]
    selected: bool
    rejection_reasons: list[str]
    evidence: dict[str, object]
    revision: RevisionEvidence
    environment: dict[str, object]
    decision_status: DecisionStatus
    decision_date: str
    gpu_validated: bool = False

    @model_validator(mode="after")
    def enforce_selection_gate(self) -> CandidateDecision:
        if self.selected and not self.required_capabilities.all_satisfied():
            raise ValueError("candidate cannot be selected with a missing required capability")
        if self.selected and self.rejection_reasons:
            raise ValueError("selected candidate cannot have rejection reasons")
        if self.decision_status == DecisionStatus.rejected and self.selected:
            raise ValueError("rejected candidate cannot be selected")
        if self.decision_status == DecisionStatus.frozen and not self.gpu_validated:
            raise ValueError("CPU evidence alone cannot freeze a candidate")
        return self


class ModelSelectionGate(StrictModel):
    schema_version: int
    candidates: list[CandidateDecision] = Field(min_length=1)
    selected_candidate: str | None
    moirai2_status: Literal["incompatible_with_required_multivariate_protocol"]
    production_adapter_allowed: bool
    gpu_compatibility: Literal["pending_gpu", "passed"] = "pending_gpu"
    protocol_frozen: Literal[False] = False

    @model_validator(mode="after")
    def require_gpu_gate(self) -> ModelSelectionGate:
        if self.production_adapter_allowed and self.gpu_compatibility != "passed":
            raise ValueError("production adapter requires both-model GPU gate")
        return self

    @model_validator(mode="after")
    def enforce_priority(self) -> ModelSelectionGate:
        ordered = sorted(self.candidates, key=lambda item: item.priority)
        selected = [item for item in ordered if item.selected]
        if len(selected) > 1:
            raise ValueError("only one candidate may be selected")
        expected = selected[0].candidate if selected else None
        if self.selected_candidate != expected:
            raise ValueError("selected_candidate does not match candidate decisions")
        for index, item in enumerate(ordered):
            if item.selected and any(
                later.priority > item.priority for later in ordered[index + 1 :]
            ):
                raise ValueError(
                    "lower-priority candidates must not run after first-priority success"
                )
        return self


def next_candidate(completed: list[CandidateDecision]) -> str | None:
    """Return the next allowed candidate, stopping immediately after success."""
    if any(item.selected for item in completed):
        return None
    attempted = {item.candidate for item in completed}
    return next((name for name in CANDIDATE_PRIORITY if name not in attempted), None)


def sample_median(samples: list[list[list[list[float]]]]) -> list[list[list[float]]]:
    """Reduce [batch, sample, horizon, channel] samples without a NumPy dependency."""
    if not samples or not samples[0]:
        raise ValueError("at least one probabilistic sample is required")
    result: list[list[list[float]]] = []
    for batch in samples:
        horizon = len(batch[0])
        channels = len(batch[0][0])
        if any(len(sample) != horizon for sample in batch):
            raise ValueError("sample horizon shapes differ")
        result.append(
            [
                [median(sample[t][channel] for sample in batch) for channel in range(channels)]
                for t in range(horizon)
            ]
        )
    return result


def validate_probe_data(source_variant: str, split: str) -> None:
    if not source_variant.endswith("__official_raw"):
        raise ValueError("blocked bundle variants cannot enter model probes")
    if split not in {"train", "validation"}:
        raise ValueError("model-selection probes must not use the test split")
