"""Framework-neutral contract for future production model adapters.

This module deliberately imports neither torch nor either vendor package.  It lets
the default test suite verify safety invariants without downloading model weights.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CandidateState(StrEnum):
    candidate = "candidate"
    import_validated = "import_validated"
    inference_validated = "inference_validated"
    finetune_validated = "finetune_validated"
    pilot_validated = "pilot_validated"
    frozen = "frozen"


class ProbeStatus(StrEnum):
    not_tested = "not_tested"
    import_failed = "import_failed"
    load_failed = "load_failed"
    inference_failed = "inference_failed"
    finetune_failed = "finetune_failed"
    cpu_validated = "cpu_validated"
    pending_colab = "pending_colab"
    gpu_validated = "gpu_validated"


class CapabilityReport(StrictModel):
    zero_shot: bool
    full_parameter_finetuning: bool | None
    multivariate: bool | None
    missing_values: bool | None
    context_lengths: list[int] = Field(min_length=1)
    prediction_lengths: list[int] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class ModelMetadata(StrictModel):
    family: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=7)
    official_code_repository: str = Field(min_length=1)
    code_commit: str = Field(min_length=7)
    state: CandidateState
    point_forecast_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def prohibit_unvalidated_freeze(self) -> ModelMetadata:
        if self.state == CandidateState.frozen:
            raise ValueError("compatibility-stage candidates cannot be marked frozen")
        return self


class AdapterContract(ABC):
    """Required surface for later, model-specific production adapters."""

    optimizer_created: bool = False
    backward_calls: int = 0

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata: ...

    @abstractmethod
    def validate_capabilities(self) -> CapabilityReport: ...

    @abstractmethod
    def prepare_batch(self, batch: Any) -> Any: ...

    @abstractmethod
    def predict(self, batch: Any) -> Any: ...

    @abstractmethod
    def point_forecast(self, prediction: Any) -> Any: ...

    @abstractmethod
    def configure_finetuning(self) -> None: ...

    @abstractmethod
    def train_step(self, batch: Any) -> float: ...

    @abstractmethod
    def validation_step(self, batch: Any) -> float: ...

    @abstractmethod
    def save_training_state(self, path: Path) -> None: ...

    @abstractmethod
    def load_training_state(self, path: Path) -> None: ...

    @abstractmethod
    def count_parameters(self) -> tuple[int, int]: ...

    @abstractmethod
    def parameter_state(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def set_evaluation_mode(self) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    def zero_shot_predict(self, batch: Any) -> Any:
        """Run inference while enforcing the no-update Zero-Shot definition."""
        before = parameter_hash(self.parameter_state())
        optimizer_before = self.optimizer_created
        backward_before = self.backward_calls
        self.set_evaluation_mode()
        prediction = self.predict(self.prepare_batch(batch))
        after = parameter_hash(self.parameter_state())
        if optimizer_before or self.optimizer_created:
            raise RuntimeError("Zero-Shot must not create an optimizer")
        if self.backward_calls != backward_before:
            raise RuntimeError("Zero-Shot must not call backward")
        if before != after:
            raise RuntimeError("Zero-Shot changed model parameters")
        return prediction


def _stable_bytes(value: Any) -> bytes:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return repr(value).encode("utf-8")


def parameter_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_stable_bytes(state[name]))
        digest.update(b"\0")
    return digest.hexdigest()


def _flatten(values: Any) -> Iterable[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError("prediction must be numeric")
    if hasattr(values, "detach") and hasattr(values, "cpu") and hasattr(values, "reshape"):
        yield from values.detach().cpu().reshape(-1).tolist()
        return
    if isinstance(values, Iterable):
        for value in values:
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                yield from _flatten(value)
            else:
                yield float(value)
        return
    yield float(values)


def validate_prediction(
    prediction: Any,
    *,
    shape: tuple[int, ...],
    expected_batch: int,
    expected_horizon: int,
    expected_channels: int,
) -> None:
    if len(shape) != 3:
        raise ValueError("point prediction must have shape [batch, horizon, channel]")
    expected = (expected_batch, expected_horizon, expected_channels)
    if shape != expected:
        raise ValueError(f"prediction shape mismatch: expected {expected}, got {shape}")
    if not all(math.isfinite(value) for value in _flatten(prediction)):
        raise ValueError("prediction contains NaN or infinity")


def require_smoke_split(split: str, *, training: bool) -> None:
    allowed = {"train", "validation"} if training else {"validation"}
    if split not in allowed:
        raise ValueError(f"compatibility smoke split must be one of {sorted(allowed)}")


def require_official_raw_variant(source_variant: str, grade: str, status: str) -> None:
    if not source_variant.endswith("__official_raw"):
        raise ValueError("compatibility probes reject bundle and non-official variants")
    if grade != "A" or status != "ready_with_warnings":
        raise ValueError("compatibility probes require grade A ready_with_warnings data")
