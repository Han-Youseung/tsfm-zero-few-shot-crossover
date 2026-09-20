"""Stable human-readable identifiers backed by canonical configuration hashes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import ExperimentConfig, FreezeState


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unnamed"


def build_protocol_id(name: str, protocol_conditions: Any) -> str:
    return f"{_slug(name).replace('-', '_')}__{stable_hash(protocol_conditions, 8)}"


def protocol_conditions(config: ExperimentConfig) -> dict[str, Any]:
    """Extract only research conditions that define a protocol."""
    return {
        "split": config.split.model_dump(mode="json"),
        "sampling_strategy": config.sampling.strategy,
        "fine_tuning_budget": config.fine_tuning.budget_type,
        "evaluation_strategy": config.evaluation.strategy,
        "rolling_stride": config.evaluation.rolling_stride,
        "normalization": config.evaluation.normalization,
        "point_forecast_rule": config.evaluation.point_forecast_rule,
    }


def build_protocol_id_from_config(config: ExperimentConfig) -> str:
    return build_protocol_id(config.protocol_metadata.name, protocol_conditions(config))


class ExperimentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_id: str
    model_family: str
    model_repository: str
    model_revision: str | None = None
    model_state: FreezeState
    dataset: str
    context_length: int = Field(gt=0)
    horizon: int = Field(gt=0)
    sampling_rate: float = Field(ge=0, le=1)
    selected_train_windows: int = Field(ge=0)
    sampling_seed: int
    training_seed: int
    run_attempt: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_revision(self) -> ExperimentIdentity:
        if self.model_state == FreezeState.frozen and not self.model_revision:
            raise ValueError("frozen experiment identity requires model_revision")
        return self

    def base_payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json", exclude={"run_attempt"})
        data["revision_status"] = self.model_state.value
        if self.model_revision is None:
            data["model_revision"] = "explicitly-provisional"
        return data


def build_experiment_id(identity: ExperimentIdentity) -> tuple[str, str]:
    base_hash = stable_hash(identity.base_payload(), 12)
    readable = "-".join(
        [
            _slug(identity.model_family),
            _slug(identity.dataset),
            f"c{identity.context_length}",
            f"h{identity.horizon}",
        ]
    )
    base_id = f"{readable}__{base_hash}"
    return base_id, f"{base_id}__attempt-{identity.run_attempt:02d}"
