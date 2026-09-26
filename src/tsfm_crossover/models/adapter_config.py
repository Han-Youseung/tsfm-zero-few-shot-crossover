"""Explicit engineering settings; none are pilot-selected defaults."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    family: Literal["ttm", "moirai1"]
    condition_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sampling_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    channel_names: tuple[str, ...] = Field(min_length=1)
    context: Literal[512] = 512
    horizon: Literal[96, 192, 336, 720]
    num_samples: int = Field(ge=1)
    prediction_seed: int = Field(ge=0)
    training_seed: int = Field(ge=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_optimizer_steps: int = Field(ge=1)
    gradient_clip: float | None = Field(default=None, gt=0)
    dtype: Literal["float32"] = "float32"
    point_statistic: Literal["prediction_outputs", "torch_sample_median"]
    external_scaler: Literal[False] = False
    protocol_frozen: bool = False

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel names must be unique and ordered")
        expected = "prediction_outputs" if self.family == "ttm" else "torch_sample_median"
        if self.point_statistic != expected or (self.family == "ttm" and self.num_samples != 1):
            raise ValueError("model point statistic/sample count mismatch")
        return self


def load_adapter_config(path: str | Path) -> AdapterConfig:
    return AdapterConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
