"""Strict, exact-name dataset registry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    official_name: str
    source_url: str | None = None
    author_or_provider: str | None = None
    download_url: str | None = None
    license: str | None = None
    redistribution: str | None = None
    verification_status: Literal["verified", "provisional", "pending"] = "pending"
    verified_on: str | None = None
    file_sha256: str | None = None
    local_filename: str


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    relative_path: str
    file_format: Literal["csv"] = "csv"
    timestamp_column: str | None = None
    timestamp_required: bool | None = None
    frequency: str | None = None
    expected_rows: int | None = None
    expected_channels: int | None = None
    target_columns: list[str] | None = None
    all_numeric_targets: bool = False
    excluded_columns: list[str] = Field(default_factory=list)
    missing_value_policy: Literal["error"] = "error"
    duplicate_timestamp_policy: Literal["error"] = "error"
    sort_policy: Literal["error"] = "error"
    dtype: Literal["float32"] = "float32"
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
    source: SourceSpec

    @model_validator(mode="after")
    def validate_targets_and_ratios(self) -> DatasetSpec:
        if bool(self.target_columns) == self.all_numeric_targets:
            raise ValueError("set exactly one of target_columns or all_numeric_targets")
        if abs(sum(self.split_ratios) - 1.0) > 1e-12:
            raise ValueError("split_ratios must sum to 1")
        return self


class DatasetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    datasets: list[DatasetSpec]

    @model_validator(mode="after")
    def validate_names(self) -> DatasetRegistry:
        names = [name for item in self.datasets for name in [item.canonical_name, *item.aliases]]
        folded = [name.casefold() for name in names]
        if len(folded) != len(set(folded)):
            raise ValueError("canonical names and aliases must be unique")
        return self

    def resolve(self, name: str) -> DatasetSpec:
        needle = name.casefold()
        for spec in self.datasets:
            if needle in {spec.canonical_name.casefold(), *(a.casefold() for a in spec.aliases)}:
                return spec
        raise KeyError(f"unknown dataset {name!r}; fuzzy matching is disabled")


def load_registry(path: str | Path) -> DatasetRegistry:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    return DatasetRegistry.model_validate(payload)
