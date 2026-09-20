"""Leakage-safe chronological 60:20:20 splits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime

from .common import stable_hash


@dataclass(frozen=True)
class Interval:
    start: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SplitManifest:
    total_rows: int
    train: Interval
    validation: Interval
    test: Interval
    ratios: tuple[float, float, float]
    boundary_timestamps: dict[str, str | None]
    split_config_hash: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def chronological_split(
    total_rows: int,
    timestamps: tuple[datetime, ...] | None = None,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> SplitManifest:
    if total_rows < 1 or abs(sum(ratios) - 1.0) > 1e-12 or any(r < 0 for r in ratios):
        raise ValueError("need positive rows and non-negative ratios summing to one")
    train_end = math.floor(total_rows * ratios[0])
    validation_end = math.floor(total_rows * (ratios[0] + ratios[1]))
    intervals = (
        Interval(0, train_end),
        Interval(train_end, validation_end),
        Interval(validation_end, total_rows),
    )
    boundaries = {
        "train_start": None,
        "validation_start": None,
        "test_start": None,
        "test_end": None,
    }
    if timestamps:
        boundaries = {
            "train_start": timestamps[0].isoformat(),
            "validation_start": timestamps[train_end].isoformat()
            if train_end < total_rows
            else None,
            "test_start": timestamps[validation_end].isoformat()
            if validation_end < total_rows
            else None,
            "test_end": timestamps[-1].isoformat(),
        }
    config_hash = stable_hash(
        {"method": "chronological_ratio_floor", "ratios": ratios, "total_rows": total_rows}
    )
    return SplitManifest(total_rows, *intervals, ratios, boundaries, config_hash)
