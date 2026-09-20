"""Channel-wise population StandardScaler fitted only on an explicit train interval."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .common import stable_hash
from .splits import Interval


@dataclass(frozen=True)
class TrainOnlyStandardScaler:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    constant_channels: tuple[int, ...]
    fit_interval: Interval
    fit_data_hash: str

    @classmethod
    def fit(cls, values: tuple[tuple[float, ...], ...], train: Interval) -> TrainOnlyStandardScaler:
        fit_rows = values[train.start : train.end]
        if not fit_rows:
            raise ValueError("train interval contains no rows")
        width = len(fit_rows[0])
        if not width or any(len(row) != width for row in fit_rows):
            raise ValueError("values must be a non-empty rectangular [time, channel] array")
        means = tuple(sum(row[c] for row in fit_rows) / len(fit_rows) for c in range(width))
        stds = tuple(
            math.sqrt(sum((row[c] - means[c]) ** 2 for row in fit_rows) / len(fit_rows))
            for c in range(width)
        )
        constants = tuple(index for index, std in enumerate(stds) if std == 0)
        scales = tuple(1.0 if std == 0 else std for std in stds)
        return cls(means, scales, constants, train, stable_hash(fit_rows))

    def transform(self, values: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple((value - self.means[c]) / self.scales[c] for c, value in enumerate(row))
            for row in values
        )

    def inverse_transform(
        self, values: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(value * self.scales[c] + self.means[c] for c, value in enumerate(row))
            for row in values
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "means": self.means,
            "scales": self.scales,
            "constant_channels": self.constant_channels,
            "fit_start": self.fit_interval.start,
            "fit_end": self.fit_interval.end,
            "fit_data_hash": self.fit_data_hash,
            "formula": "population_standard_deviation_ddof_0",
        }
