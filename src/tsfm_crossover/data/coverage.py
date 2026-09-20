"""Raw-time-point coverage using interval sweeps, never a window-by-time matrix."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .windows import WindowIndex


@dataclass(frozen=True)
class CoverageStats:
    unique_context_time_points: int
    unique_target_time_points: int
    unique_total_observed_time_points: int
    context_coverage_ratio: float
    target_coverage_ratio: float
    total_coverage_ratio: float
    average_time_point_exposure_count: float
    maximum_time_point_exposure_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sweep(intervals: Iterable[tuple[int, int]]) -> tuple[int, int, int]:
    events: dict[int, int] = {}
    incidences = 0
    for start, end in intervals:
        if end <= start:
            continue
        events[start] = events.get(start, 0) + 1
        events[end] = events.get(end, 0) - 1
        incidences += end - start
    active = unique = maximum = 0
    previous: int | None = None
    for position in sorted(events):
        if previous is not None and active > 0:
            unique += position - previous
        active += events[position]
        maximum = max(maximum, active)
        previous = position
    return unique, incidences, maximum


def compute_coverage(windows: Iterable[WindowIndex], universe: tuple[int, int]) -> CoverageStats:
    selected = list(windows)
    denominator = universe[1] - universe[0]
    if denominator <= 0:
        raise ValueError("coverage universe must be non-empty")
    context, _, _ = _sweep((w.context_start, w.context_end) for w in selected)
    target, _, _ = _sweep((w.target_start, w.target_end) for w in selected)
    total, incidences, maximum = _sweep((w.context_start, w.target_end) for w in selected)
    return CoverageStats(
        context,
        target,
        total,
        context / denominator,
        target / denominator,
        total / denominator,
        incidences / total if total else 0.0,
        maximum,
    )
