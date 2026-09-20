"""Index-only train and rolling-origin window construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .common import stable_hash
from .splits import Interval


@dataclass(frozen=True)
class WindowIndex:
    dataset: str
    split: str
    context_start: int
    context_end: int
    target_start: int
    target_end: int
    context_length: int
    horizon: int
    origin_index: int
    origin_timestamp: str | None
    window_id: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _window(
    dataset: str,
    fingerprint: str,
    split: str,
    target_start: int,
    context_length: int,
    horizon: int,
    origin_timestamp: str | None = None,
) -> WindowIndex:
    context_start = target_start - context_length
    payload = {
        "dataset_fingerprint": fingerprint,
        "split": split,
        "context_length": context_length,
        "horizon": horizon,
        "context_start": context_start,
        "target_start": target_start,
    }
    return WindowIndex(
        dataset,
        split,
        context_start,
        target_start,
        target_start,
        target_start + horizon,
        context_length,
        horizon,
        target_start,
        origin_timestamp,
        stable_hash(payload),
    )


def generate_train_windows(
    dataset: str,
    fingerprint: str,
    train: Interval,
    context_length: int,
    horizon: int,
    stride: int = 1,
) -> list[WindowIndex]:
    _validate_lengths(context_length, horizon, stride)
    first = train.start + context_length
    last = train.end - horizon
    if first > last:
        raise ValueError("no train windows: context_length + horizon exceeds train split")
    return [
        _window(dataset, fingerprint, "train", start, context_length, horizon)
        for start in range(first, last + 1, stride)
    ]


def generate_rolling_windows(
    dataset: str,
    fingerprint: str,
    split_name: str,
    split: Interval,
    context_length: int,
    horizon: int,
    stride: int = 1,
    timestamps: tuple[object, ...] | None = None,
) -> list[WindowIndex]:
    _validate_lengths(context_length, horizon, stride)
    if split_name not in {"validation", "test"}:
        raise ValueError("rolling split must be validation or test")
    first = split.start
    last = split.end - horizon
    if first - context_length < 0 or first > last:
        raise ValueError("no rolling windows for requested context and horizon")
    result = []
    for start in range(first, last + 1, stride):
        stamp = str(timestamps[start]) if timestamps is not None else None
        result.append(
            _window(dataset, fingerprint, split_name, start, context_length, horizon, stamp)
        )
    return result


def _validate_lengths(context_length: int, horizon: int, stride: int) -> None:
    if context_length < 1 or horizon < 1 or stride < 1:
        raise ValueError("context_length, horizon, and stride must be positive")
