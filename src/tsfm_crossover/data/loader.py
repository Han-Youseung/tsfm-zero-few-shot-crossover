"""Model-independent CSV loader with explicit validation."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .registry import DatasetSpec
from .validation import ValidationReport


@dataclass(frozen=True)
class LoadedTimeSeries:
    canonical_name: str
    timestamps: tuple[datetime, ...] | None
    positional_index: tuple[int, ...] | None
    values: tuple[tuple[float, ...], ...]
    channel_names: tuple[str, ...]
    inferred_frequency_seconds: float | None
    source_path: str
    source_sha256: str
    validation_report: ValidationReport

    @property
    def row_count(self) -> int:
        return len(self.values)

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _frequency_seconds(value: str) -> float | None:
    normalized = value.strip().casefold()
    suffixes = {"min": 60.0, "h": 3600.0, "d": 86400.0, "s": 1.0}
    for suffix, multiplier in suffixes.items():
        if normalized.endswith(suffix):
            try:
                return float(normalized[: -len(suffix)]) * multiplier
            except ValueError:
                return None
    return None


def load_dataset(spec: DatasetSpec, root: str | Path = ".") -> LoadedTimeSeries:
    path = Path(root) / spec.relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset file not found: {path}. Place {spec.source.local_filename} at this path; "
            "automatic download is disabled."
        )
    report = ValidationReport()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows:
        report.add("empty_dataset", "error", "dataset contains no rows")

    if spec.target_columns:
        channels = list(spec.target_columns)
    else:
        excluded = set(spec.excluded_columns)
        if spec.timestamp_column:
            excluded.add(spec.timestamp_column)
        channels = [name for name in fields if name not in excluded]
    missing_columns = [name for name in channels if name not in fields]
    for name in missing_columns:
        report.add("missing_target_column", "error", f"missing target column {name!r}", column=name)
    channels = [name for name in channels if name in fields]

    values: list[tuple[float, ...]] = []
    for row_number, row in enumerate(rows, start=2):
        parsed: list[float] = []
        for name in channels:
            raw = (row.get(name) or "").strip()
            try:
                value = float(raw)
            except ValueError:
                report.add(
                    "non_numeric_target",
                    "error",
                    f"cannot parse {raw!r}",
                    column=name,
                    row=row_number,
                )
                value = math.nan
            if math.isnan(value):
                report.add(
                    "missing_value", "error", "NaN or empty value", column=name, row=row_number
                )
            elif math.isinf(value):
                report.add("infinite_value", "error", "infinite value", column=name, row=row_number)
            parsed.append(value)
        values.append(tuple(parsed))

    for index, name in enumerate(channels):
        finite = [row[index] for row in values if math.isfinite(row[index])]
        if not finite:
            report.add("all_missing_channel", "error", "channel has no finite values", column=name)
        elif len(set(finite)) == 1:
            report.add("constant_channel", "warning", "channel is constant", column=name)

    timestamps: list[datetime] | None = None
    inferred: float | None = None
    if spec.timestamp_column:
        if spec.timestamp_column not in fields:
            severity = "error" if spec.timestamp_required else "warning"
            report.add("missing_timestamp_column", severity, "timestamp column is absent")
        else:
            timestamps = []
            for row_number, row in enumerate(rows, start=2):
                try:
                    timestamps.append(_parse_timestamp(row[spec.timestamp_column]))
                except (ValueError, TypeError):
                    report.add(
                        "invalid_timestamp", "error", "timestamp cannot be parsed", row=row_number
                    )
            if len(timestamps) == len(rows) and len(timestamps) > 1:
                seconds = [
                    (b - a).total_seconds()
                    for a, b in zip(timestamps, timestamps[1:], strict=False)
                ]
                if any(delta == 0 for delta in seconds):
                    report.add("duplicate_timestamp", "error", "duplicate timestamps found")
                if any(delta < 0 for delta in seconds):
                    report.add("non_monotonic_timestamp", "error", "timestamps are not increasing")
                positive = [delta for delta in seconds if delta > 0]
                if positive:
                    inferred = positive[0]
                    if any(delta != inferred for delta in positive):
                        report.add(
                            "irregular_frequency", "warning", "timestamp intervals are irregular"
                        )
                    expected_seconds = (
                        _frequency_seconds(spec.frequency) if spec.frequency else None
                    )
                    if expected_seconds is not None and inferred != expected_seconds:
                        report.add(
                            "unexpected_frequency",
                            "warning",
                            f"expected {spec.frequency}, inferred {inferred} seconds",
                        )

    if spec.expected_rows is not None and len(rows) != spec.expected_rows:
        report.add("unexpected_rows", "warning", f"expected {spec.expected_rows}, got {len(rows)}")
    if spec.expected_channels is not None and len(channels) != spec.expected_channels:
        report.add(
            "unexpected_channels",
            "warning",
            f"expected {spec.expected_channels}, got {len(channels)}",
        )
    source_sha256 = _sha256(path)
    if spec.source.file_sha256 and source_sha256 != spec.source.file_sha256:
        report.add("source_sha256_mismatch", "error", "file SHA256 differs from registry")
    return LoadedTimeSeries(
        canonical_name=spec.canonical_name,
        timestamps=tuple(timestamps) if timestamps is not None else None,
        positional_index=None if timestamps is not None else tuple(range(len(rows))),
        values=tuple(values),
        channel_names=tuple(channels),
        inferred_frequency_seconds=inferred,
        source_path=str(path.resolve()),
        source_sha256=source_sha256,
        validation_report=report,
    )
