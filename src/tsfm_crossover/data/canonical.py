"""Independent, aggregation-free canonicalization for long time-series records."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from .common import stable_hash


class CanonicalizationError(ValueError):
    """Raised when a long table cannot be reshaped without changing information."""


@dataclass(frozen=True)
class CanonicalSeries:
    timestamps: tuple[str, ...]
    channel_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    missing_mask: tuple[tuple[bool, ...], ...]
    source_file_sha256: str | None
    canonicalization_config_hash: str
    canonical_data_fingerprint: str
    validation_report: dict[str, object]


def _timestamp_sort_key(value: str) -> tuple[int, object]:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return (0, datetime.fromisoformat(normalized))
    except ValueError:
        for pattern in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"):
            try:
                return (0, datetime.strptime(normalized, pattern))
            except ValueError:
                pass
    return (1, value)


def _put_text(hasher: object, value: str) -> None:
    payload = value.encode("utf-8")
    hasher.update(struct.pack(">Q", len(payload)))
    hasher.update(payload)


def canonical_fingerprint(
    timestamps: Sequence[str],
    channels: Sequence[str],
    values: Sequence[Sequence[float]],
    missing_mask: Sequence[Sequence[bool]] | None = None,
) -> str:
    """Hash a canonical matrix using UTF-8 length prefixes and big-endian float64."""
    if len(values) != len(timestamps):
        raise ValueError("one value row is required per timestamp")
    mask = missing_mask or [[math.isnan(value) for value in row] for row in values]
    if len(mask) != len(timestamps):
        raise ValueError("missing mask shape does not match timestamps")
    timestamp_digest = hashlib.sha256()
    for timestamp in timestamps:
        _put_text(timestamp_digest, timestamp)
    channel_digests = [hashlib.sha256() for _ in channels]
    for row, mask_row in zip(values, mask, strict=True):
        if len(row) != len(channels) or len(mask_row) != len(channels):
            raise ValueError("canonical matrix is not rectangular")
        for index, (value, is_missing) in enumerate(zip(row, mask_row, strict=True)):
            channel_digests[index].update(b"\x01" if is_missing else b"\x00")
            channel_digests[index].update(
                struct.pack(">d", float("nan") if is_missing else float(value))
            )
    digest = hashlib.sha256()
    digest.update(b"tsfm-canonical-v1\0float64-be\0component-digests\0")
    digest.update(struct.pack(">QQ", len(timestamps), len(channels)))
    digest.update(timestamp_digest.digest())
    for channel, channel_digest in zip(channels, channel_digests, strict=True):
        _put_text(digest, channel)
        digest.update(channel_digest.digest())
    return digest.hexdigest()


def fingerprint_from_component_digests(
    *,
    timestamp_count: int,
    timestamp_digest: bytes,
    channels: Sequence[str],
    channel_value_digests: Sequence[bytes],
) -> str:
    """Finalize the documented fingerprint from independently streamed components."""
    if len(channels) != len(channel_value_digests):
        raise ValueError("one value digest is required per channel")
    digest = hashlib.sha256()
    digest.update(b"tsfm-canonical-v1\0float64-be\0component-digests\0")
    digest.update(struct.pack(">QQ", timestamp_count, len(channels)))
    digest.update(timestamp_digest)
    for channel, value_digest in zip(channels, channel_value_digests, strict=True):
        _put_text(digest, channel)
        digest.update(value_digest)
    return digest.hexdigest()


def canonicalize_long_records(
    records: Iterable[Mapping[str, object]],
    *,
    timestamp_column: str = "date",
    value_column: str = "data",
    channel_column: str = "cols",
    channel_order: Sequence[str] | None = None,
    channel_order_source: str = "first_appearance",
    source_file_sha256: str | None = None,
) -> CanonicalSeries:
    """Convert long records to a sorted time-by-channel matrix without aggregation."""
    cells: dict[tuple[str, str], float] = {}
    first_channels: list[str] = []
    seen_channels: set[str] = set()
    value_hashes: list[bytes] = []
    input_count = 0
    for row in records:
        input_count += 1
        timestamp = str(row[timestamp_column])
        channel = str(row[channel_column])
        value = float(row[value_column])
        key = (timestamp, channel)
        if key in cells:
            kind = "identical" if cells[key] == value else "conflicting"
            raise CanonicalizationError(f"{kind} duplicate composite key: {key!r}")
        cells[key] = value
        if channel not in seen_channels:
            seen_channels.add(channel)
            first_channels.append(channel)
        value_hashes.append(struct.pack(">d", value))
    if not cells:
        raise CanonicalizationError("no records")
    channels = tuple(channel_order or first_channels)
    if set(channels) != seen_channels or len(channels) != len(seen_channels):
        raise CanonicalizationError("channel order does not match observed channels")
    timestamps = tuple(sorted({timestamp for timestamp, _ in cells}, key=_timestamp_sort_key))
    expected = len(timestamps) * len(channels)
    if len(cells) != expected:
        raise CanonicalizationError("missing channel-time combination")
    rows: list[tuple[float, ...]] = []
    for timestamp in timestamps:
        try:
            rows.append(tuple(cells[(timestamp, channel)] for channel in channels))
        except KeyError as exc:
            raise CanonicalizationError("channel timestamp sets differ") from exc
    output_hashes = [struct.pack(">d", value) for row in rows for value in row]
    if sorted(value_hashes) != sorted(output_hashes) or input_count != expected:
        raise CanonicalizationError("value multiset or element count changed")
    missing = tuple(tuple(math.isnan(value) for value in row) for row in rows)
    config = {
        "schema": "aggregation-free-long-to-wide-v1",
        "timestamp_column": timestamp_column,
        "value_column": value_column,
        "channel_column": channel_column,
        "timestamp_order": "parsed_datetime_ascending_then_lexicographic_fallback",
        "channel_order": list(channels),
        "channel_order_source": channel_order_source,
        "dtype_serialization": "IEEE-754-float64-big-endian",
    }
    config_hash = stable_hash(config)
    return CanonicalSeries(
        timestamps,
        channels,
        tuple(rows),
        missing,
        source_file_sha256,
        config_hash,
        canonical_fingerprint(timestamps, channels, rows, missing),
        {
            "input_elements": input_count,
            "output_elements": expected,
            "aggregation_performed": False,
            "rectangular": True,
            "channel_order_source": channel_order_source,
            "value_multiset_preserved": True,
        },
    )
