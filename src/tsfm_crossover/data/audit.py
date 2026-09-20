"""Stream a local dataset bundle and write metadata-only audit summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tsfm_crossover.tracking.atomic import write_csv_atomic, write_json_atomic

from .archive import file_sha256
from .common import stable_hash
from .coverage import compute_coverage
from .registry import DatasetSpec, load_registry
from .sampling import DEFAULT_RATES, selected_count, temporal_master_order
from .splits import chronological_split

SCHEMA_VERSION = "1.0"
VALIDATOR_VERSION = "streaming-long-csv-v1"


@dataclass
class ScanResult:
    raw_rows: int
    time_points: int
    channels: int
    columns: int
    numeric_columns: int
    non_numeric_columns: int
    encoding: str
    inferred_frequency_seconds: float | None
    timestamp_first: str | None
    timestamp_last: str | None
    timestamps: list[str]
    missing_values: int
    infinite_values: int
    parse_errors: int
    duplicate_timestamps: int
    non_monotonic_timestamps: int
    irregular_intervals: int
    duplicate_rows: int
    constant_channels: int
    all_missing_channels: int
    unequal_channel_lengths: bool
    float32_failures: int
    near_constant_channels: int
    scaler_hash: str
    train_mean_min: float | None
    train_mean_max: float | None
    train_std_min: float | None
    train_std_max: float | None
    max_float32_roundtrip_error: float
    issues: list[str]


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(normalized, pattern)
            except ValueError:
                pass
    raise ValueError(f"unparseable timestamp {value!r}")


def _channel_key(row: dict[str, str], spec: DatasetSpec) -> str:
    channel = row.get(spec.channel_column or "", "")
    if spec.series_id_column:
        return f"{row.get(spec.series_id_column, '')}::{channel}"
    return channel


def _welford_update(state: tuple[int, float, float], value: float) -> tuple[int, float, float]:
    count, mean, m2 = state
    count += 1
    delta = value - mean
    mean += delta / count
    return count, mean, m2 + delta * (value - mean)


def scan_long_csv(path: Path, spec: DatasetSpec) -> ScanResult:
    counts: Counter[str] = Counter()
    previous_time: dict[str, datetime] = {}
    previous_row: dict[str, tuple[str, str]] = {}
    deltas: Counter[float] = Counter()
    extrema: dict[str, list[float]] = {}
    missing = infinite = parse_errors = duplicates = non_monotonic = duplicate_rows = 0
    first_channel: str | None = None
    first_timestamps: list[str] = []
    raw_rows = 0
    required = {spec.timestamp_column, spec.value_column, spec.channel_column} - {None}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        absent = required - set(headers)
        if absent:
            raise ValueError(f"missing required columns: {sorted(absent)}")
        for row in reader:
            raw_rows += 1
            channel = _channel_key(row, spec)
            if first_channel is None:
                first_channel = channel
            counts[channel] += 1
            raw_value = (row.get(spec.value_column or "") or "").strip()
            try:
                value = float(raw_value)
                if math.isnan(value):
                    missing += 1
                elif math.isinf(value):
                    infinite += 1
                else:
                    limits = extrema.setdefault(channel, [value, value])
                    limits[0] = min(limits[0], value)
                    limits[1] = max(limits[1], value)
            except ValueError:
                parse_errors += 1
            raw_timestamp = row.get(spec.timestamp_column or "", "")
            try:
                timestamp = _parse_timestamp(raw_timestamp)
                if channel == first_channel:
                    first_timestamps.append(raw_timestamp)
                if channel in previous_time:
                    delta = (timestamp - previous_time[channel]).total_seconds()
                    deltas[delta] += 1
                    if delta == 0:
                        duplicates += 1
                    elif delta < 0:
                        non_monotonic += 1
                previous_time[channel] = timestamp
            except ValueError:
                parse_errors += 1
            row_identity = (raw_timestamp, raw_value)
            if previous_row.get(channel) == row_identity:
                duplicate_rows += 1
            previous_row[channel] = row_identity
    channel_lengths = set(counts.values())
    time_points = max(channel_lengths, default=0)
    positive_deltas = Counter({key: value for key, value in deltas.items() if key > 0})
    common_delta = positive_deltas.most_common(1)[0][0] if positive_deltas else None
    irregular = sum(value for key, value in positive_deltas.items() if key != common_delta)
    constant = sum(low == high for low, high in extrema.values())
    all_missing = len(counts) - len(extrema)
    train_end = math.floor(time_points * 0.6)
    positions: Counter[str] = Counter()
    stats: dict[str, tuple[int, float, float]] = defaultdict(lambda: (0, 0.0, 0.0))
    float32_failures = 0
    max_roundtrip_error = 0.0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            channel = _channel_key(row, spec)
            position = positions[channel]
            positions[channel] += 1
            if position >= train_end:
                continue
            try:
                value = float((row.get(spec.value_column or "") or "").strip())
                if not math.isfinite(value):
                    continue
                stats[channel] = _welford_update(stats[channel], value)
                converted = struct.unpack("f", struct.pack("f", value))[0]
                max_roundtrip_error = max(max_roundtrip_error, abs(converted - value))
            except (ValueError, OverflowError, struct.error):
                float32_failures += 1
    means: list[float] = []
    stds: list[float] = []
    scaler_payload = []
    for channel in sorted(counts):
        count, mean, m2 = stats[channel]
        if count:
            std = math.sqrt(m2 / count)
            means.append(mean)
            stds.append(std)
            scaler_payload.append((channel, count, mean, 1.0 if std == 0 else std))
    issues = []
    if missing:
        issues.append("missing_values")
    if infinite:
        issues.append("infinite_values")
    if parse_errors:
        issues.append("parse_errors")
    if duplicates:
        issues.append("duplicate_timestamps")
    if non_monotonic:
        issues.append("non_monotonic_timestamps")
    if irregular:
        issues.append("irregular_frequency")
    if len(channel_lengths) > 1:
        issues.append("unequal_channel_lengths")
    if constant:
        issues.append("constant_channels")
    near_constant = sum(0 < std < 1e-8 for std in stds)
    return ScanResult(
        raw_rows,
        time_points,
        len(counts),
        len(headers),
        1,
        len(headers) - 1,
        "utf-8-sig",
        common_delta,
        first_timestamps[0] if first_timestamps else None,
        first_timestamps[-1] if first_timestamps else None,
        first_timestamps,
        missing,
        infinite,
        parse_errors,
        duplicates,
        non_monotonic,
        irregular,
        duplicate_rows,
        constant,
        all_missing,
        len(channel_lengths) > 1,
        float32_failures,
        near_constant,
        stable_hash(scaler_payload),
        min(means, default=None),
        max(means, default=None),
        min(stds, default=None),
        max(stds, default=None),
        max_roundtrip_error,
        issues,
    )


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _inventory(
    data_root: Path, specs: list[DatasetSpec]
) -> tuple[list[dict[str, Any]], dict[str, list[Path]]]:
    aliases = {
        name.casefold(): spec.canonical_name
        for spec in specs
        for name in [spec.canonical_name, *spec.aliases]
    }
    mapping: dict[str, list[Path]] = defaultdict(list)
    inventory = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        stem = path.stem.casefold()
        canonical = aliases.get(stem)
        if canonical:
            mapping[canonical].append(path)
        inventory.append(
            {
                "relative_path": path.relative_to(data_root).as_posix(),
                "filename": path.name,
                "extension": path.suffix.casefold(),
                "byte_size": path.stat().st_size,
                "sha256": file_sha256(path),
                "inferred_candidate_dataset": path.stem if canonical else None,
                "canonical_dataset": canonical,
                "alias_applied": bool(canonical and path.stem != canonical),
                "validation_status": "candidate" if canonical else "not_in_study",
            }
        )
    return inventory, mapping


def _common(
    generated_at: str, code_sha: str, dataset_sha: str | None, config_hash: str, status: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "code_commit_sha": code_sha,
        "dataset_file_sha256": dataset_sha,
        "config_hash": config_hash,
        "validator_version": VALIDATOR_VERSION,
        "status": status,
    }


def run_audit(
    *,
    registry_path: Path,
    data_root: Path,
    output: Path,
    contexts: list[int],
    horizons: list[int],
    rates: list[float],
    seed: int,
    datasets: list[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    fail_on_warning: bool = False,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    specs = registry.datasets if not datasets else [registry.resolve(name) for name in datasets]
    registry_hash = file_sha256(registry_path)
    config_hash = stable_hash(
        {
            "registry_sha256": registry_hash,
            "contexts": contexts,
            "horizons": horizons,
            "rates": rates,
            "seed": seed,
            "datasets": [s.canonical_name for s in specs],
        }
    )
    generated_at = datetime.now(UTC).isoformat()
    code_sha = _git_sha()
    inventory, mapping = _inventory(data_root, specs)
    signature = stable_hash(
        {
            "config_hash": config_hash,
            "files": [(item["relative_path"], item["sha256"]) for item in inventory],
        }
    )
    run_path = output / "validation_run.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if existing.get("input_signature") == signature:
            return {"skipped": True, "reason": "identical_input_signature", **existing}
        if not overwrite:
            raise FileExistsError(f"audit output exists with different inputs: {output}")
    validation_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    states: dict[str, str] = {}
    for spec in specs:
        candidates = mapping.get(spec.canonical_name, [])
        if not candidates:
            status = "missing"
            states[spec.canonical_name] = status
            validation_rows.append(
                _common(generated_at, code_sha, None, config_hash, status)
                | {"dataset": spec.canonical_name, "issues": "file_missing"}
            )
            continue
        if len(candidates) > 1:
            status = "ambiguous"
            states[spec.canonical_name] = status
            validation_rows.append(
                _common(generated_at, code_sha, None, config_hash, status)
                | {"dataset": spec.canonical_name, "issues": "multiple_exact_filename_candidates"}
            )
            continue
        path = candidates[0]
        dataset_sha = next(
            item["sha256"]
            for item in inventory
            if item["relative_path"] == path.relative_to(data_root).as_posix()
        )
        try:
            if spec.layout != "long":
                raise ValueError("local bundle requires declared long layout")
            scan = scan_long_csv(path, spec)
            inventory_item = next(
                item
                for item in inventory
                if item["relative_path"] == path.relative_to(data_root).as_posix()
            )
            inventory_item.update(
                {
                    "row_count": scan.raw_rows,
                    "observed_time_points": scan.time_points,
                    "column_count": scan.columns,
                    "timestamp_candidate": spec.timestamp_column,
                    "numeric_column_count": scan.numeric_columns,
                    "non_numeric_column_count": scan.non_numeric_columns,
                    "encoding": scan.encoding,
                }
            )
            technical_errors = (
                scan.missing_values
                or scan.infinite_values
                or scan.parse_errors
                or scan.unequal_channel_lengths
            )
            issues = list(scan.issues)
            if spec.local_preprocessing_status != "verified_raw":
                issues.append("local_bundle_preprocessing_unverified")
            provenance_pending = spec.source.verification_status != "verified"
            if provenance_pending:
                issues.append("license_or_provenance_pending")
            if technical_errors or spec.local_preprocessing_status != "verified_raw":
                status = "blocked"
            elif provenance_pending or issues:
                status = "ready_with_warnings"
            else:
                status = "ready"
            states[spec.canonical_name] = status
            inventory_item["validation_status"] = status
            base = _common(generated_at, code_sha, dataset_sha, config_hash, status)
            validation_rows.append(
                base
                | {
                    "dataset": spec.canonical_name,
                    "raw_rows": scan.raw_rows,
                    "observed_rows": scan.time_points,
                    "observed_channels": scan.channels,
                    "observed_frequency_seconds": scan.inferred_frequency_seconds,
                    "columns": scan.columns,
                    "missing_values": scan.missing_values,
                    "infinite_values": scan.infinite_values,
                    "parse_errors": scan.parse_errors,
                    "duplicate_timestamps": scan.duplicate_timestamps,
                    "non_monotonic_timestamps": scan.non_monotonic_timestamps,
                    "irregular_intervals": scan.irregular_intervals,
                    "duplicate_rows": scan.duplicate_rows,
                    "constant_channels": scan.constant_channels,
                    "all_missing_channels": scan.all_missing_channels,
                    "estimated_float32_bytes": scan.time_points * scan.channels * 4,
                    "scaler_hash": scan.scaler_hash,
                    "train_mean_min": scan.train_mean_min,
                    "train_mean_max": scan.train_mean_max,
                    "train_std_min": scan.train_std_min,
                    "train_std_max": scan.train_std_max,
                    "near_constant_channels": scan.near_constant_channels,
                    "float32_failures": scan.float32_failures,
                    "max_float32_roundtrip_error": scan.max_float32_roundtrip_error,
                    "issues": ";".join(issues),
                }
            )
            split = chronological_split(scan.time_points)

            def boundary(index: int, timestamps: list[str] = scan.timestamps) -> str | None:
                return timestamps[index] if 0 <= index < len(timestamps) else None

            split_rows.append(
                base
                | {
                    "dataset": spec.canonical_name,
                    "total_rows": scan.time_points,
                    "train_start": split.train.start,
                    "train_end": split.train.end,
                    "train_rows": split.train.rows,
                    "validation_start": split.validation.start,
                    "validation_end": split.validation.end,
                    "validation_rows": split.validation.rows,
                    "test_start": split.test.start,
                    "test_end": split.test.end,
                    "test_rows": split.test.rows,
                    "train_start_timestamp": boundary(0),
                    "validation_start_timestamp": boundary(split.validation.start),
                    "test_start_timestamp": boundary(split.test.start),
                    "test_end_timestamp": boundary(scan.time_points - 1),
                    "gap": False,
                    "overlap": False,
                    "split_config_hash": split.split_config_hash,
                }
            )
            for context in contexts:
                for horizon in horizons:
                    train_count = max(0, split.train.rows - context - horizon + 1)
                    validation_count = (
                        max(0, split.validation.rows - horizon + 1)
                        if split.validation.start >= context
                        else 0
                    )
                    test_count = (
                        max(0, split.test.rows - horizon + 1) if split.test.start >= context else 0
                    )
                    feasible = train_count > 0 and validation_count > 0 and test_count > 0
                    warning = "few_test_windows" if 0 < test_count < 100 else ""
                    window_rows.append(
                        base
                        | {
                            "dataset": spec.canonical_name,
                            "context_length": context,
                            "horizon": horizon,
                            "stride": 1,
                            "train_windows": train_count,
                            "validation_windows": validation_count,
                            "test_windows": test_count,
                            "feasible": feasible,
                            "warning": warning,
                        }
                    )
                    if not feasible:
                        continue
                    order = temporal_master_order([None] * train_count, seed)
                    previous_count: int | None = None
                    for rate in rates:
                        count = selected_count(rate, train_count)
                        warnings = []
                        if count == 1:
                            warnings.append("single_window")
                        if previous_count == count and rate != 0:
                            warnings.append("same_count_as_previous_rate")
                        selected = order[:count]
                        windows = [
                            SimpleNamespace(
                                context_start=index,
                                context_end=index + context,
                                target_start=index + context,
                                target_end=index + context + horizon,
                            )
                            for index in selected
                        ]
                        coverage = compute_coverage(windows, (0, split.train.rows))
                        if count and coverage.total_coverage_ratio > (count / train_count) * 5:
                            warnings.append("raw_coverage_much_larger_than_window_rate")
                        if coverage.average_time_point_exposure_count > 10:
                            warnings.append("high_window_overlap")
                        sampling_rows.append(
                            base
                            | {
                                "dataset": spec.canonical_name,
                                "context_length": context,
                                "horizon": horizon,
                                "sampling_seed": seed,
                                "requested_rate": rate,
                                "selected_train_windows": count,
                                "effective_rate": count / train_count,
                                "total_train_windows": train_count,
                                "nestedness_valid": True,
                                "warnings": ";".join(warnings),
                            }
                        )
                        coverage_rows.append(
                            base
                            | {
                                "dataset": spec.canonical_name,
                                "context_length": context,
                                "horizon": horizon,
                                "sampling_seed": seed,
                                "requested_rate": rate,
                                **coverage.as_dict(),
                            }
                        )
                        previous_count = count
        except Exception as exc:
            states[spec.canonical_name] = "blocked"
            validation_rows.append(
                _common(generated_at, code_sha, dataset_sha, config_hash, "blocked")
                | {"dataset": spec.canonical_name, "issues": f"{type(exc).__name__}: {exc}"}
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "code_commit_sha": code_sha,
        "config_hash": config_hash,
        "validator_version": VALIDATOR_VERSION,
        "input_signature": signature,
        "status": "audit_completed",
        "dataset_states": states,
        "dry_run": dry_run,
    }
    if dry_run:
        return result | {"inventory_count": len(inventory)}
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output / "inventory.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "code_commit_sha": code_sha,
            "config_hash": config_hash,
            "files": inventory,
        },
        overwrite=overwrite,
    )
    tables = {
        "validation_summary.csv": validation_rows,
        "split_summary.csv": split_rows,
        "window_feasibility.csv": window_rows,
        "sampling_count_summary.csv": sampling_rows,
        "coverage_summary.csv": coverage_rows,
    }
    provenance_rows = [
        _common(
            generated_at,
            code_sha,
            next(
                (
                    item["sha256"]
                    for item in inventory
                    if item["canonical_dataset"] == spec.canonical_name
                ),
                None,
            ),
            config_hash,
            states.get(spec.canonical_name, "missing"),
        )
        | {
            "dataset": spec.canonical_name,
            "original_source": spec.source.source_url,
            "download_source": spec.source.download_url,
            "local_bundle_source": "data/incoming/TSFM_dataset.zip (user supplied)",
            "original_license": spec.source.license,
            "redistribution_permission": spec.source.redistribution,
            "expected_rows": spec.expected_rows,
            "expected_channels": spec.expected_channels,
            "expected_frequency": spec.frequency,
            "official_rows": spec.official_rows,
            "official_channels": spec.official_channels,
            "official_frequency": spec.official_frequency,
            "citation": None,
            "verification_status": spec.source.verification_status,
            "checked_date": generated_at[:10],
        }
        for spec in specs
    ]
    tables["provenance_summary.csv"] = provenance_rows
    for filename, rows in tables.items():
        if rows:
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            write_csv_atomic(output / filename, rows, fieldnames=fieldnames, overwrite=overwrite)
    write_json_atomic(run_path, result, overwrite=overwrite)
    if fail_on_warning and any(state != "ready" for state in states.values()):
        raise RuntimeError("audit produced warnings or blocked datasets")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/datasets/registry.yaml")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--sampling-rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--output", default="results/manifests/datasets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        registry_path=Path(args.registry),
        data_root=Path(args.data_root),
        output=Path(args.output),
        contexts=args.context_lengths,
        horizons=args.horizons,
        rates=args.sampling_rates,
        seed=args.sampling_seed,
        datasets=args.datasets,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        fail_on_warning=args.fail_on_warning,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
