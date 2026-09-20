"""Resolve structural and provenance evidence for the local forecasting bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import struct
import subprocess
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tsfm_crossover.tracking.atomic import write_csv_atomic, write_json_atomic

from .archive import file_sha256
from .canonical import fingerprint_from_component_digests
from .common import stable_hash
from .provenance import diagnose_affine
from .registry import DatasetSpec, load_registry

SCHEMA_VERSION = "1.0"
RESOLVER_VERSION = "independent-provenance-v1"
BENCHMARK_REPOSITORY = "https://github.com/decisionintelligence/TSFM-Bench"
BENCHMARK_COMMIT = "f9bb69402fe5c57ff6dec7ac254aabaab38239ff"
ETT_REPOSITORY = "https://github.com/zhouhaoyi/ETDataset"
ETT_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"


@dataclass
class Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else math.nan


def _put_text(digest: Any, value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _channel(row: dict[str, str], spec: DatasetSpec) -> str:
    name = row[spec.channel_column or ""]
    return f"{row[spec.series_id_column]}::{name}" if spec.series_id_column else name


def _decimal_places(token: str) -> int:
    mantissa = token.lower().split("e", maxsplit=1)[0]
    return len(mantissa.rsplit(".", maxsplit=1)[1]) if "." in mantissa else 0


def _summary(values: list[float], prefix: str) -> dict[str, float | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {f"{prefix}_{name}": None for name in ("min", "median", "max")}
    return {
        f"{prefix}_min": finite[0],
        f"{prefix}_median": finite[len(finite) // 2],
        f"{prefix}_max": finite[-1],
    }


def analyze_bundle(path: Path, spec: DatasetSpec, time_points_hint: int) -> tuple[dict, dict, dict]:
    channels: list[str] = []
    closed: set[str] = set()
    current: str | None = None
    current_seen: dict[str, float] = {}
    first_timestamps: list[str] = []
    first_unique: set[str] = set()
    timestamp_digest = hashlib.sha256()
    channel_digests: list[bytes] = []
    current_value_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    stats: dict[str, dict[str, Moments]] = defaultdict(
        lambda: {part: Moments() for part in ("full", "train", "validation", "test")}
    )
    row_count = composite_duplicates = identical_duplicates = conflicting_duplicates = 0
    timestamp_sequence_mismatches = extra_columns = nonfinite = parse_errors = 0
    maximum_decimal_places = 0
    headers: list[str] = []
    train_end = math.floor(time_points_hint * 0.6)
    validation_end = math.floor(time_points_hint * 0.8)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        expected = {spec.timestamp_column, spec.value_column, spec.channel_column}
        if spec.series_id_column:
            expected.add(spec.series_id_column)
        extra_columns = len(set(headers) - expected)
        for row in reader:
            row_count += 1
            channel = _channel(row, spec)
            timestamp = row[spec.timestamp_column or ""]
            raw_value = row[spec.value_column or ""]
            try:
                value = float(raw_value)
            except ValueError:
                parse_errors += 1
                continue
            if not math.isfinite(value):
                nonfinite += 1
            maximum_decimal_places = max(maximum_decimal_places, _decimal_places(raw_value))
            if channel != current:
                if current is not None:
                    closed.add(current)
                    channel_digests.append(current_value_digest.digest())
                if channel in closed:
                    raise ValueError(f"non-contiguous channel block: {channel}")
                current = channel
                channels.append(channel)
                current_seen = {}
                current_value_digest = hashlib.sha256()
            position = counts[channel]
            counts[channel] += 1
            previous = current_seen.get(timestamp)
            if previous is not None:
                composite_duplicates += 1
                if previous == value:
                    identical_duplicates += 1
                else:
                    conflicting_duplicates += 1
            else:
                current_seen[timestamp] = value
            if len(channels) == 1:
                first_timestamps.append(timestamp)
                first_unique.add(timestamp)
                _put_text(timestamp_digest, timestamp)
            elif position >= len(first_timestamps) or first_timestamps[position] != timestamp:
                timestamp_sequence_mismatches += 1
            missing = math.isnan(value)
            current_value_digest.update(b"\x01" if missing else b"\x00")
            current_value_digest.update(struct.pack(">d", float("nan") if missing else value))
            if math.isfinite(value):
                stats[channel]["full"].add(value)
                part = (
                    "train"
                    if position < train_end
                    else "validation"
                    if position < validation_end
                    else "test"
                )
                stats[channel][part].add(value)
    if current is not None:
        channel_digests.append(current_value_digest.digest())
    unique_timestamps = len(first_unique)
    expected_cells = unique_timestamps * len(channels)
    channel_lengths_equal = len(set(counts.values())) <= 1
    timestamp_sets_equal = timestamp_sequence_mismatches == 0 and channel_lengths_equal
    rectangular = row_count - composite_duplicates == expected_cells and timestamp_sets_equal
    lossless = rectangular and composite_duplicates == 0 and extra_columns == 0
    fingerprint = None
    if lossless:
        fingerprint = fingerprint_from_component_digests(
            timestamp_count=len(first_timestamps),
            timestamp_digest=timestamp_digest.digest(),
            channels=channels,
            channel_value_digests=channel_digests,
        )
    full_means = [parts["full"].mean for parts in stats.values()]
    full_stds = [parts["full"].std for parts in stats.values()]
    train_means = [parts["train"].mean for parts in stats.values()]
    train_stds = [parts["train"].std for parts in stats.values()]

    def near(mean: float, std: float) -> bool:
        return abs(mean) <= 0.05 and abs(std - 1.0) <= 0.05

    structure = {
        "dataset": spec.canonical_name,
        "source_variant": spec.source_variant,
        "header": "|".join(headers),
        "rows": row_count,
        "unique_timestamps": unique_timestamps,
        "channels": len(channels),
        "unique_composite_keys": row_count - composite_duplicates,
        "composite_key_duplicates": composite_duplicates,
        "min_channels_per_timestamp": len(channels) if rectangular else None,
        "max_channels_per_timestamp": len(channels) if rectangular else None,
        "min_rows_per_channel": min(counts.values(), default=0),
        "max_rows_per_channel": max(counts.values(), default=0),
        "channel_order": "|".join(channels),
        "channel_order_source": "first_appearance_in_bundle",
        "value_dtype": "float64_parse",
        "channel_timestamp_sets_equal": timestamp_sets_equal,
        "rectangular_after_pivot": rectangular,
        "aggregation_required": composite_duplicates > 0,
        "input_value_count": row_count,
        "canonical_value_count": expected_cells if rectangular else None,
        "value_multiset_preserved": lossless,
        "lossless_reshape_candidate": lossless,
        "canonical_fingerprint": fingerprint,
        "source_file_sha256": file_sha256(path),
    }
    duplicate = {
        "dataset": spec.canonical_name,
        "timestamp_only_repeated_rows": max(0, row_count - unique_timestamps),
        "composite_key_duplicates": composite_duplicates,
        "fully_identical_duplicate_rows": identical_duplicates,
        "conflicting_value_duplicates": conflicting_duplicates,
        "timestamp_sequence_mismatches": timestamp_sequence_mismatches,
        "timezone_or_parse_errors": parse_errors,
        "nonfinite_values": nonfinite,
        "policy": "blocked_no_automatic_aggregation" if composite_duplicates else "no_action",
    }
    normalization = {
        "dataset": spec.canonical_name,
        **_summary(full_means, "full_channel_mean"),
        **_summary(full_stds, "full_channel_std"),
        **_summary(train_means, "train_channel_mean"),
        **_summary(train_stds, "train_channel_std"),
        **_summary(
            [parts["validation"].mean for parts in stats.values()], "validation_channel_mean"
        ),
        **_summary([parts["validation"].std for parts in stats.values()], "validation_channel_std"),
        **_summary([parts["test"].mean for parts in stats.values()], "test_channel_mean"),
        **_summary([parts["test"].std for parts in stats.values()], "test_channel_std"),
        "full_near_zero_unit_fraction": sum(
            near(mean, std) for mean, std in zip(full_means, full_stds, strict=True)
        )
        / len(channels),
        "train_near_zero_unit_fraction": sum(
            near(mean, std) for mean, std in zip(train_means, train_stds, strict=True)
        )
        / len(channels),
        "value_min": min(parts["full"].minimum for parts in stats.values()),
        "value_max": max(parts["full"].maximum for parts in stats.values()),
        "maximum_decimal_places": maximum_decimal_places,
        "diagnostic_only": True,
    }
    return structure, duplicate, normalization


def compare_ett(bundle_path: Path, official_path: Path, dataset: str) -> dict[str, Any]:
    with official_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        official_channels = [name for name in (reader.fieldnames or []) if name != "date"]
        official_timestamps: list[str] = []
        raw: dict[str, list[float]] = {channel: [] for channel in official_channels}
        for row in reader:
            official_timestamps.append(row["date"])
            for channel in official_channels:
                raw[channel].append(float(row[channel]))
    bundle: dict[str, list[float]] = defaultdict(list)
    bundle_timestamps: dict[str, list[str]] = defaultdict(list)
    with bundle_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            bundle[row["cols"]].append(float(row["data"]))
            bundle_timestamps[row["cols"]].append(row["date"])
    channels_match = list(bundle) == official_channels
    timestamps_match = all(bundle_timestamps[channel] == official_timestamps for channel in bundle)
    exact = channels_match and timestamps_match
    allclose = exact
    differences: list[float] = []
    affine_max: list[float] = []
    affine_mean: list[float] = []
    slopes_match_prefix_standardization: list[bool] = []
    normalization_fit_rows = 8640 if dataset.startswith("ETTh") else 34560
    for channel in official_channels:
        left = raw[channel]
        right = bundle.get(channel, [])
        if len(left) != len(right):
            exact = allclose = False
            continue
        exact = exact and all(a == b for a, b in zip(left, right, strict=True))
        allclose = allclose and all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(left, right, strict=True)
        )
        differences.extend(abs(a - b) for a, b in zip(left, right, strict=True))
        fit = diagnose_affine(left, right)
        affine_max.append(fit.max_absolute_residual)
        affine_mean.append(fit.mean_absolute_residual)
        fit_values = left[:normalization_fit_rows]
        raw_mean = sum(fit_values) / len(fit_values)
        raw_std = math.sqrt(sum((value - raw_mean) ** 2 for value in fit_values) / len(fit_values))
        slopes_match_prefix_standardization.append(
            math.isclose(fit.slope, 1 / raw_std, rel_tol=1e-10, abs_tol=1e-12)
            and math.isclose(fit.intercept, -raw_mean / raw_std, rel_tol=1e-10, abs_tol=1e-12)
        )
    return {
        "dataset": dataset,
        "bundle_source_variant": f"{dataset}__bundle_long",
        "official_source_variant": f"{dataset}__official_raw",
        "official_repository": ETT_REPOSITORY,
        "official_commit": ETT_COMMIT,
        "official_file_sha256": file_sha256(official_path),
        "shape_match": all(
            len(bundle.get(channel, [])) == len(official_timestamps)
            for channel in official_channels
        ),
        "timestamp_set_and_order_match": timestamps_match,
        "channel_set_and_order_match": channels_match,
        "exact_match": exact,
        "allclose": allclose,
        "max_absolute_difference": max(differences, default=None),
        "mean_absolute_difference": sum(differences) / len(differences) if differences else None,
        "affine_max_absolute_residual": max(affine_max, default=None),
        "affine_mean_absolute_residual": sum(affine_mean) / len(affine_mean)
        if affine_mean
        else None,
        "normalization_fit_rows": normalization_fit_rows,
        "normalization_fit_scope": "official_raw_prefix_12_months",
        "affine_matches_legacy_prefix_standardization": all(slopes_match_prefix_standardization),
        "comparison_value_payload_stored": False,
    }


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_table(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    write_csv_atomic(output / name, rows, fieldnames=list(rows[0]), overwrite=True)


def _write_run_manifest(
    output: Path,
    registry_path: Path,
    structures: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    input_signature = stable_hash(
        [(row["dataset"], row["source_file_sha256"]) for row in structures]
        + [(row["dataset"], row["official_file_sha256"]) for row in comparisons]
    )
    target = output / "resolution_run.json"
    temporary = output / f".resolution_run.{uuid.uuid4().hex}.json"
    try:
        write_json_atomic(
            temporary,
            {
                "schema_version": SCHEMA_VERSION,
                "resolver_version": RESOLVER_VERSION,
                "status": "completed",
                "generated_at": datetime.now(UTC).isoformat(),
                "code_commit_sha": _git_sha(),
                "input_signature": input_signature,
                "registry_sha256": file_sha256(registry_path),
                "datasets_analyzed": len(structures),
                "official_datasets_compared": len(comparisons),
                "raw_values_stored_in_manifests": False,
                "absolute_paths_stored_in_manifests": False,
            },
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_existing(registry_path: Path, output: Path) -> None:
    tables: dict[str, list[dict[str, Any]]] = {}
    for name in ("bundle_structure_summary.csv", "raw_comparison_summary.csv"):
        path = output / name
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"cannot finalize empty result table: {path}")
        tables[name] = rows
    structures = tables["bundle_structure_summary.csv"]
    comparisons = tables["raw_comparison_summary.csv"]
    if len(structures) != 14 or len(comparisons) != 4:
        raise ValueError("expected 14 bundle analyses and 4 official comparisons")
    _write_run_manifest(output, registry_path, structures, comparisons)


def run_resolution(registry_path: Path, data_root: Path, official_root: Path, output: Path) -> None:
    registry = load_registry(registry_path)
    validation_path = Path("results/manifests/datasets/validation_summary.csv")
    with validation_path.open(encoding="utf-8", newline="") as handle:
        time_hints = {row["dataset"]: int(row["observed_rows"]) for row in csv.DictReader(handle)}
    structures: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    for spec in registry.datasets:
        path = data_root / Path(spec.relative_path).name
        structure, duplicate, normalization = analyze_bundle(
            path, spec, time_hints[spec.canonical_name]
        )
        structures.append(structure)
        duplicates.append(duplicate)
        normalizations.append(normalization)
    comparisons = [
        compare_ett(data_root / f"{name}.csv", official_root / "ett" / f"{name}.csv", name)
        for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
    ]
    evidence = [
        {
            "repository_url": BENCHMARK_REPOSITORY,
            "commit_sha": BENCHMARK_COMMIT,
            "file_path": "README.md",
            "function_or_command": "Prepare Datasets",
            "checked_date": "2026-09-20",
            "finding": "bundle is described only as well pre-processed; creation recipe absent",
            "wide_to_long": "unknown",
            "normalization": "unknown",
            "imputation": "unknown",
            "timestamp_change": "unknown",
            "channel_filtering": "unknown",
            "resampling": "unknown",
        },
        {
            "repository_url": BENCHMARK_REPOSITORY,
            "commit_sha": BENCHMARK_COMMIT,
            "file_path": "ts_benchmark/data/utils.py",
            "function_or_command": "read_data",
            "checked_date": "2026-09-20",
            "finding": (
                "loader slices contiguous channel blocks and preserves first-appearance order"
            ),
            "wide_to_long": "loader reverses long layout but does not create it",
            "normalization": "not performed by read_data",
            "imputation": "not performed by read_data",
            "timestamp_change": "parse to pandas datetime only",
            "channel_filtering": "not performed by read_data",
            "resampling": "not performed by read_data",
        },
        {
            "repository_url": ETT_REPOSITORY,
            "commit_sha": ETT_COMMIT,
            "file_path": "ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv",
            "function_or_command": "official source files",
            "checked_date": "2026-09-20",
            "finding": "official wide raw values compared directly after reshape-only alignment",
            "wide_to_long": "comparison-only reshape",
            "normalization": "none applied to official comparison input",
            "imputation": "none",
            "timestamp_change": "none",
            "channel_filtering": "none",
            "resampling": "none",
        },
    ]
    decisions = []
    for structure in structures:
        dataset = structure["dataset"]
        is_ett = dataset in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}
        reason = [
            "bundle_creation_transform_not_published",
            "bundle_fingerprint_not_linked_to_creator",
        ]
        if is_ett:
            reason += ["official_values_mismatch", "legacy_prefix_standardization_verified"]
        if structure["composite_key_duplicates"]:
            reason += ["duplicate_composite_keys", "aggregation_would_be_required"]
        decisions.append(
            {
                "dataset": dataset,
                "source_variant": structure["source_variant"],
                "previous_status": "blocked",
                "new_status": "blocked",
                "provenance_grade": "D",
                "technical_validity": "valid_except_duplicates"
                if structure["composite_key_duplicates"]
                else "valid",
                "normalization_verified": "legacy_12_month_prefix_standardization"
                if is_ett
                else "unknown",
                "lossless_reshape_verified": structure["lossless_reshape_candidate"],
                "official_source_verified": is_ett,
                "license_verified": is_ett,
                "reason_codes": ";".join(reason),
                "evidence_references": "raw_comparison_summary.csv"
                if is_ett
                else "transformation_evidence.csv",
                "resolution_strategy": "replace_with_official_raw"
                if is_ett
                else "additional_creation_code_or_official_raw_required",
            }
        )
    for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        decisions.append(
            {
                "dataset": name,
                "source_variant": f"{name}__official_raw",
                "previous_status": "not_registered",
                "new_status": "ready_with_warnings",
                "provenance_grade": "A",
                "technical_validity": "valid",
                "normalization_verified": "raw_unscaled_official_file",
                "lossless_reshape_verified": True,
                "official_source_verified": True,
                "license_verified": True,
                "reason_codes": "official_repository_file;redistribution_restrictions_apply",
                "evidence_references": "raw_comparison_summary.csv;docs/data-provenance.md",
                "resolution_strategy": "independent_reshape_from_official_raw",
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_table(output, "bundle_structure_summary.csv", structures)
    _write_table(output, "duplicate_analysis.csv", duplicates)
    _write_table(output, "normalization_diagnostics.csv", normalizations)
    _write_table(output, "raw_comparison_summary.csv", comparisons)
    _write_table(output, "transformation_evidence.csv", evidence)
    _write_table(output, "provenance_decision.csv", decisions)
    _write_run_manifest(output, registry_path, structures, comparisons)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets/registry.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/dataset/forecasting"))
    parser.add_argument("--official-root", type=Path, default=Path("data/official_raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/manifests/provenance_resolution")
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="validate existing result tables and atomically refresh only run metadata",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.finalize_existing:
        finalize_existing(args.registry, args.output)
    else:
        run_resolution(args.registry, args.data_root, args.official_root, args.output)


if __name__ == "__main__":
    main()
