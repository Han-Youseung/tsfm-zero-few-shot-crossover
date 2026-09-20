"""Validation primitives for provenance decisions and source variants."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum


class ProvenanceGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class ProvenanceEvidence:
    official_exact: bool = False
    code_verified_lossless: bool = False
    fingerprint_linked: bool = False
    transformed_or_ambiguous: bool = False


def validate_manifest_safety(payload: object) -> None:
    """Reject raw-value payload fields and local absolute paths in small manifests."""
    forbidden_keys = {"raw_values", "canonical_values", "time_series_values"}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            bad = forbidden_keys.intersection(value)
            if bad:
                raise ValueError(f"raw value payload is forbidden: {sorted(bad)}")
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str) and (
            re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/")
        ):
            raise ValueError("local absolute path is forbidden in manifests")

    visit(payload)


def validate_provenance_decision(
    grade: ProvenanceGrade | str, status: str, evidence: ProvenanceEvidence
) -> None:
    grade = ProvenanceGrade(grade)
    if grade == ProvenanceGrade.A and not evidence.official_exact:
        raise ValueError("grade A requires official exact-match evidence")
    if grade == ProvenanceGrade.B and not (
        evidence.code_verified_lossless and evidence.fingerprint_linked
    ):
        raise ValueError("grade B requires linked, code-verified lossless evidence")
    if status in {"ready", "ready_with_warnings"} and grade not in {
        ProvenanceGrade.A,
        ProvenanceGrade.B,
    }:
        raise ValueError("blocked status cannot be released for grade C or D")


def validate_single_source_variant(variants: Iterable[str]) -> str:
    observed = {item for item in variants if item}
    if len(observed) != 1:
        raise ValueError("an experiment group must use exactly one source variant")
    return next(iter(observed))


@dataclass(frozen=True)
class NumericComparison:
    exact: bool
    allclose: bool
    max_absolute_difference: float
    mean_absolute_difference: float


def compare_numeric(
    left: Sequence[float],
    right: Sequence[float],
    *,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-8,
) -> NumericComparison:
    if len(left) != len(right):
        raise ValueError("comparison vectors must have equal length")
    differences = [abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)]
    return NumericComparison(
        exact=all(a == b for a, b in zip(left, right, strict=True)),
        allclose=all(
            math.isclose(a, b, rel_tol=relative_tolerance, abs_tol=absolute_tolerance)
            for a, b in zip(left, right, strict=True)
        ),
        max_absolute_difference=max(differences, default=0.0),
        mean_absolute_difference=sum(differences) / len(differences) if differences else 0.0,
    )


@dataclass(frozen=True)
class AffineFit:
    slope: float
    intercept: float
    max_absolute_residual: float
    mean_absolute_residual: float


def diagnose_affine(raw: Sequence[float], transformed: Sequence[float]) -> AffineFit:
    if len(raw) != len(transformed) or len(raw) < 2:
        raise ValueError("affine diagnosis requires equal vectors with at least two values")
    raw_mean = sum(raw) / len(raw)
    out_mean = sum(transformed) / len(transformed)
    denominator = sum((value - raw_mean) ** 2 for value in raw)
    if denominator == 0:
        raise ValueError("affine diagnosis requires non-constant raw values")
    slope = (
        sum((x - raw_mean) * (y - out_mean) for x, y in zip(raw, transformed, strict=True))
        / denominator
    )
    intercept = out_mean - slope * raw_mean
    residuals = [abs(y - (slope * x + intercept)) for x, y in zip(raw, transformed, strict=True)]
    return AffineFit(
        slope,
        intercept,
        max(residuals),
        sum(residuals) / len(residuals),
    )
