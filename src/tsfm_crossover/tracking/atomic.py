"""Atomic persistence for small JSON and CSV records."""

from __future__ import annotations

import csv
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class AtomicWriteError(RuntimeError):
    """Base class for safe result persistence errors."""


class CorruptExistingResultError(AtomicWriteError):
    """Raised when an existing JSON result cannot be decoded."""


class CompletedResultExistsError(AtomicWriteError):
    """Raised when an existing completed result would be overwritten."""


class DuplicateResultError(AtomicWriteError):
    """Raised when a result exists and replacement was not requested."""


def _guard_json_target(path: Path, experiment_id: str | None, overwrite: bool) -> None:
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorruptExistingResultError(f"existing JSON is corrupt: {path}") from exc
    if experiment_id is not None and existing.get("experiment_id") not in (None, experiment_id):
        raise DuplicateResultError("existing result belongs to a different experiment_id")
    if existing.get("status") == "completed":
        raise CompletedResultExistsError(f"completed result already exists: {path}")
    if not overwrite:
        raise DuplicateResultError(f"result already exists: {path}")


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    experiment_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _guard_json_target(target, experiment_id, overwrite)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_csv_atomic(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: list[str],
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise DuplicateResultError(f"CSV already exists: {target}")
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
