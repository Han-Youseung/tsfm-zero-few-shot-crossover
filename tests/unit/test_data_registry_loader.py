from __future__ import annotations

from pathlib import Path

import pytest

from tsfm_crossover.data.loader import load_dataset
from tsfm_crossover.data.registry import DatasetSpec, SourceSpec, load_registry


def spec(
    path: str, *, timestamp: str | None = "date", targets: list[str] | None = None
) -> DatasetSpec:
    return DatasetSpec(
        canonical_name="Synthetic",
        relative_path=path,
        timestamp_column=timestamp,
        timestamp_required=timestamp is not None,
        target_columns=targets or ["a", "b"],
        all_numeric_targets=False,
        source=SourceSpec(official_name="Synthetic", local_filename=Path(path).name),
    )


def write_csv(tmp_path: Path, rows: list[str], header: str = "date,a,b") -> Path:
    path = tmp_path / "series.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def codes(loaded: object) -> set[str]:
    return {issue.code for issue in loaded.validation_report.issues}


def test_registry_has_fourteen_and_exact_alias() -> None:
    registry = load_registry("configs/datasets/registry.yaml")
    assert len(registry.datasets) == 14
    assert registry.resolve("CzenLan").canonical_name == "CzeLan"
    with pytest.raises(KeyError, match="fuzzy matching"):
        registry.resolve("CzeLann")


def test_normal_multivariate_timestamp_csv(tmp_path: Path) -> None:
    path = write_csv(tmp_path, ["2024-01-01T00:00:00,1,2", "2024-01-01T01:00:00,2,3"])
    loaded = load_dataset(spec(path.name), tmp_path)
    assert loaded.validation_report.valid
    assert loaded.channel_names == ("a", "b")
    assert loaded.inferred_frequency_seconds == 3600
    assert loaded.positional_index is None


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (["2024-01-01T00:00:00,1,2", "2024-01-01T00:00:00,2,3"], "duplicate_timestamp"),
        (["2024-01-02T00:00:00,1,2", "2024-01-01T00:00:00,2,3"], "non_monotonic_timestamp"),
        (
            ["2024-01-01T00:00:00,1,2", "2024-01-01T01:00:00,2,3", "2024-01-01T03:00:00,3,4"],
            "irregular_frequency",
        ),
        (["2024-01-01T00:00:00,,2"], "missing_value"),
        (["2024-01-01T00:00:00,inf,2"], "infinite_value"),
        (["2024-01-01T00:00:00,nope,2"], "non_numeric_target"),
        (["2024-01-01T00:00:00,1,2", "2024-01-01T01:00:00,1,3"], "constant_channel"),
    ],
)
def test_validation_diagnostics(tmp_path: Path, rows: list[str], expected: str) -> None:
    loaded = load_dataset(spec(write_csv(tmp_path, rows).name), tmp_path)
    assert expected in codes(loaded)


def test_timestamp_free_uses_position(tmp_path: Path) -> None:
    path = write_csv(tmp_path, ["1,2", "3,4"], header="a,b")
    loaded = load_dataset(spec(path.name, timestamp=None), tmp_path)
    assert loaded.timestamps is None
    assert loaded.positional_index == (0, 1)


def test_missing_file_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="automatic download is disabled"):
        load_dataset(spec("missing.csv"), tmp_path)


def test_expected_frequency_and_sha_are_checked(tmp_path: Path) -> None:
    path = write_csv(tmp_path, ["2024-01-01T00:00:00,1,2", "2024-01-01T01:00:00,2,3"])
    item = spec(path.name)
    item.frequency = "15min"
    item.source.file_sha256 = "0" * 64
    loaded = load_dataset(item, tmp_path)
    assert {"unexpected_frequency", "source_sha256_mismatch"}.issubset(codes(loaded))
