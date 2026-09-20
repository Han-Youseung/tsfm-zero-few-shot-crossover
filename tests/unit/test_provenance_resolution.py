import csv
from pathlib import Path

from tsfm_crossover.data.registry import DatasetSpec, SourceSpec
from tsfm_crossover.data.resolve_provenance import analyze_bundle


def spec() -> DatasetSpec:
    return DatasetSpec(
        canonical_name="WeatherFixture",
        source_variant="WeatherFixture__bundle_long",
        relative_path="WeatherFixture.csv",
        layout="long",
        timestamp_column="date",
        target_columns=["data"],
        value_column="data",
        channel_column="cols",
        source=SourceSpec(official_name="fixture", local_filename="WeatherFixture.csv"),
    )


def write_rows(path: Path, rows: list[tuple[str, float, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "data", "cols"])
        writer.writerows(rows)


def test_timestamp_repetition_across_channels_is_not_composite_duplicate(tmp_path: Path):
    path = tmp_path / "WeatherFixture.csv"
    rows = [("t1", 1, "a"), ("t2", 2, "a"), ("t1", 3, "b"), ("t2", 4, "b")]
    write_rows(path, rows)
    structure, duplicate, _ = analyze_bundle(path, spec(), 2)
    assert duplicate["timestamp_only_repeated_rows"] == 2
    assert duplicate["composite_key_duplicates"] == 0
    assert structure["lossless_reshape_candidate"] is True


def test_identical_weather_style_duplicate_is_counted_per_channel(tmp_path: Path):
    path = tmp_path / "WeatherFixture.csv"
    rows = []
    for channel in ("a", "b"):
        rows.extend([("t1", 1, channel), ("t1", 1, channel), ("t2", 2, channel)])
    write_rows(path, rows)
    structure, duplicate, _ = analyze_bundle(path, spec(), 3)
    assert duplicate["composite_key_duplicates"] == 2
    assert duplicate["fully_identical_duplicate_rows"] == 2
    assert duplicate["conflicting_value_duplicates"] == 0
    assert structure["lossless_reshape_candidate"] is False
