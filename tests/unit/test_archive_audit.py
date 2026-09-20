from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tsfm_crossover.data.archive import UnsafeArchiveError, extract_zip_safely, inspect_zip
from tsfm_crossover.data.audit import run_audit


def make_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as handle:
        for name, value in members.items():
            handle.writestr(name, value)
    return path


@pytest.mark.parametrize("member", ["../escape.csv", "/absolute.csv", "C:/absolute.csv"])
def test_archive_rejects_traversal_and_absolute(tmp_path: Path, member: str) -> None:
    archive = make_zip(tmp_path / "bad.zip", {member: "x"})
    assert not inspect_zip(archive).safe
    with pytest.raises(UnsafeArchiveError):
        extract_zip_safely(archive, tmp_path / "out")


def test_archive_checksum_and_overwrite_guard(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "ok.zip", {"folder/a.csv": "x"})
    inspection = inspect_zip(archive)
    assert len(inspection.sha256) == 64 and inspection.safe
    destination = tmp_path / "out"
    extract_zip_safely(archive, destination)
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_zip_safely(archive, destination)


def write_registry(path: Path, datasets: list[str]) -> Path:
    entries = []
    for name in datasets:
        entries.append(
            {
                "canonical_name": name,
                "aliases": ["CzenLan"] if name == "CzeLan" else [],
                "relative_path": f"data/{name}.csv",
                "file_format": "csv",
                "layout": "long",
                "timestamp_column": "date",
                "timestamp_required": True,
                "frequency": None,
                "expected_rows": None,
                "expected_channels": None,
                "official_rows": None,
                "official_channels": None,
                "official_frequency": None,
                "target_columns": ["data"],
                "all_numeric_targets": False,
                "excluded_columns": ["cols"],
                "value_column": "data",
                "channel_column": "cols",
                "series_id_column": None,
                "local_preprocessing_status": "unverified",
                "source": {
                    "official_name": name,
                    "verification_status": "pending",
                    "local_filename": f"{name}.csv",
                },
            }
        )
    import yaml

    path.write_text(yaml.safe_dump({"version": 1, "datasets": entries}), encoding="utf-8")
    return path


def write_long_csv(path: Path, length: int = 1500) -> None:
    from datetime import datetime, timedelta

    rows = ["date,data,cols"]
    start = datetime(2024, 1, 1)
    for channel in ("a", "b"):
        rows.extend(
            f"{(start + timedelta(hours=i)).isoformat(sep=' ')},{i + 1},{channel}"
            for i in range(length)
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_synthetic_audit_missing_summaries_and_safe_manifests(tmp_path: Path) -> None:
    registry = write_registry(tmp_path / "registry.yaml", ["ETTh1", "CzeLan", "Missing"])
    data = tmp_path / "raw"
    data.mkdir()
    write_long_csv(data / "ETTh1.csv")
    write_long_csv(data / "CzenLan.csv")
    output = tmp_path / "results"
    result = run_audit(
        registry_path=registry,
        data_root=data,
        output=output,
        contexts=[256, 512],
        horizons=[96, 192, 336, 720],
        rates=[0, 0.005, 1],
        seed=42,
    )
    assert result["dataset_states"]["Missing"] == "missing"
    assert result["dataset_states"]["ETTh1"] == "blocked"
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    serialized = json.dumps(inventory)
    assert str(tmp_path) not in serialized
    assert "2024-01-01 00:00:00" not in serialized
    czen = next(item for item in inventory["files"] if item["filename"] == "CzenLan.csv")
    assert czen["alias_applied"] is True
    window_text = (output / "window_feasibility.csv").read_text(encoding="utf-8")
    assert ",256,96," in window_text and ",512,720," in window_text
    assert (output / "sampling_count_summary.csv").is_file()
    assert (output / "coverage_summary.csv").is_file()
    provenance = (output / "provenance_summary.csv").read_text(encoding="utf-8")
    assert "expected_rows" in provenance and "official_rows" in provenance
    skipped = run_audit(
        registry_path=registry,
        data_root=data,
        output=output,
        contexts=[256, 512],
        horizons=[96, 192, 336, 720],
        rates=[0, 0.005, 1],
        seed=42,
    )
    assert skipped["skipped"] is True

    with pytest.raises(FileExistsError, match="different inputs"):
        run_audit(
            registry_path=registry,
            data_root=data,
            output=output,
            contexts=[128],
            horizons=[24],
            rates=[0, 1],
            seed=42,
        )


def test_ambiguous_exact_mapping_is_not_selected(tmp_path: Path) -> None:
    registry = write_registry(tmp_path / "registry.yaml", ["ETTh1"])
    data = tmp_path / "raw"
    (data / "one").mkdir(parents=True)
    (data / "two").mkdir()
    write_long_csv(data / "one" / "ETTh1.csv", 10)
    write_long_csv(data / "two" / "ETTh1.csv", 10)
    result = run_audit(
        registry_path=registry,
        data_root=data,
        output=tmp_path / "results",
        contexts=[2],
        horizons=[1],
        rates=[0, 1],
        seed=42,
        dry_run=True,
    )
    assert result["dataset_states"]["ETTh1"] == "ambiguous"


def test_verified_raw_clean_dataset_can_be_ready(tmp_path: Path) -> None:
    import yaml

    registry = write_registry(tmp_path / "registry.yaml", ["ETTh1"])
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    payload["datasets"][0]["local_preprocessing_status"] = "verified_raw"
    payload["datasets"][0]["source"]["verification_status"] = "verified"
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")
    data = tmp_path / "raw"
    data.mkdir()
    write_long_csv(data / "ETTh1.csv", 100)
    result = run_audit(
        registry_path=registry,
        data_root=data,
        output=tmp_path / "results",
        contexts=[12],
        horizons=[6],
        rates=[0, 1],
        seed=42,
        dry_run=True,
    )
    assert result["dataset_states"]["ETTh1"] == "ready"
