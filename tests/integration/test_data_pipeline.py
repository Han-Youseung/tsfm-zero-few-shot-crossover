from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from tsfm_crossover.data.loader import load_dataset
from tsfm_crossover.data.registry import DatasetSpec, SourceSpec
from tsfm_crossover.data.sampling import build_sampling_manifest
from tsfm_crossover.data.scaling import TrainOnlyStandardScaler
from tsfm_crossover.data.splits import chronological_split
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.tracking.atomic import write_json_atomic


def test_synthetic_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.csv"
    start = datetime(2024, 1, 1)
    rows = [f"{(start + timedelta(hours=i)).isoformat()},{i},{i * 2}" for i in range(100)]
    path.write_text("date,a,b\n" + "\n".join(rows) + "\n", encoding="utf-8")
    spec = DatasetSpec(
        canonical_name="Synthetic",
        relative_path=path.name,
        timestamp_column="date",
        timestamp_required=True,
        target_columns=["a", "b"],
        source=SourceSpec(official_name="Synthetic", local_filename=path.name),
    )
    loaded = load_dataset(spec, tmp_path)
    assert loaded.validation_report.valid
    split = chronological_split(loaded.row_count, loaded.timestamps)
    scaler = TrainOnlyStandardScaler.fit(loaded.values, split.train)
    train = generate_train_windows("Synthetic", loaded.source_sha256, split.train, 10, 5)
    validation = generate_rolling_windows(
        "Synthetic", loaded.source_sha256, "validation", split.validation, 10, 5
    )
    test = generate_rolling_windows("Synthetic", loaded.source_sha256, "test", split.test, 10, 5)
    assert scaler.fit_interval.end == 60 and validation and test
    manifest = build_sampling_manifest(
        train,
        dataset_sha256=loaded.source_sha256,
        split_config_hash=split.split_config_hash,
        seed=42,
        code_commit_sha="test",
        generated_at="fixed",
    )
    target = write_json_atomic(tmp_path / "manifest.json", manifest.as_dict())
    restored = json.loads(target.read_text(encoding="utf-8"))
    assert restored["manifest_hash"] == manifest.manifest_hash
    assert restored["selections"][-1]["selected_window_ids"] == [
        w.window_id for w in (train[i] for i in temporal_order(train))
    ]


def temporal_order(train):
    from tsfm_crossover.data.sampling import temporal_master_order

    return temporal_master_order(train, 42)
