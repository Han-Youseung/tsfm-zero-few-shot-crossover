import json
from pathlib import Path

import pytest

from tsfm_crossover.tracking.atomic import (
    CompletedResultExistsError,
    CorruptExistingResultError,
    write_json_atomic,
)


def test_atomic_json_write(tmp_path: Path):
    path = tmp_path / "nested" / "result.json"
    write_json_atomic(path, {"experiment_id": "one", "status": "running"})
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_id"] == "one"
    assert list(path.parent.glob("*.tmp")) == []


def test_completed_result_is_not_overwritten(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text('{"experiment_id":"one","status":"completed"}', encoding="utf-8")
    with pytest.raises(CompletedResultExistsError):
        write_json_atomic(
            path,
            {"experiment_id": "one", "status": "running"},
            experiment_id="one",
            overwrite=True,
        )


def test_corrupt_json_is_detected(tmp_path: Path):
    path = tmp_path / "result.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(CorruptExistingResultError):
        write_json_atomic(path, {"status": "running"}, overwrite=True)
