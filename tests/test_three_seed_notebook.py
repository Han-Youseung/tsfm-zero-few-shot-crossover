"""Offline orchestration tests; no GPU, vendor imports, or weight downloads."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/41_main_study_three_seeds.ipynb"


def cells():
    return {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text("utf-8"))["cells"]}


def setup_run(tmp_path, completed=False, fail_at=None, pending=False):
    jobs = [
        dict(
            id=f"job-{seed}-{rate}", seed=seed, rate=rate, family="ttm", dataset="ETTh2", horizon=96
        )
        for seed in (1729, 2718, 31415)
        for rate in range(9)
    ]
    calls = []
    for row in jobs:
        directory = tmp_path / row["id"]
        directory.mkdir()
        if completed:
            (directory / "result.json").write_text('{"status": "completed"}')

    def run(command, check):
        assert check
        condition = command[-1]
        calls.append(condition)
        if len(calls) == fail_at:
            raise RuntimeError("runner failure")
        result = tmp_path / condition / "result.json"
        if not result.exists() and not pending:
            result.write_text('{"status": "completed"}')

    namespace = dict(
        SEEDS=(1729, 2718, 31415),
        jobs=jobs,
        FAMILY="ttm",
        DATASET="ETTh2",
        HORIZON=96,
        OUT=tmp_path,
        BASE=["python", "runner"],
        MAX_CONDITIONS=27,
        WALL_HOURS=18,
        time=SimpleNamespace(monotonic=lambda: 0),
        json=json,
        subprocess=SimpleNamespace(run=run),
    )
    return namespace, calls


def test_notebook_clean_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text("utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), cell["id"], "exec")
    assert "input(" not in cells()["run"]
    assert "s{SEED}" not in cells()["export"]


@pytest.mark.parametrize("completed", [False, True])
def test_three_seeds_sequential_and_existing_validated(tmp_path, completed):
    namespace, calls = setup_run(tmp_path, completed=completed)
    exec(cells()["run"], namespace)
    assert calls == [r["id"] for r in namespace["jobs"]]
    assert namespace["executed"] == (0 if completed else 27)


def test_failure_stops_remaining_jobs(tmp_path):
    namespace, calls = setup_run(tmp_path, fail_at=2)
    with pytest.raises(RuntimeError, match="runner failure"):
        exec(cells()["run"], namespace)
    assert len(calls) == 2


def test_resource_pending_stops_remaining_jobs(tmp_path):
    namespace, calls = setup_run(tmp_path, pending=True)
    with pytest.raises(RuntimeError, match="incomplete condition"):
        exec(cells()["run"], namespace)
    assert len(calls) == 1


def test_wall_budget_stops_before_launch(tmp_path):
    namespace, calls = setup_run(tmp_path)
    times = iter([0, 18 * 3600])
    namespace["time"] = SimpleNamespace(monotonic=lambda: next(times))
    exec(cells()["run"], namespace)
    assert calls == []
