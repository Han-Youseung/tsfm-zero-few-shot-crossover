"""CPU-only queue tests: fake model CLI, real files/archives and tiny subprocesses."""

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from scripts.colab_queue import execute_queue, export_evidence, run_process, select_queue

from tsfm_crossover.experiments.main_plan import grid, load_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    model, _ = load_plan(ROOT, ROOT / "configs/study/main.yaml")
    return {"commit": "a" * 40, "conditions": grid(model), "config": model.model_dump(mode="json")}


def test_selection_order_all_three_seeds_and_all_datasets(plan):
    selected = select_queue(plan, "ttm", ["ETTh2", "ETTh1"], [192, 96], [1729, 2718, 31415])
    assert len(selected) == 108
    assert selected[0]["dataset"] == "ETTh2" and selected[0]["horizon"] == 192
    assert selected[9]["seed"] == 2718
    assert selected[27]["horizon"] == 96
    assert selected[54]["dataset"] == "ETTh1"
    assert (
        len(select_queue(plan, "moirai1", ["ALL"], [96, 192, 336, 720], [1729, 2718, 31415])) == 972
    )


@pytest.mark.parametrize(
    "datasets,horizons,seeds",
    [
        ([], [96], [1729]),
        (["invalid"], [96], [1729]),
        (["ETTh1", "ETTh1"], [96], [1729]),
        (["ETTh1"], [100], [1729]),
        (["ETTh1"], [96], [1]),
    ],
)
def test_invalid_selection_rejected(plan, datasets, horizons, seeds):
    with pytest.raises(ValueError):
        select_queue(plan, "ttm", datasets, horizons, seeds)


def fixture_run(tmp_path, plan, *, behavior="complete", existing=False):
    out = tmp_path / "results"
    out.mkdir()
    (out / "plan.json").write_text(json.dumps(plan))
    jobs = select_queue(plan, "ttm", ["ETTh1"], [96], [1729, 2718, 31415])
    calls = []

    def result(row):
        directory = out / row["id"]
        directory.mkdir(exist_ok=True)
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "identity": {"condition": row, "commit": plan["commit"]},
                }
            )
        )

    if existing:
        for row in jobs:
            result(row)

    def process(command, cwd, log, label):
        calls.append(command)
        if "--dataset" in command:
            return
        row = next(r for r in jobs if r["id"] == command[-1])
        directory = out / row["id"]
        directory.mkdir(exist_ok=True)
        if behavior == "fail":
            (directory / "failure-fixture.json").write_text('{"status": "failed"}')
            raise subprocess.CalledProcessError(1, command)
        if behavior == "interrupt":
            (directory / "running.lock").write_text("retained")
            raise KeyboardInterrupt
        if behavior == "pending":
            (directory / "pending-fixture.json").write_text('{"status": "pending_gpu"}')
            return
        if not existing:
            result(row)

    args = dict(
        jobs=jobs,
        family="ttm",
        commit=plan["commit"],
        root=tmp_path,
        out=out,
        python="python",
        backup_dir=tmp_path / "backups",
        local_logs=tmp_path / "logs",
        session_started=0,
        process=process,
        now=lambda: 0,
    )
    return args, calls


@pytest.mark.parametrize("existing", [True, False])
def test_sequential_resume_validate_and_backup(tmp_path, plan, existing):
    args, calls = fixture_run(tmp_path, plan, existing=existing)
    original = {p: p.read_bytes() for p in args["out"].rglob("result.json")}
    record = execute_queue(**args)
    assert record["status"] == "completed" and record["verified"] == 27
    assert record["newly_executed"] == (0 if existing else 27)
    runs = [c for c in calls if "--condition-id" in c]
    assert [c[-1] for c in runs] == [r["id"] for r in args["jobs"]]
    assert len(calls) == (27 if existing else 28)  # prepare once, not every seed/rate
    assert all(p.read_bytes() == content for p, content in original.items())
    assert len(list(args["backup_dir"].glob("*.zip"))) == 27
    for path in args["backup_dir"].glob("*.zip"):
        meta = json.loads(path.with_suffix(".sha256.json").read_text())
        assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            assert len([n for n in archive.namelist() if n.endswith("/result.json")]) == 1


@pytest.mark.parametrize(
    "behavior,error",
    [
        ("fail", subprocess.CalledProcessError),
        ("pending", RuntimeError),
        ("interrupt", KeyboardInterrupt),
    ],
)
def test_failure_or_interrupt_stops_and_preserves_evidence(tmp_path, plan, behavior, error):
    args, calls = fixture_run(tmp_path, plan, behavior=behavior)
    with pytest.raises(error):
        execute_queue(**args)
    assert sum("--condition-id" in c for c in calls) == 1
    journal = json.loads(next(args["backup_dir"].glob("queue-*.json")).read_text())
    assert journal["status"] != "completed" and journal["verified"] == 0
    assert len(list(args["backup_dir"].glob("*.zip"))) == 1
    if behavior == "interrupt":
        assert (args["out"] / args["jobs"][0]["id"] / "running.lock").read_text() == "retained"


def test_wrong_result_stops_even_if_cli_returned_zero(tmp_path, plan):
    args, calls = fixture_run(tmp_path, plan, existing=True)
    result = args["out"] / args["jobs"][0]["id"] / "result.json"
    bad = json.loads(result.read_text())
    bad["identity"]["commit"] = "b" * 40
    result.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="identity"):
        execute_queue(**args)
    assert len(calls) == 1
    assert json.loads(result.read_text()) == bad


def test_cutoff_does_not_start_process_or_change_results(tmp_path, plan):
    args, calls = fixture_run(tmp_path, plan)
    args["now"] = lambda: 17.5 * 3600
    record = execute_queue(**args)
    assert record["status"] == "session_budget_reached" and calls == []


def test_cutoff_rechecked_after_data_preparation(tmp_path, plan):
    args, calls = fixture_run(tmp_path, plan)
    times = iter([0, 0, 18 * 3600, 18 * 3600])
    args["now"] = lambda: next(times)
    record = execute_queue(**args)
    assert record["status"] == "session_budget_reached"
    assert len(calls) == 1 and "--dataset" in calls[0]


def test_new_condition_limit(tmp_path, plan):
    args, calls = fixture_run(tmp_path, plan)
    record = execute_queue(**args, max_new=1)
    assert record["newly_executed"] == 1 and record["status"] == "session_budget_reached"
    assert sum("--condition-id" in c for c in calls) == 1


def test_archive_excludes_large_and_non_evidence_and_keeps_old_archive(tmp_path, plan):
    args, _ = fixture_run(tmp_path, plan, existing=True)
    row = args["jobs"][0]
    directory = args["out"] / row["id"]
    for name in ("training.json", "test-progress.json", "checkpoint.pt", "token.json", "raw.csv"):
        (directory / name).write_text("not evidence")
    first = export_evidence(args["out"], [row], args["backup_dir"], "fixture")
    old = first.read_bytes()
    second = export_evidence(args["out"], [row], args["backup_dir"], "fixture")
    assert first != second and first.read_bytes() == old
    with zipfile.ZipFile(second) as archive:
        assert set(archive.namelist()) == {"plan.json", f"{row['id']}/result.json"}


def test_real_cpu_process_success_and_failure(tmp_path):
    run_process(
        [sys.executable, "-c", "print('fixture only')"], tmp_path, tmp_path / "ok.log", "fixture"
    )
    assert "fixture only" in (tmp_path / "ok.log").read_text()
    with pytest.raises(subprocess.CalledProcessError):
        run_process(
            [sys.executable, "-c", "raise RuntimeError('fixture only')"],
            tmp_path,
            tmp_path / "fail.log",
            "fixture",
        )


@pytest.mark.parametrize(
    "name,family", [("42_ttm_main_queue", "ttm"), ("43_moirai1_main_queue", "moirai1")]
)
def test_notebooks_output_free_embedded_helper_and_original_engine(name, family):
    nb = json.loads((ROOT / f"notebooks/{name}.ipynb").read_text("utf-8"))
    sources = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    assert (
        sources["queue-tools"].strip()
        == (ROOT / "scripts/colab_queue.py").read_text("utf-8").strip()
    )
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), cell["id"], "exec")
    settings = {}
    exec(sources["settings"], settings)
    assert settings["FAMILY"] == family
    assert settings["COMMIT"] == "3e18305032484239966026f14395b903c9abf268"
    assert settings["SEEDS"] == [1729, 2718, 31415]
    assert "input(" not in sources["run"]
    assert '"pull"' not in sources["repository"]
