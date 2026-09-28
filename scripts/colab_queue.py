# Notebook-embedded orchestration only; the pinned main-study engine is unchanged.
# Standard library only. Embedded verbatim into notebooks 42/43 so they can run
# the historical execution commit without upgrading its source.

import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_queue_json(path, record):
    """Replace only this invocation's queue journal, never experiment evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_queue(plan, family, datasets, horizons, seeds):
    rows = plan["conditions"]
    available = list(dict.fromkeys(r["dataset"] for r in rows if r["family"] == family))
    if datasets == ["ALL"]:
        datasets = available
    for values, allowed, label in (
        (datasets, available, "datasets"),
        (horizons, plan["config"]["horizons"], "horizons"),
        (seeds, plan["config"]["seeds"], "seeds"),
    ):
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f"Invalid {label}; allowed: {allowed}")
    selected = [
        r
        for r in rows
        if r["family"] == family
        and r["dataset"] in datasets
        and r["horizon"] in horizons
        and r["seed"] in seeds
    ]
    rates = [0.0, *plan["config"]["sampling_rates"]]
    expected = {(d, h, s, r) for d in datasets for h in horizons for s in seeds for r in rates}
    actual = {(r["dataset"], r["horizon"], r["seed"], r["rate"]) for r in selected}
    if actual != expected or len(selected) != len(expected):
        raise ValueError("Incomplete or duplicate approved grid")
    if len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate condition IDs")
    for row in selected:
        name = row["id"]
        if not name or name in (".", "..") or any(c in name for c in "/\\:"):
            raise ValueError("Unsafe condition ID")
    return sorted(
        selected,
        key=lambda r: (
            datasets.index(r["dataset"]),
            horizons.index(r["horizon"]),
            seeds.index(r["seed"]),
            r["rate"],
        ),
    )


def evidence_files(out, rows):
    """Small evidence only; no optimizer checkpoints, samples, raw data or caches."""
    yield Path(out) / "plan.json"
    allowed = {"result.json", "runtime.json", "provenance.json", "selection.json", "started.json"}
    for row in rows:
        for path in sorted((Path(out) / row["id"]).glob("*.json")):
            if path.name in allowed or path.name.startswith(("failure-", "pending-")):
                yield path


def export_evidence(out, rows, destination, label):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{label}-{uuid.uuid4().hex}.zip"
    temporary = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in evidence_files(out, rows):
            if path.is_symlink() or path.stat().st_size > 2_000_000:
                raise ValueError(f"Unexpected evidence file: {path.name}; original retained")
            archive.write(path, path.relative_to(out).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity failure; partial retained")
    os.replace(temporary, target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    atomic_queue_json(
        target.with_suffix(".sha256.json"),
        {
            "file": target.name,
            "sha256": digest.hexdigest(),
            "bytes": target.stat().st_size,
            "condition_ids": [r["id"] for r in rows],
        },
    )
    return target


def run_process(command, cwd, log_path, label, heartbeat_seconds=30):
    """Show liveness/time, not fabricated training percentage or ETA."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = child.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    print(
                        f"  {label}: process active, elapsed {time.monotonic() - started:.0f}s",
                        flush=True,
                    )
                    with log_path.open("rb") as tail:
                        tail.seek(max(0, log_path.stat().st_size - 600))
                        text = tail.read().decode("utf-8", errors="replace").strip()
                    if text:
                        print("  latest log:", text[-400:], flush=True)
        except BaseException:
            # Stop only the subprocess this cell created. Never remove its lock/checkpoints.
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    if code:
        with log_path.open("rb") as tail:
            tail.seek(max(0, log_path.stat().st_size - 4000))
            print(tail.read().decode("utf-8", errors="replace"), flush=True)
        raise subprocess.CalledProcessError(code, command)


def execute_queue(
    *,
    jobs,
    family,
    commit,
    root,
    out,
    python,
    backup_dir,
    local_logs,
    session_started,
    wall_hours=18,
    reserve_minutes=30,
    max_new=108,
    process=run_process,
    now=time.monotonic,
):
    """Sequential queue around the original one-condition CLI. Fail closed."""
    if not jobs or wall_hours <= 0 or not 0 <= reserve_minutes < wall_hours * 60 or max_new < 1:
        raise ValueError("Invalid queue/session limits")
    out, root = Path(out), Path(root)
    plan = read_json(out / "plan.json")
    if plan["commit"] != commit or any(
        r["family"] != family or r not in plan["conditions"] for r in jobs
    ):
        raise ValueError("Queue differs from pinned plan/family/commit")
    if len({r["id"] for r in jobs}) != len(jobs):
        raise ValueError("Duplicate jobs")
    run_id = uuid.uuid4().hex
    journal = Path(backup_dir) / f"queue-{run_id}.json"
    local_logs = Path(local_logs) / run_id
    record = dict(
        schema_version=1,
        wrapper_version="queue-v1",
        commit=commit,
        family=family,
        created_at=datetime.now(UTC).isoformat(),
        status="running",
        total=len(jobs),
        condition_ids=[r["id"] for r in jobs],
        verified=0,
        newly_executed=0,
        events=[],
    )
    prepared = set()
    base = [
        str(python),
        "-u",
        "-m",
        "tsfm_crossover.experiments.main_study",
        "--output",
        str(out),
        "--expected-commit",
        commit,
    ]
    atomic_queue_json(journal, record)

    def budget_reached():
        return (
            now() - session_started >= wall_hours * 3600 - reserve_minutes * 60
            or record["newly_executed"] >= max_new
        )

    try:
        for index, row in enumerate(jobs, 1):
            result = out / row["id"] / "result.json"
            existing = result.exists() and read_json(result).get("status") == "completed"
            # No next process after cutoff; completed records use the original validator too.
            if budget_reached():
                record["status"] = "session_budget_reached"
                print("Session cutoff: no next condition. Resume with the same configuration.")
                break
            started = now()
            event = {"id": row["id"], "status": "running", "existing_candidate": existing}
            record["events"].append(event)
            atomic_queue_json(journal, record)
            print(
                f"[{index}/{len(jobs)}] {'VERIFY' if existing else 'RUN'} {row['id']}", flush=True
            )
            try:
                if not existing and row["dataset"] not in prepared:
                    process(
                        [
                            str(python),
                            "-u",
                            "scripts/prepare_study_data.py",
                            "--dataset",
                            row["dataset"],
                        ],
                        root,
                        local_logs / f"prepare-{row['dataset']}.log",
                        "data integrity",
                    )
                    prepared.add(row["dataset"])
                    if budget_reached():
                        event["status"] = "not_started_session_budget"
                        record["status"] = "session_budget_reached"
                        break
                process(
                    base + ["--condition-id", row["id"]],
                    root,
                    local_logs / f"{row['id']}.log",
                    row["id"],
                )
                if not result.exists():
                    raise RuntimeError(
                        "No completed result: inspect pending resource evidence; queue stopped"
                    )
                value = read_json(result)
                if (
                    value.get("status") != "completed"
                    or value.get("identity", {}).get("commit") != commit
                    or value.get("identity", {}).get("condition") != row
                ):
                    raise ValueError("Completed result identity/status mismatch")
                event["status"] = "verified_existing" if existing else "completed"
                record["verified"] += 1
                record["newly_executed"] += not existing
            except BaseException as exc:
                event["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                event["error_type"] = type(exc).__name__
                event["error"] = str(exc)[:600]
                raise
            finally:
                event["wall_seconds"] = now() - started
                # Save state first. A backup error stops this queue but never overwrites evidence.
                atomic_queue_json(journal, record)
                archive = export_evidence(out, [row], backup_dir, row["id"])
                event["backup"] = archive.name
                atomic_queue_json(journal, record)
            print(
                f"  {event['status']} ({event['wall_seconds']:.1f}s); Drive ZIP: {archive.name}",
                flush=True,
            )
        else:
            record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)[:600]
        raise
    finally:
        record["updated_at"] = datetime.now(UTC).isoformat()
        atomic_queue_json(journal, record)
        print(
            f"Queue {record['status']}: verified {record['verified']}/{len(jobs)}; "
            f"journal: {journal}",
            flush=True,
        )
    return record
