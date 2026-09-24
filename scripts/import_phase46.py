"""Validate the two returned archives and retain historical evidence separately."""

import argparse
import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from tsfm_crossover.models.gpu_gate import (
    MODELS,
    GPUCondition,
    identity,
    summarize,
    validate_result,
)
from tsfm_crossover.models.selection import ModelSelectionGate
from tsfm_crossover.tracking.atomic import write_json_atomic

COMMIT = "1bb93b80b0c22c990cd2b5f7d9ff2a125b21e7c5"
OLD = "129ae0397723b8fd2c9d4686af5799aac3405c54"


def read_archive(path):
    """Read bounded JSON in memory; never extract archive-controlled paths."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if len(entries) > 30 or len(names) != len(set(names)):
            raise ValueError("duplicate/excessive ZIP entries")
        result = []
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in entry.filename
                or ":" in entry.filename
                or entry.file_size > 2_000_000
                or stat.S_ISLNK(entry.external_attr >> 16)
                or name.suffix != ".json"
            ):
                raise ValueError("unsafe ZIP entry")
            raw = archive.read(entry)
            result.append((name.name, json.loads(raw), hashlib.sha256(raw).hexdigest()))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ttm", type=Path)
    parser.add_argument("moirai", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    adopted, historical, installations = [], [], []
    archive_hashes = {}
    for archive in (args.ttm, args.moirai):
        archive_hashes[archive.name] = hashlib.sha256(archive.read_bytes()).hexdigest()
        for name, payload, raw_hash in read_archive(archive):
            if payload.get("kind") == "gpu_installation":
                if payload.get("execution_commit") not in (OLD, COMMIT):
                    raise ValueError("unexpected installation execution")
                installations.append((name, payload, raw_hash))
                continue
            commit = payload["identity"]["execution_commit"]
            if commit == COMMIT:
                adopted.append(validate_result(payload, root, COMMIT))
            elif commit == OLD and payload["identity"]["model_family"] == "ttm":
                # Historical schema is preserved, never upgraded or adopted into the new gate.
                row = GPUCondition.model_validate(payload)
                expected = identity(root, "ttm", row.identity["horizon"], OLD, 1)
                del expected["determinism"]
                if row.identity != expected:
                    raise ValueError("historical identity mismatch")
                historical.append((name, payload, raw_hash))
            else:
                raise ValueError("unexpected execution commit")
    summary = summarize(adopted)
    if (
        len(adopted) != 12
        or summary["gpu_compatibility"] != "passed"
        or summary["moirai_100_samples"] != "passed"
    ):
        raise ValueError("complete twelve-condition matrix required")
    if len(historical) != 4:
        raise ValueError("four historical TTM records required")
    paths = []
    for row in adopted:
        payload = row.model_dump()
        i = row.identity
        sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        path = (
            MODELS
            / "gpu_runs"
            / COMMIT
            / (f"{i['model_family']}-{i['horizon']}-{i['num_samples']}-{sha}.json")
        )
        if not path.exists():
            write_json_atomic(path, payload)
        elif json.loads(path.read_text()) != payload:
            raise ValueError("existing evidence differs")
        paths.append(str(path.as_posix()))
    # Reuse the actual CLI after validating the entire matrix, including duplicate checks.
    subprocess.run(
        [
            __import__("sys").executable,
            "-m",
            "tsfm_crossover.models.gpu_gate",
            "--expected-commit",
            COMMIT,
            "--import-results",
            *paths,
        ],
        check=True,
    )
    for name, payload, _sha in historical + installations:
        commit = payload.get(
            "execution_commit", payload.get("identity", {}).get("execution_commit")
        )
        path = MODELS / "gpu_archive" / commit / name
        if not path.exists():
            write_json_atomic(path, payload)
        elif json.loads(path.read_text()) != payload:
            raise ValueError("historical evidence differs")
    report = {
        "kind": "gpu_evidence_review",
        "execution_commit": COMMIT,
        "archive_sha256": archive_hashes,
        "conditions": paths,
        "historical_execution_commit": OLD,
        "historical_conditions": 4,
        "amp": "not_run",
        **summary,
    }
    write_json_atomic(MODELS / "gpu_evidence_review.json", report, overwrite=True)
    for family in ("ttm", "moirai1"):
        manifest = {
            "kind": "gpu_evidence",
            "schema_version": 1,
            "model_family": family,
            "execution_commit": COMMIT,
            "gpu_status": "passed",
            "fp32": "passed",
            "installation_status": "verified",
            "amp": "not_run",
            "conditions": [p for p in paths if Path(p).name.startswith(family + "-")],
            "protocol_frozen": False,
            "production_adapter_allowed": True,
            "scope": "ETTh1; batch 1; 7 channels; context 512; horizons 96/192/336/720",
        }
        write_json_atomic(MODELS / f"{family}_gpu_compatibility.json", manifest, overwrite=True)
    gate_path = MODELS / "model_selection_gate.json"
    gate = json.loads(gate_path.read_text())
    gate.update(gpu_compatibility="passed", production_adapter_allowed=True)
    candidate = gate["candidates"][0]
    candidate.update(decision_status="gpu_validated", gpu_validated=True)
    candidate["capability_results"]["gpu"] = "fp32_validated_four_horizons"
    candidate["evidence"]["gpu_evidence_review"] = str(
        (MODELS / "gpu_evidence_review.json").as_posix()
    )
    ModelSelectionGate.model_validate(gate)
    write_json_atomic(gate_path, gate, overwrite=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
