"""Bounded phase-5 evidence import, not a rerun of successful GPU conditions."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from tsfm_crossover.models.gpu_gate import DATA_SHA
from tsfm_crossover.tracking.atomic import write_json_atomic

COMMIT = "689a284dac1c2b688b3efb6c1e2141fd966068d6"
CHECKS = {
    "finite_loss_gradients",
    "full_optimizer_coverage",
    "independent_pretrained",
    "parameters_updated",
    "prediction_rng_preserved",
    "restore_hash",
    "restore_optimizer",
    "restore_prediction",
    "restore_rng",
    "restore_step",
    "shape_finite_repeat",
    "zero_shot_no_update",
}


def validate(records):
    seen = set()
    for d in records:
        cfg = d["identity"]["config"]
        key = (cfg["family"], cfg["horizon"])
        if key in seen:
            raise ValueError("duplicate adapter condition")
        seen.add(key)
        if not (
            d["identity"]["execution_commit"] == COMMIT
            and d["identity"]["device"] == "cuda"
            and d["status"] == "passed"
            and not d["dirty_worktree"]
            and not d["test_split_used"]
            and cfg["dataset_fingerprint"] == DATA_SHA
            and CHECKS.issubset(d["checks"])
            and all(d["checks"][k] is True for k in CHECKS)
            and d["point_shape"] == [2, cfg["horizon"], 7]
            and d["metadata"]["actual_optimizer_steps"] == 1
        ):
            raise ValueError("invalid/incomplete adapter evidence")
    if seen != {(m, h) for m in ("ttm", "moirai1") for h in (96, 192, 336, 720)}:
        raise ValueError("eight required conditions missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/manifests/adapters/gpu"))
    args = parser.parse_args()
    records, archives = [], []
    for path in args.archives:
        archives.append(
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        with zipfile.ZipFile(path) as bundle:
            if len(bundle.infolist()) != 4:
                raise ValueError("unexpected ZIP members")
            for entry in bundle.infolist():
                if entry.file_size > 2_000_000 or not entry.filename.endswith(".json"):
                    raise ValueError("unexpected ZIP entry")
                records.append(json.loads(bundle.read(entry)))
    validate(records)
    for d in records:
        cfg = d["identity"]["config"]
        dest = args.output / f"{cfg['family']}-h{cfg['horizon']}.json"
        if dest.exists():
            if json.loads(dest.read_text()) != d:
                raise ValueError("existing evidence differs")
        else:
            write_json_atomic(dest, d)
    review = {
        "execution_commit": COMMIT,
        "conditions": 8,
        "status": "passed",
        "archives": archives,
        "protocol_frozen": False,
        "scope": "ETTh1, batch 2, seven channels, FP32, one training step",
        "gpus": sorted({d["metadata"]["gpu"] for d in records}),
        "moirai_adapter_100_samples": "pending",
        "pilot_status": "not_run",
    }
    dest = args.output / "review.json"
    if not dest.exists():
        write_json_atomic(dest, review)
    print(json.dumps(review))


if __name__ == "__main__":
    main()
