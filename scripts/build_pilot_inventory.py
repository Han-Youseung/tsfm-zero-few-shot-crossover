"""Generate small 14-dataset readiness and structural risk tables from real QC."""

import csv
import json
from pathlib import Path

from tsfm_crossover.data.pilot_data import TARGETS
from tsfm_crossover.experiments.pilot_config import token_risk
from tsfm_crossover.tracking.atomic import write_csv_atomic, write_json_atomic


def main():
    root = Path("results/manifests/pilot")
    prepared = json.loads((root / "prepared_data.json").read_text())
    prepared |= json.loads((root / "weather_preparation.json").read_text())
    authors = json.loads((root / "author_sources.json").read_text())
    with Path("results/manifests/datasets/validation_summary.csv").open(encoding="utf-8") as f:
        legacy = {r["dataset"]: r for r in csv.DictReader(f)}
    rows, risks = [], []
    for name in TARGETS:
        item, old = prepared.get(name, {}), legacy[name]
        qc, source = item.get("qc", {}), authors.get(name, {})
        if item:
            shape = [qc["rows"], qc["channels"]]
        else:
            shape = source.get("shape", [int(old["observed_rows"]), int(old["observed_channels"])])
        row = {
            "dataset": name,
            "variant": item.get("variant", source.get("variant", "unresolved")),
            "status": item.get("status", source.get("status", "blocked_source_identity")),
            "public_raw_or_author_acquired": bool(item or source.get("local_acquired")),
            "source": (item.get("source_files") or [{"url": source.get("source_url")}])[0]["url"],
            "fingerprint": qc.get("sha256", source.get("sha256")),
            "rows": shape[0],
            "channels": shape[1],
            "shape_basis": "acquired variant" if item or source else "legacy bundle audit only",
            "frequency_seconds": qc.get("frequency_seconds", None),
            "legacy_frequency_seconds_not_verified_for_new_source": old[
                "observed_frequency_seconds"
            ],
            "duplicate_timestamps": qc.get("duplicate_timestamps"),
            "non_monotonic": qc.get("non_monotonic"),
            "irregular_intervals": qc.get("irregular_intervals"),
            "missing_values": qc.get("missing_values", source.get("nonfinite")),
            "train_constant_channels": qc.get("train_constant_channels"),
            "preprocessing": item.get("recipe", "not fully traceable; blocked"),
            "license": item.get("license", source.get("license", "pending")),
            "legacy_bundle_status": "blocked",
        }
        rows.append(row)
        for horizon in (96, 192, 336, 720):
            risks.append(
                {
                    "dataset": name,
                    "horizon": horizon,
                    "channel_basis": row["shape_basis"],
                    **token_risk(shape[1], horizon),
                }
            )
    write_json_atomic(root / "dataset_readiness.json", {"datasets": rows, "target_count": 14})
    write_csv_atomic(root / "dataset_readiness.csv", rows, fieldnames=list(rows[0]))
    write_json_atomic(
        root / "structural_risks.json",
        {
            "method": "static official-code review, not GPU execution",
            "moirai_code_commit": "cfd46d4510ed8896f263116f32928eede05b0a75",
            "conditions": risks,
        },
    )
    print("14 readiness rows, 56 static risk conditions; no bundle released")


if __name__ == "__main__":
    main()
