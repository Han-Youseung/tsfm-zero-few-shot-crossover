"""Prepare approved Weather/Tetouan variants without overwriting source data."""

import argparse
import io
import json
import os
import uuid
import zipfile
from pathlib import Path

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.missing import POLICY
from tsfm_crossover.data.pilot_data import audit_wide, digest, download, prepare_one
from tsfm_crossover.tracking.atomic import write_json_atomic

TETOUAN_URL = "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip"
TETOUAN_SHA = "3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe"


def regularize(frame, frequency, sentinels=None):
    """Only identical rows may be removed. No values are imputed here."""
    import pandas as pd

    before = len(frame)
    deduplicated = frame.drop_duplicates().copy()
    stamps = pd.to_datetime(deduplicated.iloc[:, 0], errors="raise")
    conflicts = stamps.duplicated(keep=False)
    audit = {
        "rows_before": before,
        "rows_after_exact_dedup": len(deduplicated),
        "exact_duplicate_rows": before - len(deduplicated),
        "timestamp_conflict_rows": int(conflicts.sum()),
    }
    if conflicts.any():
        return None, {
            **audit,
            "status": "blocked_timestamp_conflict",
            "conflict_timestamps": list(
                deduplicated.loc[conflicts].iloc[:, 0].astype(str).unique()
            ),
        }
    if not stamps.is_monotonic_increasing:
        return None, {**audit, "status": "blocked_timestamp_order"}
    values = deduplicated.iloc[:, 1:].copy()
    values.index = stamps
    sentinel_counts = {}
    for column, sentinel in (sentinels or {}).items():
        sentinel_counts[column] = int(values[column].eq(sentinel).sum())
        values[column] = values[column].mask(values[column].eq(sentinel))
    grid = pd.date_range(stamps.iloc[0], stamps.iloc[-1], freq=frequency)
    if not values.index.isin(grid).all():
        return None, {**audit, "status": "blocked_off_grid_timestamp"}
    values = values.reindex(grid)
    values.index.name = "date"
    audit.update(
        status="prepared",
        rows_after_grid=len(values),
        inserted_timestamps=len(values) - len(deduplicated),
        invalid_sentinel_counts=sentinel_counts,
        missing_cells=int(values.isna().sum().sum()),
        leading_missing_cells=int(values.isna().cumprod().sum().sum()),
        channel_missing_rates=values.isna().mean().to_dict(),
        excluded_channels=[],
    )
    return values.reset_index(), audit


def persist_frame(frame, path):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format="%.10g", lineterminator="\n")
    payload = buffer.getvalue().encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing prepared data differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def prepare(name, root, cache):
    import pandas as pd

    if name == "Weather":
        original = json.loads(
            (root / "results/manifests/pilot/weather_preparation.json").read_text()
        )[name]
        raw = root / original["path"]
        if not raw.exists():
            prepare_one(
                name, root, cache, {s["url"]: s["sha256"] for s in original["source_files"]}
            )
        if digest(raw) != original["qc"]["sha256"]:
            raise ValueError("Weather source fingerprint mismatch")
        frame = pd.read_csv(raw)
        # Explicit invalid-value rule, not an assumption that all negative values are missing.
        sentinels = {"wv (m/s)": -9999, "max. PAR (µmol/m²/s)": -9999, "CO2 (ppm)": -9999}
        sources = original["source_files"]
        variant = "Weather__mpi2020_quality_v1"
    else:
        raw = download(TETOUAN_URL, cache / "tetouan.zip", TETOUAN_SHA)
        with zipfile.ZipFile(raw) as archive:
            frame = pd.read_csv(archive.open("Tetuan City power consumption.csv"))
        frame[frame.columns[0]] = pd.to_datetime(frame.iloc[:, 0], format="%m/%d/%Y %H:%M")
        sentinels = {}
        sources = [{"url": TETOUAN_URL, "sha256": TETOUAN_SHA}]
        variant = "Tetouan__uci849_numeric8"
    prepared, audit = regularize(frame, "10min", sentinels)
    if prepared is None:
        return {"name": name, "status": audit["status"], "audit": audit}
    path = root / "data/prepared" / f"{variant}.csv"
    persist_frame(prepared, path)
    return {
        "name": name,
        "variant": variant,
        "path": path.relative_to(root).as_posix(),
        "status": "ready_with_warnings",
        "missing_policy": POLICY,
        "qc": audit_wide(path),
        "audit": audit,
        "source_files": sources,
        "license": "CC-BY-4.0",
        "redistribute_raw": False,
        "recipe": (
            "Exact duplicate removal, reject timestamp conflicts, explicit invalid-value masking, "
            "complete 10-minute grid; targets remain missing"
        ),
        "gpu_status": "pending_gpu",
        "research_protocol_frozen": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/source_cache"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/prepared_primary.json")
    )
    args = parser.parse_args()
    root = Path.cwd()
    entries = json.loads((root / "results/manifests/pilot/prepared_expansion.json").read_text())
    entries["AQShunyi"]["status"] = "excluded_primary_official_loss_masking"
    for name in ("Weather", "Tetouan"):
        entries[name] = prepare(name, root, args.cache)
        print(name, entries[name]["status"], entries[name].get("audit"), flush=True)
    if args.output.exists():
        if stable_hash(json.loads(args.output.read_text())) != stable_hash(entries):
            raise ValueError("refusing to overwrite previous manifest")
    else:
        write_json_atomic(args.output, entries)


if __name__ == "__main__":
    main()
