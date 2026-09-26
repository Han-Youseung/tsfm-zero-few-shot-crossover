"""Independent public-source preparation and a test-target-blind pilot loader.

The legacy bundle registry is never promoted. A separate prepared manifest pins
the exact variant, conversion, source checksum and output checksum.
"""

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import urllib.request
import uuid
import zipfile
from pathlib import Path

from tsfm_crossover.data.splits import chronological_split
from tsfm_crossover.tracking.atomic import write_json_atomic

ETT_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
ETT_HASHES = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
    "ETTm1": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
    "ETTm2": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
}
TARGETS = (
    *ETT_HASHES,
    "Electricity",
    "Traffic",
    "PEMS08",
    "Solar",
    "Wind",
    "AQShunyi",
    "Exchange",
    "Weather",
    "ZafNoo",
    "CzeLan",
)
URLS = {
    "Electricity": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
    "AQShunyi": "https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(url, dest, expected=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temp = dest.with_name(dest.name + "." + uuid.uuid4().hex + ".partial")
        # A bounded request; incomplete files cannot masquerade as successful downloads.
        with urllib.request.urlopen(url, timeout=60) as source, temp.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if expected and digest(temp) != expected:
            raise ValueError("download fingerprint mismatch; partial preserved")
        os.replace(temp, dest)
    if expected and digest(dest) != expected:
        raise ValueError("source fingerprint mismatch")
    return dest


def audit_wide(path):
    """Full-file integrity/QC is not test performance; statistics are train-only."""
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    stamps = pd.to_datetime(frame.iloc[:, 0], errors="raise")
    values = frame.iloc[:, 1:].to_numpy(dtype="float64")
    split = chronological_split(len(frame))
    train = values[: split.train.end]
    gaps = stamps.diff().dropna().dt.total_seconds()
    frequency = float(gaps.mode().iloc[0]) if len(gaps) else None
    std = np.nanstd(train, axis=0)
    return {
        "rows": len(frame),
        "channels": len(frame.columns) - 1,
        "channel_names": list(frame.columns[1:]),
        "frequency_seconds": frequency,
        "duplicate_timestamps": int(stamps.duplicated().sum()),
        "non_monotonic": int((gaps < 0).sum()),
        "irregular_intervals": int((gaps != frequency).sum()),
        "missing_values": int(np.isnan(values).sum()),
        "infinite_values": int(np.isinf(values).sum()),
        "train_constant_channels": int((std <= 1e-8).sum()),
        "statistics_scope": "train_only; full-file integrity counts only",
        "split": split.as_dict(),
        "sha256": digest(path),
    }


def prepare_one(name, root, cache, expected_sources=None):
    import pandas as pd

    expected_sources = expected_sources or {}
    dest = root / "data/prepared" / f"{name}.csv"
    sources = []
    if name in ETT_HASHES:
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{ETT_COMMIT}/ETT-small/{name}.csv"
        source = download(url, cache / f"{name}.csv", ETT_HASHES[name])
        sources.append({"url": url, "sha256": digest(source)})
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source, dest)
        if digest(dest) != ETT_HASHES[name]:
            raise ValueError("prepared ETT mismatch")
        variant, recipe, license_name = name + "__official_raw", "identity", "CC-BY-ND-4.0"
    elif name == "Weather":
        frames = []
        for half in ("a", "b"):
            url = f"https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2020{half}.zip"
            source = download(url, cache / f"weather-2020{half}.zip", expected_sources.get(url))
            sources.append({"url": url, "sha256": digest(source)})
            with zipfile.ZipFile(source) as bundle:
                entry = next(n for n in bundle.namelist() if n.endswith(".csv"))
                frames.append(pd.read_csv(bundle.open(entry), encoding="latin1"))
        frame = pd.concat(frames, ignore_index=True)
        frame.iloc[:, 0] = pd.to_datetime(frame.iloc[:, 0], dayfirst=True).astype(str)
        frame = frame.rename(columns={frame.columns[0]: "date"})
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False, lineterminator="\n", float_format="%.10g")
        if dest.exists() and digest(dest) != digest(temp):
            raise ValueError("Weather conversion changed; preserve previous artifact")
        os.replace(temp, dest)
        variant, recipe, license_name = (
            "Weather__mpi_roof_2020_raw",
            "v1: concatenate halves; preserve duplicate rows and all numeric channels",
            "CC-BY-4.0",
        )
    elif name in URLS:
        url = URLS[name]
        source = download(url, cache / f"{name}.zip", expected_sources.get(url))
        sources.append({"url": url, "sha256": digest(source)})
        with zipfile.ZipFile(source) as bundle:
            if name == "Electricity":
                entry = next(i for i in bundle.namelist() if i == "LD2011_2014.txt")
                # All 370 clients, no data-dependent channel filtering or imputation.
                # Hour-ending bins contain the four preceding 15-minute observations.
                parts = []
                with bundle.open(entry) as handle:
                    for chunk in pd.read_csv(
                        handle, sep=";", decimal=",", index_col=0, parse_dates=True, chunksize=20000
                    ):
                        parts.append(chunk.loc["2012-01-01 00:15:00":"2015-01-01 00:00:00"])
                grouped = pd.concat(parts).resample("1h", closed="right", label="right")
                if not (grouped.size() == 4).all():
                    raise ValueError("hourly conversion requires four observed quarter-hours")
                frame = grouped.mean()
                frame.index.name = "date"
                variant = "Electricity__uci370_hourly_2012_2014"
                recipe = (
                    "v1: all 370 clients; 2012-2014; "
                    "right-closed hour-ending mean kW; no imputation"
                )
            else:
                nested = next(i for i in bundle.namelist() if i.endswith(".zip"))
                with zipfile.ZipFile(io.BytesIO(bundle.read(nested))) as inside:
                    entry = next(
                        i for i in inside.namelist() if "Shunyi" in i and i.endswith(".csv")
                    )
                    frame = pd.read_csv(inside.open(entry))
                stamps = pd.to_datetime(frame[["year", "month", "day", "hour"]])
                cols = [
                    "PM2.5",
                    "PM10",
                    "SO2",
                    "NO2",
                    "CO",
                    "O3",
                    "TEMP",
                    "PRES",
                    "DEWP",
                    "RAIN",
                    "WSPM",
                ]
                frame = frame[cols].set_axis(stamps)
                frame.index.name = "date"
                variant = "AQShunyi__uci_raw_numeric11"
                recipe = "v1: Shunyi numeric 11 measures, no categorical wd, no NA imputation"
        license_name = "CC-BY-4.0"
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(".csv.tmp")
        frame.to_csv(temp, lineterminator="\n", float_format="%.10g")
        if dest.exists() and digest(dest) != digest(temp):
            raise ValueError("conversion changed; existing prepared output preserved")
        os.replace(temp, dest)
    else:
        raise ValueError("no approved automatic preparation recipe")
    qc = audit_wide(dest)
    issues = [
        k
        for k in (
            "duplicate_timestamps",
            "non_monotonic",
            "irregular_intervals",
            "missing_values",
            "infinite_values",
        )
        if qc[k]
    ]
    return {
        "name": name,
        "variant": variant,
        "path": dest.relative_to(root).as_posix(),
        "source_files": sources,
        "license": license_name,
        "recipe": recipe,
        "qc": qc,
        "status": "blocked_quality" if issues else "ready_with_warnings",
        "issues": issues,
        "bundle_status": "blocked",
        "redistribute_raw": False,
    }


def load_pilot_values(root, entry):
    """Read target values only through validation.end; hash may read all raw bytes."""
    return _load_values(root, entry, final_test=False)


def load_final_values(root, entry):
    """Final evaluation only; caller must first persist checkpoint selection."""
    return _load_values(root, entry, final_test=True)


def _load_values(root, entry, *, final_test):
    if entry["status"] != "ready_with_warnings" or "__bundle" in entry["variant"]:
        raise ValueError("unprepared/blocked dataset")
    path = Path(root) / entry["path"]
    if digest(path) != entry["qc"]["sha256"]:
        raise ValueError("prepared fingerprint mismatch")
    split = chronological_split(entry["qc"]["rows"])
    end = split.test.end if final_test else split.validation.end
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader)[1:]
        if columns != entry["qc"]["channel_names"]:
            raise ValueError("prepared channel order mismatch")
        from tsfm_crossover.data.missing import POLICY

        policy = entry.get("missing_policy")
        if policy not in (None, POLICY):
            raise ValueError("unsupported missing policy")
        values = [
            [float(v) if v.strip() else float("nan") for v in row[1:]]
            for row in itertools.islice(reader, end)
        ]
        if any(not math.isfinite(v) for row in values for v in row) and policy != POLICY:
            raise ValueError("missing values require an explicit approved policy")
    if len(values) != end or any(len(row) != len(columns) for row in values):
        raise ValueError("prepared row count mismatch")
    return values, tuple(columns), split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", default=[*ETT_HASHES, "Electricity", "AQShunyi"])
    p.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/prepared_data.json")
    )
    p.add_argument("--cache", type=Path, default=Path("data/source_cache"))
    p.add_argument("--verify-against", type=Path)
    args = p.parse_args()
    expected = json.loads(args.verify_against.read_text()) if args.verify_against else {}
    entries = {}
    for name in args.names:
        try:
            pins = {s["url"]: s["sha256"] for s in expected.get(name, {}).get("source_files", [])}
            item = prepare_one(name, Path.cwd(), args.cache, pins)
            if name in expected and item["qc"]["sha256"] != expected[name]["qc"]["sha256"]:
                raise ValueError("prepared output differs from pinned manifest")
            entries[name] = item
            print(name, item["status"], item["qc"]["rows"], item["qc"]["channels"], flush=True)
        except Exception as exc:
            entries[name] = {
                "name": name,
                "status": "preparation_failed",
                "error": type(exc).__name__,
            }
            print(name, type(exc).__name__, str(exc), flush=True)
    # Never overwrite a prior manifest on partial failures or a revised download.
    if args.output.exists():
        raise ValueError("use a new preparation manifest filename; previous evidence preserved")
    write_json_atomic(args.output, entries)


if __name__ == "__main__":
    main()
