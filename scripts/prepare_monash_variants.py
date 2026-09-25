"""Independent TSF data conversion, not reuse of benchmark code or protocol."""

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.pilot_data import audit_wide, digest, download
from tsfm_crossover.tracking.atomic import write_json_atomic

SOURCES = {
    "Solar": (
        "4656144",
        "solar_10_minutes_dataset",
        "84c0de18383c911091a3cd274661b029",
        600,
        137,
        52560,
    ),
    "Traffic": (
        "4656132",
        "traffic_hourly_dataset",
        "1cf694f99f95700217845078b467fb24",
        3600,
        862,
        17544,
    ),
}


def parse_tsf(text):
    """Only the explicitly published two-attribute equal-length schema is accepted."""
    import numpy as np

    header, data = text.split("@data", 1)
    for declaration in (
        "@attribute series_name string",
        "@attribute start_timestamp date",
        "@missing false",
        "@equallength true",
    ):
        if declaration not in header:
            raise ValueError("unexpected TSF schema")
    names, starts, series = [], [], []
    for line in data.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, start, values = line.split(":", 2)
        names.append(name)
        starts.append(start)
        # Explicit float parsing rejects partial parses and missing markers.
        series.append([float(v) for v in values.split(",")])
    if not names or len(set(names)) != len(names) or len(set(starts)) != 1:
        raise ValueError("unaligned or duplicated series")
    values = np.array(series, dtype="float64").T
    if not np.isfinite(values).all():
        raise ValueError("nonfinite data")
    return names, starts[0], values


def prepare(name, root):
    import numpy as np
    import pandas as pd

    record, stem, md5, seconds, channels, rows = SOURCES[name]
    url = f"https://zenodo.org/records/{record}/files/{stem}.zip"
    archive = download(url, root / "data/source_cache" / f"{stem}.zip")
    with archive.open("rb") as handle:
        if hashlib.file_digest(handle, "md5").hexdigest() != md5:
            raise ValueError("published archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != [stem + ".tsf"]:
            raise ValueError("unexpected archive member")
        names, start, values = parse_tsf(bundle.read(stem + ".tsf").decode("utf-8"))
    if values.shape != (rows, channels):
        raise ValueError("published shape mismatch")
    author_pin = json.loads((root / "results/manifests/pilot/author_sources.json").read_text())[
        name
    ]
    author = download(
        author_pin["source_url"], root / "data/source_cache" / f"{name}.gz", author_pin["sha256"]
    )
    with gzip.open(author, "rt") as handle:
        reference = np.loadtxt(handle, delimiter=",")
    comparison = {
        "source_sha256": digest(author),
        "all_values_identical": bool(np.array_equal(values, reference)),
        "maximum_absolute_difference": float(np.max(np.abs(values - reference))),
        "comparison_atol": 1e-12,
        "comparison_rtol": 0.0,
        "within_serialization_tolerance": bool(np.allclose(values, reference, rtol=0, atol=1e-12)),
    }
    if not comparison["within_serialization_tolerance"]:
        raise ValueError("author matrix differs; investigate instead of silently substituting")
    frame = pd.DataFrame(values, columns=names)
    stamps = pd.date_range(
        pd.to_datetime(start, format="%Y-%m-%d %H-%M-%S"),
        periods=rows,
        freq=pd.Timedelta(seconds=seconds),
    )
    frame.insert(0, "date", stamps)
    path = root / "data/prepared" / f"{name}__monash.csv"
    # Deterministic in-memory serialization; refuse changes to an existing variant.
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format="%.10g", lineterminator="\n")
    payload = buffer.getvalue().encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("existing prepared variant differs")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
    qc = audit_wide(path)
    if any(
        qc[k]
        for k in (
            "duplicate_timestamps",
            "non_monotonic",
            "irregular_intervals",
            "missing_values",
            "infinite_values",
        )
    ):
        raise ValueError("converted data failed quality gate")
    return {
        "name": name,
        "variant": f"{name}__monash_{record}",
        "path": path.relative_to(root).as_posix(),
        "status": "ready_with_warnings",
        "license": "CC-BY-4.0 (Zenodo distribution metadata)",
        "source_files": [{"url": url, "sha256": digest(archive), "published_md5": md5}],
        "source_record": f"https://zenodo.org/records/{record}",
        "qc": qc,
        "author_matrix_comparison": comparison,
        "bundle_status": "blocked",
        "redistribute_raw": False,
        "recipe": (
            "Transpose all TSF series; preserve order and every value; expand published "
            "start/frequency; no normalization, imputation, filtering or trimming"
        ),
        "warnings": [
            "Published start has one-second offset, preserved exactly",
            "Distribution calendar, not independently verified observation timestamps",
            "Upstream aggregated research matrix, not unprocessed sensor data",
        ],
        "pilot_gpu_status": "not_run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/prepared_expansion.json")
    )
    args = parser.parse_args()
    root = Path.cwd()
    original = json.loads((root / "results/manifests/pilot/prepared_data.json").read_text())
    for name in SOURCES:
        original[name] = prepare(name, root)
        print(name, original[name]["qc"]["sha256"], original[name]["author_matrix_comparison"])
    if args.output.exists():
        if stable_hash(json.loads(args.output.read_text())) != stable_hash(original):
            raise ValueError("refusing to replace prior manifest")
    else:
        write_json_atomic(args.output, original)


if __name__ == "__main__":
    main()
