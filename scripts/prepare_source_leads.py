"""Acquire author-distributed matrices for provenance work, not automatic pilot release."""

import gzip
import json
from pathlib import Path

from tsfm_crossover.data.pilot_data import digest, download
from tsfm_crossover.tracking.atomic import write_json_atomic

AUTHOR = "7f402f185cc2435b5e66aed13a3b560ed142e023"
SOURCES = {
    "Solar": f"https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/{AUTHOR}/solar-energy/solar_AL.txt.gz",
    "Traffic": f"https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/{AUTHOR}/traffic/traffic.txt.gz",
    "Exchange": f"https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/{AUTHOR}/exchange_rate/exchange_rate.txt.gz",
    "PEMS08": "https://raw.githubusercontent.com/Davidham3/ASTGCN/d5d40645fa45471f2b473fc49a29ec3f0a993c6d/data/PEMS08/pems08.npz",
}


def main():
    import numpy as np

    output = Path("results/manifests/pilot/author_sources.json")
    if output.exists():
        print("existing source metadata retained")
        return
    records = {}
    for name, url in SOURCES.items():
        try:
            path = download(
                url, Path("data/source_cache") / (name + (".npz" if name == "PEMS08" else ".gz"))
            )
            if name == "PEMS08":
                with np.load(path, allow_pickle=False) as item:
                    values = item["data"]
                    shape = list(values.shape)
                    missing = int((~np.isfinite(values)).sum())
            else:
                with gzip.open(path, "rt") as handle:
                    values = np.loadtxt(handle, delimiter=",")
                shape = list(values.shape)
                missing = int((~np.isfinite(values)).sum())
            records[name] = {
                "source_url": url,
                "sha256": digest(path),
                "shape": shape,
                "nonfinite": missing,
                "timestamps": "not supplied in matrix",
                "variant": name + "__author_matrix_unreleased",
                "local_acquired": True,
                "status": "blocked_provenance_or_license",
                "license": "permission_verification_pending",
                "remaining": (
                    "verify timestamps, full preprocessing and data-use terms; "
                    "no automatic replacement"
                ),
            }
            print(name, shape, flush=True)
        except Exception as exc:
            records[name] = {
                "source_url": url,
                "status": "download_failed",
                "error": type(exc).__name__,
            }
            print(name, type(exc).__name__, flush=True)
    write_json_atomic(output, records)
    print(json.dumps({k: v["status"] for k, v in records.items()}))


if __name__ == "__main__":
    main()
