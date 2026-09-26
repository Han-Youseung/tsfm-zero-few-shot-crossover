"""Reuse approved recipes; never revise dataset variants or canonical manifests."""

import argparse
import json
from pathlib import Path

from prepare_monash_variants import prepare as prepare_monash
from prepare_primary_candidates import prepare as prepare_quality

from tsfm_crossover.data.pilot_data import digest, prepare_one
from tsfm_crossover.experiments.main_plan import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    pins = json.loads((root / "results/manifests/pilot/prepared_primary9.json").read_text())
    entry = pins[args.dataset]
    dest = root / entry["path"]
    if not dest.exists():
        if args.dataset in {"Solar", "Traffic"}:
            prepare_monash(args.dataset, root)
        elif args.dataset in {"Weather", "Tetouan"}:
            prepare_quality(args.dataset, root, root / "data/source_cache")
        else:
            prepare_one(
                args.dataset,
                root,
                root / "data/source_cache",
                {s["url"]: s["sha256"] for s in entry["source_files"]},
            )
    if digest(dest) != entry["qc"]["sha256"]:
        raise ValueError("prepared data differs from frozen primary9; do not substitute")
    print(args.dataset, "canonical fingerprint verified; no test metric computed", flush=True)


if __name__ == "__main__":
    main()
