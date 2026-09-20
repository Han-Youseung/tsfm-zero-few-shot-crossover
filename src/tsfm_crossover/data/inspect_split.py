"""Inspect chronological splits and rolling-window counts."""

import argparse
import json

from .cli_common import add_dataset_arguments, resolve_and_load
from .splits import chronological_split
from .windows import generate_rolling_windows, generate_train_windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    try:
        _, loaded = resolve_and_load(args)
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")
    split = chronological_split(loaded.row_count, loaded.timestamps)
    fingerprint = loaded.source_sha256
    counts = {
        "train": len(
            generate_train_windows(
                loaded.canonical_name, fingerprint, split.train, args.context_length, args.horizon
            )
        ),
        "validation": len(
            generate_rolling_windows(
                loaded.canonical_name,
                fingerprint,
                "validation",
                split.validation,
                args.context_length,
                args.horizon,
            )
        ),
        "test": len(
            generate_rolling_windows(
                loaded.canonical_name,
                fingerprint,
                "test",
                split.test,
                args.context_length,
                args.horizon,
            )
        ),
    }
    print(json.dumps({"split": split.as_dict(), "window_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
