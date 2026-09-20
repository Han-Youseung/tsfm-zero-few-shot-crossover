"""Preview nested few-shot counts without writing a manifest."""

import argparse
import json

from .cli_common import add_dataset_arguments, resolve_and_load
from .sampling import build_sampling_manifest
from .splits import chronological_split
from .windows import generate_train_windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="compute and print only; no files are written"
    )
    args = parser.parse_args()
    try:
        _, loaded = resolve_and_load(args)
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")
    split = chronological_split(loaded.row_count, loaded.timestamps)
    windows = generate_train_windows(
        loaded.canonical_name, loaded.source_sha256, split.train, args.context_length, args.horizon
    )
    manifest = build_sampling_manifest(
        windows,
        dataset_sha256=loaded.source_sha256,
        split_config_hash=split.split_config_hash,
        seed=args.seed,
        code_commit_sha="working-tree-preview",
    )
    print(
        json.dumps(
            {
                "total_train_windows": manifest.total_train_windows,
                "selections": [
                    {
                        "requested_rate": item.requested_rate,
                        "selected_count": item.selected_count,
                        "effective_rate": item.effective_rate,
                    }
                    for item in manifest.selections
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
