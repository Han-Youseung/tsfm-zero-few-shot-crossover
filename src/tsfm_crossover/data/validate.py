"""Validate one local dataset without downloading it."""

import argparse
import json

from .cli_common import add_dataset_arguments, resolve_and_load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    args = parser.parse_args()
    try:
        _, loaded = resolve_and_load(args)
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(loaded.validation_report.as_dict(), indent=2))
    return 0 if loaded.validation_report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
