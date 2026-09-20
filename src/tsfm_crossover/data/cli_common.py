"""Shared CLI setup."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .loader import LoadedTimeSeries, load_dataset
from .registry import DatasetSpec, load_registry


def add_dataset_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--registry", default="configs/datasets/registry.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", default=".")


def resolve_and_load(args: Namespace) -> tuple[DatasetSpec, LoadedTimeSeries]:
    registry_path = Path(args.registry)
    spec = load_registry(registry_path).resolve(args.dataset)
    return spec, load_dataset(spec, args.root)
