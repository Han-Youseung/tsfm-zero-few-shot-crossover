"""Leakage-safe model-independent data pipeline."""

from .loader import LoadedTimeSeries, load_dataset
from .registry import DatasetRegistry, DatasetSpec, load_registry
from .sampling import SamplingManifest, build_sampling_manifest
from .scaling import TrainOnlyStandardScaler
from .splits import SplitManifest, chronological_split
from .windows import WindowIndex, generate_rolling_windows, generate_train_windows

__all__ = [
    "DatasetRegistry",
    "DatasetSpec",
    "LoadedTimeSeries",
    "SamplingManifest",
    "SplitManifest",
    "TrainOnlyStandardScaler",
    "WindowIndex",
    "build_sampling_manifest",
    "chronological_split",
    "generate_rolling_windows",
    "generate_train_windows",
    "load_dataset",
    "load_registry",
]
