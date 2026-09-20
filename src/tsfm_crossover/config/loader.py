"""YAML configuration loader with strict schema validation."""

from pathlib import Path

import yaml

from .schema import ExperimentConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("experiment YAML root must be a mapping")
    return ExperimentConfig.model_validate(payload)
