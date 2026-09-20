from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tsfm_crossover.config.loader import load_experiment_config
from tsfm_crossover.config.schema import ExperimentConfig


def test_yaml_config_loading(tmp_path: Path, valid_config_payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(valid_config_payload), encoding="utf-8")
    config = load_experiment_config(path)
    assert config.dataset.name == "fixture"
    assert config.split.train == 0.6


def test_unknown_field_is_rejected(valid_config_payload):
    valid_config_payload["dataset"]["surprise"] = True
    with pytest.raises(ValidationError, match="surprise"):
        ExperimentConfig.model_validate(valid_config_payload)


def test_split_must_sum_to_one(valid_config_payload):
    valid_config_payload["split"]["test"] = 0.3
    with pytest.raises(ValidationError, match="must sum to 1"):
        ExperimentConfig.model_validate(valid_config_payload)


def test_zero_shot_disables_training_and_sampling(valid_config_payload):
    valid_config_payload["fine_tuning"]["adaptation_mode"] = "zero_shot"
    valid_config_payload["sampling"]["requested_rate"] = 0
    valid_config_payload["fine_tuning"]["optimizer"] = "not-allowed"
    with pytest.raises(ValidationError, match="zero-shot"):
        ExperimentConfig.model_validate(valid_config_payload)


def test_few_shot_requires_positive_sampling_rate(valid_config_payload):
    valid_config_payload["sampling"]["requested_rate"] = 0
    with pytest.raises(ValidationError, match="few-shot"):
        ExperimentConfig.model_validate(valid_config_payload)


def test_training_source_must_be_train(valid_config_payload):
    valid_config_payload["sampling"]["source_split"] = "validation"
    with pytest.raises(ValidationError, match="train split"):
        ExperimentConfig.model_validate(valid_config_payload)


def test_storage_roots_must_not_overlap(valid_config_payload):
    valid_config_payload["storage"]["checkpoints_root"] = "results/checkpoints"
    with pytest.raises(ValidationError, match="must not overlap"):
        ExperimentConfig.model_validate(valid_config_payload)


def test_source_variant_must_match_dataset_namespace(valid_config_payload):
    valid_config_payload["dataset"]["source_variant"] = "other__official_raw"
    with pytest.raises(ValidationError, match="namespaced by dataset"):
        ExperimentConfig.model_validate(valid_config_payload)
