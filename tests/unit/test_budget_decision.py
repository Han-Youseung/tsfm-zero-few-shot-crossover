import copy
from pathlib import Path

import pytest
import yaml

from tsfm_crossover.experiments.budget_review import replay_stopping
from tsfm_crossover.experiments.pilot_config import PilotConfig
from tsfm_crossover.experiments.study_preflight import StudyPlan


def stopping_fixture():
    return {
        "stopping_step": 400,
        "best_validation_step": 100,
        "metadata": {"window_exposures": 400},
        "data": {"selected_train_windows": 20},
        "equivalent_epochs": 20,
        "loop": {
            "best_metric": 1.0,
            "bad_evals": 3,
            "history": [
                {"step": s, "metrics": {"macro": {"normalized_mae": v}}}
                for s, v in [(100, 1.0), (200, 1.1), (300, 1.2), (400, 1.3)]
            ],
            "loss_history": [{"step": s, "actual_batch_size": 1} for s in range(1, 401)],
        },
    }


def test_early_stopping_replay():
    cfg = PilotConfig.model_validate(
        yaml.safe_load(Path("configs/pilot/budget_confirmation.yaml").read_text())
    )
    assert replay_stopping(stopping_fixture(), cfg) == "early_stopping"


@pytest.mark.parametrize(
    "field,value",
    [("best_validation_step", 400), ("stopping_step", 500), ("equivalent_epochs", 10)],
)
def test_stop_or_exposure_tamper(field, value):
    cfg = PilotConfig.model_validate(
        yaml.safe_load(Path("configs/pilot/budget_confirmation.yaml").read_text())
    )
    r = stopping_fixture()
    r[field] = value
    with pytest.raises(ValueError):
        replay_stopping(r, cfg)


def test_design_locked_to_no_execution():
    raw = yaml.safe_load(Path("configs/study/preexperiment.yaml").read_text())
    cfg = StudyPlan.model_validate(raw)
    assert cfg.max_optimizer_steps == 1000 and len(cfg.sampling_rates) == 8
    for field in ("test_evaluation", "protocol_frozen", "main_experiment_allowed"):
        changed = copy.deepcopy(raw)
        changed[field] = True
        with pytest.raises(ValueError):
            StudyPlan.model_validate(changed)
