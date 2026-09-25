import copy
from pathlib import Path

import pytest
import yaml

from tsfm_crossover.experiments.pilot_config import PilotConfig, execution_plan
from tsfm_crossover.experiments.primary_selection import PRIMARY, build, select


def confirmation():
    return yaml.safe_load(Path("configs/pilot/budget_confirmation.yaml").read_text())


def test_primary_evidence_and_count():
    prepared, decision = build(Path.cwd())
    assert tuple(prepared) == PRIMARY and "Traffic" not in prepared
    assert decision["verified_common_feasibility_conditions"] == 64
    assert decision["dry_run_design"]["total_conditions"] == 8 * 2 * 4 * 9 * 3
    assert decision["main_experiment_allowed"] is False
    assert decision["traffic"]["retry_80gb"]["experiment_started"] is False


def test_missing_evidence_cannot_promote():
    prepared, _ = build(Path.cwd())
    with pytest.raises(ValueError, match="missing common feasibility"):
        select(prepared, [{"test_evaluation": False, "protocol_frozen": False, "conditions": []}])


def test_confirmation_has_exactly_four_runs():
    prepared, _ = build(Path.cwd())
    config = PilotConfig.model_validate(confirmation())
    plan = execution_plan(config, prepared, "a" * 40)
    assert len(plan["conditions"]) == 4
    assert all(r["kind"] == "stability" and r["horizon"] == 96 for r in plan["conditions"])
    assert {r["dataset"] for r in plan["conditions"]} == {"ETTh1", "Electricity"}
    assert plan["planned_groups"] == 4 and plan["blocked_groups"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_steps", 1001),
        ("training_batch", 2),
        ("horizons", [96, 720]),
        ("test_evaluation", True),
        ("protocol_frozen", True),
        ("learning_rates", {"ttm": [1e-5, 1e-4], "moirai1": [5e-6]}),
    ],
)
def test_confirmation_is_not_expanded_search(field, value):
    config = copy.deepcopy(confirmation())
    config[field] = value
    with pytest.raises(ValueError):
        PilotConfig.model_validate(config)


def test_historical_pilot_bound_unchanged():
    with pytest.raises(ValueError, match="explicit budget-confirmation"):
        PilotConfig(max_steps=201)
    assert PilotConfig().max_steps == 200
