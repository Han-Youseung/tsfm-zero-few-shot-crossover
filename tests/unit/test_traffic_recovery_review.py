import copy
import json
from pathlib import Path

import pytest

from tsfm_crossover.experiments.traffic_recovery_review import validate_envelope


@pytest.fixture
def returned():
    folder = Path("results/manifests/pilot")
    summary = json.loads((folder / "traffic_recovery_review.json").read_text(encoding="utf-8"))
    data = json.loads((folder / "prepared_primary.json").read_text(encoding="utf-8"))
    return copy.deepcopy(summary["conditions"][1]), data["Traffic"]


def test_returned_traffic_envelope(returned):
    record, data = returned
    validate_envelope(record, "Traffic", data)


@pytest.mark.parametrize("mutation", ["running", "commit", "check", "cpu", "bf16", "test"])
def test_wrong_or_incomplete_evidence_rejected(returned, mutation):
    record, data = returned
    if mutation == "running":
        record["status"] = "running"
    elif mutation == "commit":
        record["identity"]["commit"] = "old"
    elif mutation == "check":
        del record["checks"]["restore_prediction"]
    elif mutation == "cpu":
        record["prediction"]["device"] = "cpu"
    elif mutation == "bf16":
        record["precision"]["autocast"] = "bfloat16"
    else:
        record["test_evaluation"] = True
    with pytest.raises(ValueError):
        validate_envelope(record, "Traffic", data)
