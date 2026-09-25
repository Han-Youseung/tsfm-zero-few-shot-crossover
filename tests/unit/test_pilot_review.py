"""Synthetic importer fixtures; these are not model execution evidence."""

import copy
import json
import zipfile

import pytest

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.sampling import build_sampling_manifest
from tsfm_crossover.data.splits import chronological_split
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.experiments.pilot import validation_subset
from tsfm_crossover.experiments.pilot_config import PilotConfig, execution_plan
from tsfm_crossover.experiments.pilot_review import (
    EXECUTION_COMMIT,
    read_archive,
    validate_metadata,
    validate_record,
)


@pytest.fixture
def evidence():
    config = PilotConfig()
    entry = {
        "status": "ready_with_warnings",
        "variant": "ETTh1__fixture",
        "qc": {"rows": 5000, "sha256": "a" * 64, "channel_names": ["x"]},
    }
    row = execution_plan(config, {"ETTh1": entry}, EXECUTION_COMMIT)["conditions"][0]
    identity = {
        "commit": EXECUTION_COMMIT,
        "condition": row,
        "config": config.model_dump(mode="json"),
        "data_sha256": "a" * 64,
        "device": "cuda",
    }
    split = chronological_split(5000)
    sampling = build_sampling_manifest(
        generate_train_windows(entry["variant"], "a" * 64, split.train, 512, 96),
        dataset_sha256="a" * 64,
        split_config_hash=split.split_config_hash,
        seed=config.seed,
        code_commit_sha=EXECUTION_COMMIT,
        rates=[0.05],
    )
    windows = validation_subset(
        generate_rolling_windows(
            entry["variant"], "a" * 64, "validation", split.validation, 512, 96
        ),
        64,
    )
    record = {
        "identity": identity,
        "status": "completed",
        "test_evaluation": False,
        "protocol_frozen": False,
        "environment": {"gpu": "synthetic GPU fixture", "vram_mb": 1},
        "data": {
            "test_targets_read": False,
            "split": json.loads(json.dumps(split.as_dict())),
            "fingerprint": "a" * 64,
            "sampling_manifest": sampling.as_dict(),
            "validation_windows": [w.as_dict() for w in windows],
            "validation_window_hash": stable_hash([w.window_id for w in windows]),
        },
        "attempts": [],
        "fp32_batch1_supported": False,
    }
    return record, row, config, entry, {}


def test_completed_ladder_is_not_success(evidence):
    validate_record(*evidence)
    evidence[0]["fp32_batch1_supported"] = True
    with pytest.raises(ValueError, match="support mismatch"):
        validate_record(*evidence)


def test_regenerated_eligibility_must_match(evidence):
    expected = copy.deepcopy(evidence[0]["data"])
    validate_record(*evidence, expected_data=expected)
    evidence[0]["data"]["total_train_windows"] = 999
    with pytest.raises(ValueError, match="eligibility/data mismatch"):
        validate_record(*evidence, expected_data=expected)


def test_new_execution_commit_is_checked(evidence):
    from tsfm_crossover.experiments.new_data_review import COMMIT

    with pytest.raises(ValueError, match="commit/config"):
        validate_record(*evidence, commit=COMMIT, expected_data=evidence[0]["data"])


def test_wrong_model_revision_rejected():
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_metadata(
            {"model": {"revision": "wrong"}},
            {"family": "moirai1"},
            None,
            None,
            {"revision": "pinned"},
        )


@pytest.mark.parametrize(
    "field,value", [("commit", "wrong"), ("device", "cpu"), ("data_sha256", "b" * 64)]
)
def test_identity_mismatch_blocked(evidence, field, value):
    evidence[0]["identity"][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        validate_record(*evidence)


@pytest.mark.parametrize("field", ["test_evaluation", "protocol_frozen"])
def test_test_and_freeze_blocked(evidence, field):
    evidence[0][field] = True
    with pytest.raises(ValueError):
        validate_record(*evidence)


def test_sampling_and_validation_tamper(evidence):
    changed = copy.deepcopy(evidence)
    changed[0]["data"]["sampling_manifest"]["sampling_seed"] = 9
    with pytest.raises(ValueError, match="sampling mismatch"):
        validate_record(*changed)
    evidence[0]["data"]["validation_windows"][0]["split"] = "test"
    with pytest.raises(ValueError, match="window mismatch"):
        validate_record(*evidence)


@pytest.mark.parametrize("name", ["../a.json", "/a.json", "C:/a.json", "a\\b.json", "model.pt"])
def test_unsafe_zip_blocked(tmp_path, name):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("placeholder.json")
        info.filename = name  # Bypass Windows ZipInfo path normalization in this fixture.
        archive.writestr(info, "{}")
    with pytest.raises(ValueError, match="unsafe"):
        read_archive(path)


def test_read_only_zip(tmp_path):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("condition/result.json", '{"status":"failed"}')
    assert read_archive(path) == {"condition/result.json": {"status": "failed"}}
    assert not (tmp_path / "condition").exists()


def test_preparation_rerun_accepts_json_tuple_roundtrip(tmp_path, monkeypatch):
    from scripts import prepare_monash_variants as module

    folder = tmp_path / "results/manifests/pilot"
    folder.mkdir(parents=True)
    (folder / "prepared_data.json").write_text("{}")
    item = {"qc": {"sha256": "fixture", "ratios": (0.6, 0.2, 0.2)}, "author_matrix_comparison": {}}
    target = folder / "prepared_expansion.json"
    target.write_text(json.dumps(dict.fromkeys(module.SOURCES, item)))
    original = target.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["prepare"])
    monkeypatch.setattr(module, "prepare", lambda name, root: item)
    module.main()
    assert target.read_bytes() == original
