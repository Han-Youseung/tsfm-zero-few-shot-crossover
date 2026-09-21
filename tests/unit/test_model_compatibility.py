from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tsfm_crossover.models.compatibility import validate_candidate, validate_manifest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("family", ["ttm", "moirai"])
def test_candidate_and_manifest_validate(family: str) -> None:
    candidate = validate_candidate(ROOT / "configs" / "models" / f"{family}_candidate.yaml")
    manifest = validate_manifest(
        ROOT / "results" / "manifests" / "models" / f"{family}_compatibility.json"
    )
    assert candidate["revision"] == manifest.revision
    assert manifest.gpu_status.value == "pending_colab"


def test_dependency_environments_are_separate() -> None:
    ttm = (ROOT / "requirements" / "ttm.txt").read_text(encoding="utf-8")
    moirai = (ROOT / "requirements" / "moirai.txt").read_text(encoding="utf-8")
    assert "granite-tsfm" in ttm and "uni2ts" not in ttm
    assert "uni2ts" in moirai and "granite-tsfm" not in moirai
    assert "fe7a35697723e2a2f5246ae979474bfc554e26c0" in ttm
    assert "cfd46d4510ed8896f263116f32928eede05b0a75" in moirai


def test_manifests_contain_no_local_paths_or_secret_values() -> None:
    for path in (ROOT / "results" / "manifests" / "models").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text
        assert "/Users/" not in text
        assert "/home/" not in text
        validate_manifest(path)


def test_model_revision_is_required(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "models" / "ttm_candidate.yaml").read_text(encoding="utf-8")
    )
    payload["revision"] = None
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        validate_candidate(path)


def test_gpu_claim_requires_evidence(tmp_path: Path) -> None:
    source = ROOT / "results" / "manifests" / "models" / "ttm_compatibility.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["gpu_status"] = "gpu_validated"
    payload["tested_device"] = []
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="tested device"):
        validate_manifest(path)


def test_moirai_finetune_is_not_overclaimed() -> None:
    path = ROOT / "results" / "manifests" / "models" / "moirai_compatibility.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["capabilities"]["full_parameter_finetuning"] is None
    assert payload["finetuning_smoke"] == "finetune_failed"
    assert payload["state"] == "import_validated"
