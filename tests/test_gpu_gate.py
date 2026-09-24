"""Gate tests use synthetic metadata only: no torch imports, GPU, or weights."""

import copy
import json
from pathlib import Path

import pytest

from tsfm_crossover.models.gpu_gate import (
    CHECKS,
    GPUCondition,
    identity,
    summarize,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "fb617a975d3f2087a3e8b81c327134d1508a7fff"


def record(family="ttm", horizon=96, samples=1):
    return GPUCondition(identity=identity(ROOT, family, horizon, COMMIT, samples))


def evidence(family="ttm", horizon=96, samples=1):
    item = record(family, horizon, samples).model_dump()
    item.update(
        status="passed",
        installation_status="verified",
        environment={"cuda_available": True, "torch_cuda": "test fixture", "gpu": "fixture"},
        checks=dict.fromkeys(CHECKS, True),
    )
    return item


def test_no_cuda_pending():
    item = record()
    assert item.status == "pending_gpu"
    assert not summarize([item])["production_adapter_allowed"]


@pytest.mark.parametrize("key", CHECKS)
def test_missing_evidence_cannot_pass(key):
    item = evidence()
    del item["checks"][key]
    with pytest.raises(ValueError, match="missing required"):
        GPUCondition.model_validate(item)


def test_cpu_cannot_masquerade():
    item = evidence()
    item["environment"]["cuda_available"] = False
    with pytest.raises(ValueError, match="CPU"):
        GPUCondition.model_validate(item)


@pytest.mark.parametrize(
    "key", ["revision", "execution_commit", "code_commit", "data_sha256", "config_sha256"]
)
def test_import_exact_identity(key):
    item = record().model_dump()
    item["identity"][key] = "0" * 40
    with pytest.raises(ValueError, match="mismatch"):
        validate_result(item, ROOT, COMMIT)


def test_test_split_forbidden():
    item = record().model_dump()
    item["identity"]["windows"]["validation"]["split"] = "test"
    with pytest.raises(ValueError, match="registered train/validation"):
        GPUCondition.model_validate(item)


def test_all_horizons_both_models_required_not_protocol_freeze():
    rows = [
        GPUCondition.model_validate(evidence(f, h, s))
        for f, s in (("ttm", 1), ("moirai1", 8))
        for h in (96, 192, 336, 720)
    ]
    assert not summarize(rows[:-1])["production_adapter_allowed"]
    summary = summarize(rows)
    assert summary["production_adapter_allowed"]
    assert not summary["protocol_frozen"]
    assert summary["moirai_100_samples"] == "pending_gpu"
    payload = rows[0].model_dump()
    payload["protocol_frozen"] = True
    with pytest.raises(ValueError):
        GPUCondition.model_validate(payload)


def test_mixed_runs_do_not_pass():
    rows = [
        GPUCondition.model_validate(evidence(f, h, s))
        for f, s in (("ttm", 1), ("moirai1", 8))
        for h in (96, 192, 336, 720)
    ]
    rows[-1].identity["execution_commit"] = "a" * 40
    assert not summarize(rows)["production_adapter_allowed"]


def test_validation_does_not_modify_cpu_evidence():
    paths = [
        ROOT / "results/manifests/models" / name
        for name in (
            "ttm_compatibility.json",
            "moirai1_compatibility.json",
            "moirai_compatibility.json",
        )
    ]
    before = {p: p.read_bytes() for p in paths}
    item = record().model_dump()
    original = copy.deepcopy(item)
    validate_result(json.loads(json.dumps(item)), ROOT, COMMIT)
    assert item == original
    assert all(p.read_bytes() == data for p, data in before.items())


def test_ttm_revisions_are_horizon_specific():
    assert record(horizon=96).identity["revision"] != record(horizon=720).identity["revision"]
    assert record(horizon=192).identity["revision"] == record(horizon=336).identity["revision"]


def test_cuda_median_determinism_policy_is_explicit():
    policy = record(horizon=96).identity["determinism"]
    assert policy == {
        "algorithms_enabled": True,
        "warn_only": True,
        "reason": "CUDA median indices lack a strict deterministic implementation",
        "repeat_comparison": {"rtol": 1e-4, "atol": 1e-5},
    }


def test_100_samples_inference_does_not_require_duplicate_training():
    item = evidence("moirai1", 720, 100)
    item["checks"] = {k: True for k in CHECKS[: CHECKS.index("full_requires_grad")]}
    assert GPUCondition.model_validate(item).status == "passed"


def test_actual_no_cuda_path_persists_pending_without_loading_weights(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tsfm_crossover.models import gpu_smoke

    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda=None),
            __version__="fixture+cpu",
        ),
    )
    monkeypatch.setattr(
        gpu_smoke.subprocess, "check_output", lambda cmd, **kwargs: COMMIT if "HEAD" in cmd else ""
    )
    args = SimpleNamespace(
        horizon=96, num_samples=1, expected_commit=COMMIT, output=tmp_path / "pending.json"
    )
    assert gpu_smoke.run(args, "ttm", None, None, None, None, None) == 2
    saved = validate_result(json.loads(args.output.read_text()), ROOT, COMMIT)
    assert saved.status == "pending_gpu" and not saved.environment["cuda_available"]


def test_completed_condition_resume_requires_no_model_runtime(tmp_path):
    from types import SimpleNamespace

    from tsfm_crossover.models.gpu_smoke import run
    from tsfm_crossover.tracking.atomic import write_json_atomic

    args = SimpleNamespace(
        horizon=96, num_samples=1, expected_commit=COMMIT, output=tmp_path / "complete.json"
    )
    write_json_atomic(args.output, evidence())
    assert run(args, "ttm", None, None, None, None, None) == 0
