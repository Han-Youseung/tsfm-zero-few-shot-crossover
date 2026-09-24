"""Small, CPU-only evidence contract for phase 4.6 (not a research freeze)."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_crossover.data.splits import chronological_split
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.tracking.atomic import write_json_atomic

HORIZONS = (96, 192, 336, 720)
DATA_SHA = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
MODELS = Path("results/manifests/models")
CHECKS = (
    "cuda_parameters",
    "cuda_input",
    "synthetic_finite",
    "shape_finite",
    "eval_no_grad",
    "no_optimizer_zero_shot",
    "zero_shot_hash_unchanged",
    "seed_repeat",
    "channel_order",
    "full_requires_grad",
    "optimizer_all_parameters",
    "finite_loss",
    "finite_gradients",
    "optimizer_step",
    "parameters_updated",
    "validation_after_training",
    "restore_hash",
    "restore_prediction",
    "restore_optimizer",
    "restore_rng",
    "restore_config",
    "restore_step",
)


@lru_cache(maxsize=4)
def _registered_windows(horizon: int) -> dict:
    """Use the existing 60/20/20 splitter and index constructors; never emit test."""
    split = chronological_split(17420)
    common = ("ETTh1__official_raw", DATA_SHA)
    train = generate_train_windows(*common, split.train, 512, horizon)[0]
    validation = generate_rolling_windows(*common, "validation", split.validation, 512, horizon)[
        512
    ]
    return {"train": train.as_dict(), "validation": validation.as_dict()}


def windows(horizon: int) -> dict:
    return copy.deepcopy(_registered_windows(horizon))


def identity(root: Path, family: str, horizon: int, commit: str, samples: int) -> dict:
    if horizon not in HORIZONS or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected immutable project commit and supported horizon")
    if samples not in ((1,) if family == "ttm" else (8, 100)):
        raise ValueError("unsupported sample count")
    cpu = json.loads((root / MODELS / f"{family}_compatibility.json").read_text())
    if family == "ttm":
        selected = cpu["selected_model_revisions"][
            0 if horizon == 96 else 2 if horizon == 720 else 1
        ]
        revision, config_hash = selected["revision"], selected["config_sha256"]
        checkpoint = selected["name"]
    else:
        revision, config_hash = cpu["revision"], cpu["config_sha256"]
        checkpoint = cpu["repository"]
    return dict(
        model_family=family,
        repository=cpu["repository"],
        revision=revision,
        checkpoint=checkpoint,
        code_commit=cpu["code_commit"],
        config_sha256=config_hash,
        execution_commit=commit,
        data_sha256=DATA_SHA,
        horizon=horizon,
        context=512,
        channels=7,
        batch_size=1,
        dtype="float32",
        seed=1729,
        num_samples=samples,
        windows=windows(horizon),
        scaling="model_native_only",
        external_scaler=False,
        split=json.loads(json.dumps(chronological_split(17420).as_dict())),
        input_transform={
            "left_zero_padding": 512 if family == "ttm" and horizon == 720 else 0,
            "prediction_filter_length": horizon if family == "ttm" else None,
            "native_horizon": (336 if horizon == 192 else horizon) if family == "ttm" else horizon,
            "recursive": False,
            "patch_size": 64 if family == "moirai1" else None,
        },
        objective="official model forward loss"
        if family == "ttm"
        else "MoiraiFinetune/PackedNLLLoss",
        determinism={
            "algorithms_enabled": True,
            "warn_only": True,
            "reason": "CUDA median indices lack a strict deterministic implementation",
            "repeat_comparison": {"rtol": 1e-4, "atol": 1e-5},
        },
    )


class GPUCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["gpu_condition"] = "gpu_condition"
    schema_version: Literal[1] = 1
    identity: dict
    status: Literal["pending_gpu", "passed", "failed"] = "pending_gpu"
    installation_status: Literal["installation_pending", "verified"] = "installation_pending"
    environment: dict = Field(default_factory=dict)
    checks: dict[str, bool] = Field(default_factory=dict)
    details: dict = Field(default_factory=dict)
    amp: Literal["not_run"] = "not_run"
    error: dict | None = None
    protocol_frozen: Literal[False] = False

    @model_validator(mode="after")
    def evidence(self):
        i = self.identity
        if i.get("windows") != windows(i["horizon"]):
            raise ValueError("only the registered train/validation windows are allowed")
        if self.status == "passed":
            if self.installation_status != "verified":
                raise ValueError("installation has not been verified")
            e = self.environment
            if not (e.get("cuda_available") is True and e.get("torch_cuda") and e.get("gpu")):
                raise ValueError("CPU evidence cannot pass the CUDA gate")
            required = (
                CHECKS[: CHECKS.index("full_requires_grad")] if i["num_samples"] == 100 else CHECKS
            )
            if not all(self.checks.get(key) is True for key in required):
                raise ValueError("missing required GPU evidence")
            if self.error:
                raise ValueError("passed result cannot contain an error")
        return self


class GPUPendingManifest(BaseModel):
    """Unexecuted preparation metadata, deliberately incapable of representing success."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["gpu_preparation"]
    schema_version: Literal[1] = 1
    model_family: Literal["ttm", "moirai1"]
    gpu_status: Literal["pending_gpu"]
    installation_status: Literal["installation_pending"]
    fp32: Literal["not_run"]
    amp: Literal["not_run"]
    conditions: list = Field(max_length=0)
    protocol_frozen: Literal[False]
    production_adapter_allowed: Literal[False]
    notebook: str
    results_policy: str


def validate_result(payload: dict, root: Path, expected_commit: str) -> GPUCondition:
    from .compatibility import _reject_local_paths_and_secrets

    _reject_local_paths_and_secrets(payload)
    result = GPUCondition.model_validate(payload)
    i = result.identity
    if i != identity(root, i["model_family"], i["horizon"], expected_commit, i["num_samples"]):
        raise ValueError("revision/code/config/project commit/fingerprint identity mismatch")
    return result


def summarize(results: list[GPUCondition]) -> dict:
    completed = {
        (r.identity["model_family"], r.identity["horizon"], r.identity["num_samples"])
        for r in results
        if r.status == "passed"
    }
    commits = {r.identity["execution_commit"] for r in results}
    required = {(f, h, s) for f, s in (("ttm", 1), ("moirai1", 8)) for h in HORIZONS}
    passed = len(commits) == 1 and required <= completed
    return {
        "gpu_compatibility": "passed" if passed else "pending_gpu",
        "production_adapter_allowed": passed,
        "protocol_frozen": False,
        "moirai_100_samples": "passed"
        if all(("moirai1", h, 100) in completed for h in HORIZONS)
        else "pending_gpu",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate/import small GPU JSON; never alter CPU evidence"
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--import-results", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        ["git", "-C", str(args.root), "cat-file", "-e", args.expected_commit + "^{commit}"],
        check=True,
    )
    results = [
        validate_result(json.loads(p.read_text()), args.root, args.expected_commit)
        for p in args.files
    ]
    if args.import_results:
        for result in results:
            i = result.identity
            # Immutable attempt records. Do not overwrite previous failures or CPU manifests.
            import hashlib

            payload = result.model_dump()
            sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            target = (
                args.root
                / MODELS
                / "gpu_runs"
                / i["execution_commit"]
                / (f"{i['model_family']}-{i['horizon']}-{i['num_samples']}-{sha}.json")
            )
            if not target.exists():
                write_json_atomic(target, payload)
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
