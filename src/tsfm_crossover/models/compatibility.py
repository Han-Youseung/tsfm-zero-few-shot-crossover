"""Validate model candidate metadata without importing heavyweight runtimes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contract import CandidateState, ProbeStatus


class CompatibilityManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    model_family: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=7)
    official_code_repository: str = Field(min_length=1)
    code_commit: str = Field(min_length=7)
    state: CandidateState
    cpu_status: ProbeStatus
    gpu_status: ProbeStatus
    tested_device: list[str]
    failures: list[str]
    warnings: list[str]

    @model_validator(mode="after")
    def validate_stage_and_gpu(self) -> CompatibilityManifest:
        if self.state in {CandidateState.pilot_validated, CandidateState.frozen}:
            raise ValueError("phase 4 cannot claim pilot_validated or frozen")
        if self.gpu_status == ProbeStatus.gpu_validated and not self.tested_device:
            raise ValueError("gpu_validated requires a tested device")
        return self


def _reject_local_paths_and_secrets(value: Any, key: str = "") -> None:
    secret_keys = {"token", "hf_token", "access_token", "secret", "password", "api_key"}
    if key.casefold() in secret_keys and value not in (None, False, "not_required"):
        raise ValueError(f"secret-bearing field is forbidden: {key}")
    if isinstance(value, dict):
        for child_key, child in value.items():
            _reject_local_paths_and_secrets(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            _reject_local_paths_and_secrets(child, key)
    elif isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("/")):
        raise ValueError("local absolute paths are forbidden")


def validate_manifest(path: str | Path) -> CompatibilityManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_local_paths_and_secrets(payload)
    return CompatibilityManifest.model_validate(payload)


def validate_candidate(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate YAML root must be a mapping")
    for key in ("family", "repository", "revision", "official_code_repository", "code_commit"):
        if not payload.get(key):
            raise ValueError(f"candidate requires {key}")
    if payload.get("state") == "frozen":
        raise ValueError("phase 4 candidate cannot be frozen")
    _reject_local_paths_and_secrets(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="compatibility JSON to validate")
    parser.add_argument("--candidate", type=Path, help="candidate YAML to validate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.manifest and not args.candidate:
        build_parser().print_help()
        return 0
    if args.manifest:
        validate_manifest(args.manifest)
    if args.candidate:
        validate_candidate(args.candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
