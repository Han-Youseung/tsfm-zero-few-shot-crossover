"""Git, operating-system, and optional accelerator metadata."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_git_metadata(cwd: str | Path = ".") -> dict[str, Any]:
    root = Path(cwd)
    sha = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    return {
        "commit_sha": sha,
        "dirty": None if status is None else bool(status),
        "branch": _git(["branch", "--show-current"], root),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment_metadata(torch_module: Any = ...) -> dict[str, Any]:
    torch = torch_module
    if torch is ...:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            torch = None

    cuda_available = False
    cuda_version = None
    gpu_name = None
    pytorch_version = None
    if torch is not None:
        pytorch_version = getattr(torch, "__version__", None)
        cuda = getattr(torch, "cuda", None)
        cuda_available = bool(cuda is not None and cuda.is_available())
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_available:
            gpu_name = cuda.get_device_name(0)

    packages = {
        name: _package_version(name)
        for name in ("pydantic", "PyYAML", "numpy", "torch", "transformers", "uni2ts")
    }
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "pytorch_version": pytorch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "packages": packages,
    }
