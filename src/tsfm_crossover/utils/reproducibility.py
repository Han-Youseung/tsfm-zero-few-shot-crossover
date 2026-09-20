"""Central seed management without promising bitwise reproducibility."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SeedSettings:
    python: int
    numpy: int
    torch_cpu: int
    torch_cuda: int
    dataloader_worker: int
    model_sampling: int
    deterministic_requested: bool = False


def apply_seeds(settings: SeedSettings, torch_module: Any = ...) -> dict[str, Any]:
    random.seed(settings.python)
    applied = {"python": True, "numpy": False, "torch_cpu": False, "torch_cuda": False}

    try:
        numpy = importlib.import_module("numpy")
    except ImportError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(settings.numpy)
        applied["numpy"] = True

    torch = torch_module
    if torch is ...:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            torch = None

    deterministic_applied = False
    if torch is not None:
        torch.manual_seed(settings.torch_cpu)
        applied["torch_cpu"] = True
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.manual_seed_all(settings.torch_cuda)
            applied["torch_cuda"] = True
        if settings.deterministic_requested and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
            deterministic_applied = True

    return {
        "requested": asdict(settings),
        "applied": applied,
        "deterministic_requested": settings.deterministic_requested,
        "deterministic_applied": deterministic_applied,
        "bitwise_reproducibility_guaranteed": False,
    }


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    def seed_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        try:
            numpy = importlib.import_module("numpy")
        except ImportError:
            return
        numpy.random.seed(worker_seed % (2**32))

    return seed_worker
