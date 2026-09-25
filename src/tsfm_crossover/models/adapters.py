"""Shared lifecycle using the existing contract; vendor imports remain lazy.

Each instance is one independent condition. Only same-condition checkpoints may
resume it. FP32 engineering support is distinct from a frozen experiment protocol.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import subprocess
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

from .adapter_config import AdapterConfig
from .contract import AdapterContract, CapabilityReport, ModelMetadata, parameter_hash
from .gpu_gate import MODELS
from .gpu_smoke import rng_state, set_rng
from .window_batch import WindowBatch


def verified_source(package: str, version: str, commit: str, module_name: str) -> dict:
    """Check PEP610 immutable VCS origin, or a byte-matched local source checkout."""
    dist = md.distribution(package)
    if dist.version != version:
        raise RuntimeError("official package version mismatch")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") == commit:
        return {"package": package, "version": version, "code_commit": commit, "source": "vcs"}
    parsed = urlparse(direct.get("url", ""))
    if parsed.scheme != "file":
        raise RuntimeError("official source commit unavailable in installation metadata")
    location = unquote(parsed.path)
    if os.name == "nt" and location.startswith("/"):
        location = location[1:]
    source = Path(location)
    git = ["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source)]
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != commit:
        raise RuntimeError("local official source commit mismatch")
    if subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise RuntimeError("local official source has tracked changes")
    prefix = "src/" if package == "uni2ts" else ""
    tracked = subprocess.check_output(
        [*git, "ls-files", prefix + module_name], text=True
    ).splitlines()
    for name in tracked:
        if name.endswith(".py"):
            installed = Path(dist.locate_file(name.removeprefix(prefix)))
            if not installed.is_file() or installed.read_bytes() != (source / name).read_bytes():
                raise RuntimeError("installed official Python source differs from pinned checkout")
    if not tracked:
        raise RuntimeError("official source files unavailable")
    return {
        "package": package,
        "version": version,
        "code_commit": commit,
        "source": "byte_matched_local_checkout",
    }


class ExperimentAdapter(AdapterContract):
    """FP32 lifecycle; subclasses supply only vendor-specific operations."""

    def __init__(
        self,
        config: AdapterConfig,
        *,
        root: Path,
        device="cpu",
        cache_dir: Path | None = None,
        local_files_only=False,
    ):
        self.config, self.root = config, Path(root)
        self.device, self.cache_dir = device, cache_dir
        self.local_files_only = local_files_only
        self.model = self.optimizer = None
        self.optimizer_created = False
        self.backward_calls = self.global_step = self.window_exposures = 0
        self.source = {}
        cpu = json.loads((self.root / MODELS / f"{config.family}_compatibility.json").read_text())
        self.spec = dict(
            repository=cpu["repository"],
            code_commit=cpu["code_commit"],
            official_code_repository=cpu["official_code_repository"],
        )
        chosen = (
            cpu["selected_model_revisions"][
                0 if config.horizon == 96 else 2 if config.horizon == 720 else 1
            ]
            if config.family == "ttm"
            else cpu
        )
        self.spec.update(revision=chosen["revision"], config_sha256=chosen["config_sha256"])
        self.initial_hash = None

    def _load(self):
        raise NotImplementedError

    def _predict(self, past):
        raise NotImplementedError

    def _loss(self, batch, *, training):
        raise NotImplementedError

    def _configure(self):
        raise NotImplementedError

    def load_model(self):
        if self.model is not None:
            raise RuntimeError("use a new adapter for an independent pretrained condition")
        import torch
        from huggingface_hub import hf_hub_download

        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; adapter GPU validation remains pending")
        config_path = hf_hub_download(
            self.spec["repository"],
            "config.json",
            revision=self.spec["revision"],
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        if hashlib.sha256(Path(config_path).read_bytes()).hexdigest() != self.spec["config_sha256"]:
            raise ValueError("pretrained config hash mismatch")
        self.pretrained_config = json.loads(Path(config_path).read_text())
        self.model = self._load().to(self.device).float().eval()
        self.initial_hash = parameter_hash(self.parameter_state())

    def pretrained_kwargs(self):
        return {
            "revision": self.spec["revision"],
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }

    def prepare_batch(self, batch):
        import torch

        if not isinstance(batch, WindowBatch):
            raise TypeError("WindowBatch required")
        if batch.split not in {"train", "validation"}:
            raise ValueError("test split blocked in this engineering adapter")
        if (
            batch.channel_names != self.config.channel_names
            or batch.dataset_fingerprint != self.config.dataset_fingerprint
            or batch.sampling_manifest_hash != self.config.sampling_manifest_hash
        ):
            raise ValueError("channel order/data/sampling identity mismatch")
        past = torch.as_tensor(batch.past, dtype=torch.float32, device=self.device)
        channels = len(self.config.channel_names)
        if (
            past.ndim != 3
            or tuple(past.shape[1:]) != (self.config.context, channels)
            or past.shape[0] < 1
            or len(batch.window_ids) != past.shape[0]
        ):
            raise ValueError("past must be (batch, context, channels) with window IDs")
        if not torch.isfinite(past).all():
            raise ValueError("nonfinite past")
        future = None
        if batch.future is not None:
            future = torch.as_tensor(batch.future, dtype=torch.float32, device=self.device)
            if tuple(future.shape) != (past.shape[0], self.config.horizon, channels):
                raise ValueError("future must exactly match prediction horizon and channels")
            if not torch.isfinite(future).all():
                raise ValueError("nonfinite future")
        return WindowBatch(
            past,
            future,
            batch.split,
            batch.channel_names,
            batch.window_ids,
            batch.dataset_fingerprint,
            batch.sampling_manifest_hash,
        )

    def predict(self, batch):
        from dataclasses import replace

        import torch

        if self.model is None:
            raise RuntimeError("load model first")
        # Prediction never needs targets; missing targets belong to evaluation only.
        batch = self.prepare_batch(replace(batch, future=None))
        state, mode = rng_state(), self.model.training
        try:
            self.model.eval()
            # Evaluation consumes its own reproducible stream, including wrapper construction.
            torch.manual_seed(self.config.prediction_seed)
            with torch.inference_mode():
                point = self.point_forecast(self._predict(batch.past))
            expected = (batch.past.shape[0], self.config.horizon, len(self.config.channel_names))
            if tuple(point.shape) != expected or not torch.isfinite(point).all():
                raise ValueError("prediction shape/nonfinite output")
            return point
        finally:
            self.model.train(mode)
            set_rng(state)

    def configure_finetuning(self):
        import numpy as np
        import torch

        if self.model is None or self.optimizer is not None:
            raise RuntimeError("load pretrained once, then configure optimizer once")
        if parameter_hash(self.parameter_state()) != self.initial_hash:
            raise RuntimeError("fine-tuning must start from this condition's pretrained weights")
        random.seed(self.config.training_seed)
        np.random.seed(self.config.training_seed)
        torch.manual_seed(self.config.training_seed)
        self.optimizer = self._configure()
        parameters = list(self.model.parameters())
        if not all(p.requires_grad for p in parameters) or {id(p) for p in parameters} != {
            id(p) for group in self.optimizer.param_groups for p in group["params"]
        }:
            raise RuntimeError("full parameter optimizer coverage required")
        self.optimizer_created = True

    def train_step(self, batch, *, forward_context=None):
        from contextlib import nullcontext

        import torch

        batch = self.prepare_batch(batch)
        if batch.split != "train" or batch.future is None:
            raise ValueError("training requires train targets")
        if self.optimizer is None or self.global_step >= self.config.max_optimizer_steps:
            raise RuntimeError("optimizer absent or step budget exhausted")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with forward_context if forward_context is not None else nullcontext():
            loss = self._loss(batch, training=True)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("nonfinite/non-scalar official loss")
        loss.backward()
        self.backward_calls += 1
        parameters = list(self.model.parameters())
        if not any(p.grad is not None for p in parameters) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters
        ):
            raise ValueError("missing/nonfinite gradients")
        self.last_gradient_missing = [n for n, p in self.model.named_parameters() if p.grad is None]
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters, self.config.gradient_clip, error_if_nonfinite=True
            )
        self.optimizer.step()
        self.global_step += 1
        self.window_exposures += batch.past.shape[0]
        return float(loss.detach())

    def validation_step(self, batch):
        import torch

        batch = self.prepare_batch(batch)
        if batch.split != "validation" or batch.future is None:
            raise ValueError("validation objective requires validation targets")
        state, mode = rng_state(), self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                loss = self._loss(batch, training=False)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite validation loss")
            return float(loss)
        finally:
            self.model.train(mode)
            set_rng(state)

    def save_training_state(self, path: Path, *, loop_state=None):
        import torch

        if self.optimizer is None:
            raise RuntimeError("training state requires an optimizer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        payload = {
            "config": self.config.model_dump(),
            "spec": self.spec,
            "pretrained_config": self.pretrained_config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "window_exposures": self.window_exposures,
            "backward_calls": self.backward_calls,
            "rng": rng_state(),
            "grad_scaler": None,
            "scheduler": None,
            "parameter_hash": parameter_hash(self.parameter_state()),
            "loop_state": loop_state,
        }
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_training_state(self, path: Path):
        """Load a trusted locally produced checkpoint, never an arbitrary downloaded pickle."""
        import torch

        if self.model is None or self.global_step or self.optimizer is not None:
            raise RuntimeError("resume requires a fresh loaded adapter")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (
            saved["config"] != self.config.model_dump()
            or saved["spec"] != self.spec
            or saved["pretrained_config"] != self.pretrained_config
        ):
            raise ValueError("resume condition/config/revision mismatch; warm start is forbidden")
        self.configure_finetuning()
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.global_step = saved["global_step"]
        self.window_exposures = saved["window_exposures"]
        self.backward_calls = saved["backward_calls"]
        if parameter_hash(self.parameter_state()) != saved["parameter_hash"]:
            raise ValueError("restored parameter hash mismatch")
        set_rng(saved["rng"])
        self.loaded_loop_state = saved.get("loop_state")

    def parameter_state(self):
        if self.model is None:
            raise RuntimeError("load model first")
        return self.model.state_dict()

    def set_evaluation_mode(self):
        self.model.eval()

    def count_parameters(self):
        return (
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            sum(p.numel() for p in self.model.parameters()),
        )

    def get_model_metadata(self):
        return ModelMetadata(
            family=self.config.family,
            repository=self.spec["repository"],
            revision=self.spec["revision"],
            official_code_repository=self.spec["official_code_repository"],
            code_commit=self.spec["code_commit"],
            state="finetune_validated",
            point_forecast_rule=self.config.point_statistic + " (provisional)",
        )

    def validate_capabilities(self):
        return CapabilityReport(
            zero_shot=True,
            full_parameter_finetuning=True,
            multivariate=True,
            missing_values=False,
            context_lengths=[512],
            prediction_lengths=[96, 192, 336, 720],
            notes=[
                "Prior probe GPU scope: batch 1, 7 channels, ETTh1, FP32.",
                "Adapter and other channel counts require separate validation.",
            ],
        )

    def execution_metadata(self):
        import torch

        trainable, total = self.count_parameters()
        return {
            "model": self.get_model_metadata().model_dump(mode="json"),
            "config": self.config.model_dump(mode="json"),
            "official_source": self.source,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "packages": {d.metadata["Name"]: d.version for d in md.distributions()},
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name() if str(self.device).startswith("cuda") else None,
            "actual_optimizer_steps": self.global_step,
            "window_exposures": self.window_exposures,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "parameters_without_gradient": getattr(self, "last_gradient_missing", []),
            "scheduler": "none; smoke only",
            "amp": "not_run",
            "protocol_frozen": False,
        }

    def cleanup(self):
        self.optimizer = self.model = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_adapter(config: AdapterConfig, **kwargs) -> ExperimentAdapter:
    from .moirai_adapter import MoiraiAdapter
    from .ttm_adapter import TTMAdapter

    return (TTMAdapter if config.family == "ttm" else MoiraiAdapter)(config, **kwargs)
