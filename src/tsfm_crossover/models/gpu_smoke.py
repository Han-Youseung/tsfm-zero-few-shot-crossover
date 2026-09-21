"""Shared measurement/checkpoint mechanics for the existing manual probes.

No model implementations or experiment adapters live here. Heavy imports are lazy.
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
import tempfile
import time
import uuid
from pathlib import Path

from tsfm_crossover.tracking.atomic import write_json_atomic

from .gpu_gate import GPUCondition, identity, validate_result


def digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def set_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def equal_tree(a, b):
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def measured(function):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, {
        "wall_seconds": time.perf_counter() - started,
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "warm_up": False,
    }


def environment(family, expected_code):
    import torch

    package, version = ("granite-tsfm", "0.3.9") if family == "ttm" else ("uni2ts", "2.0.0")
    direct = json.loads(md.distribution(package).read_text("direct_url.json") or "{}")
    if (
        direct.get("vcs_info", {}).get("commit_id") != expected_code
        or md.version(package) != version
    ):
        raise RuntimeError("installed official source commit/package mismatch (PEP 610 required)")
    subprocess.run([os.sys.executable, "-m", "pip", "check"], check=True, capture_output=True)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(),
        "total_vram_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "nvidia_driver": driver,
        "official_code_commit": expected_code,
        "packages": {d.metadata["Name"]: d.version for d in md.distributions()},
        "execution_platform": "colab" if os.environ.get("COLAB_RELEASE_TAG") else "other_gpu",
        "interpreter": Path(os.sys.executable).name,
        "dtype": "float32",
        "seed": torch.initial_seed(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def run(args, family, factory, predict, loss_fn, optimizer_fn, channel_check):
    """One condition/process. Callbacks use the pinned official model APIs."""
    root = Path.cwd()
    i = identity(root, family, args.horizon, args.expected_commit, args.num_samples)
    result = GPUCondition(identity=i)
    if args.output.exists():
        old = validate_result(json.loads(args.output.read_text()), root, args.expected_commit)
        if old.identity != i:
            raise ValueError("refusing to replace another condition")
        if old.status == "passed":
            print("resume: verified completed condition", args.output.name)
            return 0
        write_json_atomic(
            args.output.with_name(args.output.stem + f"-{uuid.uuid4().hex}.json"), old.model_dump()
        )
    stage = "runtime"
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if head != args.expected_commit or dirty:
            raise RuntimeError("probe requires the exact clean execution commit")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            result.environment = {
                "cuda_available": False,
                "torch_cuda": torch.version.cuda,
                "torch": torch.__version__,
                "python": platform.python_version(),
            }
            result.details["reason"] = "CUDA unavailable; no GPU execution attempted"
            return 2
        stage = "installation_dependency"
        random.seed(i["seed"])
        np.random.seed(i["seed"])
        torch.manual_seed(i["seed"])
        torch.cuda.manual_seed_all(i["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result.environment = environment(family, i["code_commit"])
        result.installation_status = "verified"
        stage = "data_identity"
        if hashlib.sha256(args.data.read_bytes()).hexdigest() != i["data_sha256"]:
            raise ValueError("official raw ETTh1 fingerprint mismatch")
        import pandas as pd

        validation_end = i["split"]["validation"]["end"]
        frame = pd.read_csv(args.data, nrows=validation_end)
        if list(frame.columns) != ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]:
            raise ValueError("ETTh1 channel order mismatch")
        if len(frame) != validation_end:
            raise ValueError("ETTh1 validation row count mismatch")
        # Values past validation are never converted or used by the model.
        values = frame.iloc[:, 1:].to_numpy(dtype="float32")
        del frame

        def tensors(window):
            return tuple(
                torch.from_numpy(values[a:b].copy()).unsqueeze(0).cuda()
                for a, b in (
                    (window["context_start"], window["context_end"]),
                    (window["target_start"], window["target_end"]),
                )
            )

        train_x, train_y = tensors(i["windows"]["train"])
        val_x, _ = tensors(i["windows"]["validation"])
        stage = "model_revision_config"
        from huggingface_hub import hf_hub_download

        config = Path(
            hf_hub_download(
                i["repository"], "config.json", revision=i["revision"], cache_dir=args.cache_dir
            )
        )
        if hashlib.sha256(config.read_bytes()).hexdigest() != i["config_sha256"]:
            raise ValueError("model config hash mismatch")
        result.details["model_config"] = json.loads(config.read_text())
        model = factory(i, args.cache_dir, False).cuda().float().eval()
        checks, details = result.checks, result.details
        details["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
        details["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        details["input_device"] = str(val_x.device)
        checks["cuda_parameters"] = all(p.device.type == "cuda" for p in model.parameters())
        checks["cuda_input"] = val_x.device.type == train_x.device.type == "cuda"
        before = digest(model)
        stage = "official_api_shape"
        with torch.inference_mode():
            synthetic = predict(model, torch.randn_like(val_x), i)
            checks["synthetic_finite"] = bool(torch.isfinite(synthetic).all())
            del synthetic
            torch.manual_seed(i["seed"])
            prediction, details["inference"] = measured(lambda: predict(model, val_x, i))
            torch.manual_seed(i["seed"])
            repeated = predict(model, val_x, i)
            expected_shape = (
                (1, i["horizon"], 7) if family == "ttm" else (1, i["num_samples"], i["horizon"], 7)
            )
            checks["shape_finite"] = tuple(prediction.shape) == expected_shape and bool(
                torch.isfinite(prediction).all()
            )
            details["sample_shape"] = list(prediction.shape) if family != "ttm" else None
            point = prediction if family == "ttm" else prediction.median(dim=1).values
            details["prediction_device"] = str(prediction.device)
            details["prediction_dtype"] = str(prediction.dtype)
            details["point_shape"] = list(point.shape)
            stage = "reproducibility"
            torch.testing.assert_close(prediction, repeated, rtol=1e-4, atol=1e-5)
            checks["seed_repeat"] = True
            stage = "official_api_shape"
            details["channel_permutation_max_absolute_error"] = channel_check(model, val_x, i)
            checks["channel_order"] = True
            checks["eval_no_grad"] = not model.training and not prediction.requires_grad
        checks["zero_shot_hash_unchanged"] = digest(model) == before
        checks["no_optimizer_zero_shot"] = True
        from .gpu_gate import CHECKS

        if not all(checks.get(key) for key in CHECKS[: CHECKS.index("full_requires_grad")]):
            raise ValueError("required Zero-Shot checks did not pass")
        details["zero_shot_parameter_hash"] = before
        details["comparison_tolerance"] = {"rtol": 1e-4, "atol": 1e-5}
        details["inference"]["warm_up"] = "one synthetic forward; shape-specific first validation"
        del model, prediction, repeated, point
        gc.collect()
        torch.cuda.empty_cache()
        if i["num_samples"] == 100:
            details["finetune"] = "not_run; covered separately at eight samples"
        else:
            stage = "official_api_shape"
            model = factory(i, args.cache_dir, True).cuda().float()
            pretrained_hash = digest(model.module if family == "moirai1" else model)
            details["training_pretrained_parameter_hash"] = pretrained_hash
            if pretrained_hash != before:
                raise ValueError("fine-tuning did not start from the original pretrained weights")
            optimizer = optimizer_fn(model)
            parameters = list(model.parameters())
            checks["full_requires_grad"] = all(p.requires_grad for p in parameters)
            checks["optimizer_all_parameters"] = {id(p) for p in parameters} == {
                id(p) for g in optimizer.param_groups for p in g["params"]
            }
            details["total_parameters"] = sum(p.numel() for p in parameters)
            details["trainable_parameters"] = sum(p.numel() for p in parameters if p.requires_grad)
            if not (checks["full_requires_grad"] and checks["optimizer_all_parameters"]):
                raise ValueError("full-parameter training coverage is incomplete")
            initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            model.train()
            stage = "nonfinite_loss_gradient"

            def train_step():
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, train_x, train_y, i)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                checks["finite_loss"] = True
                details["loss"] = float(loss.detach())
                absent = [n for n, p in model.named_parameters() if p.grad is None]
                details["parameters_without_gradient"] = absent
                checks["finite_gradients"] = any(p.grad is not None for p in parameters) and all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters
                )
                if not checks["finite_gradients"]:
                    raise ValueError("missing/nonfinite gradients")
                optimizer.step()

            _, details["training_step"] = measured(train_step)
            checks["optimizer_step"] = True
            changed = [
                n for n, p in model.named_parameters() if not torch.equal(initial[n], p.cpu())
            ]
            details["changed_parameter_tensors"] = len(changed)
            details["parameter_tensors"] = len(initial)
            checks["parameters_updated"] = bool(changed)
            details["actual_optimizer_steps"] = 1
            details["full_finetuning_definition"] = (
                "all requires_grad and optimizer membership; not every element changes"
            )
            del initial, parameters
            model.eval()
            trained_hash = digest(model)
            state_rng = rng_state()
            with torch.inference_mode():
                expected = predict(model, val_x, i).detach().cpu()
            checks["validation_after_training"] = bool(torch.isfinite(expected).all())
            stage = "checkpoint_restore"
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.cache_dir) as directory:
                checkpoint = Path(directory) / "smoke.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": 1,
                        "identity": i,
                        "config": details["model_config"],
                        "rng": state_rng,
                        "grad_scaler": None,
                    },
                    checkpoint,
                )
                del model, optimizer
                gc.collect()
                torch.cuda.empty_cache()
                model = factory(i, args.cache_dir, True).cuda().float()
                optimizer = optimizer_fn(model)
                # Trusted checkpoint just created in this process, never an imported pickle.
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                checks["restore_optimizer"] = equal_tree(optimizer.state_dict(), saved["optimizer"])
                checks["restore_step"] = saved["global_step"] == 1
                checks["restore_config"] = (
                    saved["identity"] == i and saved["config"] == details["model_config"]
                )
                checks["restore_hash"] = digest(model) == trained_hash
                set_rng(saved["rng"])
                checks["restore_rng"] = equal_tree(rng_state(), saved["rng"])
                model.eval()
                with torch.inference_mode():
                    restored = predict(model, val_x, i).detach().cpu()
                torch.testing.assert_close(expected, restored, rtol=1e-4, atol=1e-5)
                checks["restore_prediction"] = True
                details["restore_max_absolute_error"] = float((expected - restored).abs().max())
                details["trained_parameter_hash"] = trained_hash
        result.status = "passed"
        GPUCondition.model_validate(result.model_dump())
        return 0
    except Exception as exc:
        result.status = "failed"
        category = "out_of_memory" if "out of memory" in str(exc).lower() else stage
        message = str(exc).lower()
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            category = "installation_dependency"
        elif any(
            s in message for s in ("no kernel image", "driver version", "not compiled with cuda")
        ):
            category = "gpu_runtime"
        elif type(exc).__module__.startswith(("huggingface_hub", "httpx", "requests")):
            category = "download_infrastructure"
        # No exception text/paths/tokens in portable results. Log exception class and stage only.
        result.error = {
            "category": category,
            "exception_type": type(exc).__name__,
            "summary": "condition stopped; see failing stage; no model fallback performed",
        }
        print("FAILED", category, type(exc).__name__)
        return 1
    finally:
        write_json_atomic(args.output, result.model_dump(), overwrite=args.output.exists())
        print(result.status, args.output.name)


def arguments(parser, default_samples):
    parser.add_argument("--gpu-gate", action="store_true")
    parser.add_argument("--horizon", type=int, choices=(96, 192, 336, 720), default=96)
    parser.add_argument("--num-samples", type=int, default=default_samples)
    parser.add_argument("--expected-commit")
