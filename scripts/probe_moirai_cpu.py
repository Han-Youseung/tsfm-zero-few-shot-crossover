"""Manual MOIRAI 2 CPU inference probe; no unofficial fine-tuning path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path


def digest(model) -> str:
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/moirai"))
    parser.add_argument("--channels", type=int, default=7)
    args = parser.parse_args()

    import pandas as pd
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    repository = "Salesforce/moirai-2.0-R-small"
    revision = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"
    module = Moirai2Module.from_pretrained(repository, revision=revision, cache_dir=args.cache_dir)
    model = Moirai2Forecast(
        prediction_length=96,
        target_dim=args.channels,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=512,
        module=module,
    ).eval()
    synthetic = torch.randn(1, 512, args.channels)
    observed = torch.ones_like(synthetic, dtype=torch.bool)
    pad = torch.zeros(1, 512, dtype=torch.bool)
    before = digest(model)
    started = time.perf_counter()
    with torch.no_grad():
        first = model(synthetic, observed, pad)
        second = model(synthetic, observed, pad)
    inference_seconds = time.perf_counter() - started
    expected_quantile_shape = (1, 9, 96) if args.channels == 1 else (1, 9, 96, args.channels)
    expected_point_shape = (1, 96) if args.channels == 1 else (1, 96, args.channels)
    assert first.shape == expected_quantile_shape, tuple(first.shape)
    assert torch.isfinite(first).all()
    assert digest(model) == before

    values = pd.read_csv(args.data).drop(columns=["date"]).to_numpy(dtype="float32")
    values = values[:, : args.channels]
    train_end = int(0.6 * len(values))
    validation_end = int(0.8 * len(values))
    origin = train_end + 512
    assert origin + 96 <= validation_end
    past = torch.from_numpy(values[origin - 512 : origin]).unsqueeze(0)
    observed = torch.ones_like(past, dtype=torch.bool)
    pad = torch.zeros(1, 512, dtype=torch.bool)
    validation_before = digest(model)
    with torch.no_grad():
        validation_quantiles = model(past, observed, pad)
    point = validation_quantiles[:, 4]
    assert validation_quantiles.shape == expected_quantile_shape
    assert point.shape == expected_point_shape
    assert torch.isfinite(validation_quantiles).all()
    assert digest(model) == validation_before

    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "repository": repository,
        "revision": revision,
        "device": "cpu",
        "dtype": str(first.dtype),
        "channels": args.channels,
        "synthetic_quantile_shape": list(first.shape),
        "synthetic_point_shape": list(first[:, 4].shape),
        "synthetic_finite": True,
        "synthetic_bitwise_repeat": torch.equal(first, second),
        "zero_shot_parameter_hash_unchanged": True,
        "validation_quantile_shape": list(validation_quantiles.shape),
        "validation_point_shape": list(point.shape),
        "validation_finite": True,
        "validation_window_origin": origin,
        "test_split_used": False,
        "finetune_steps": 0,
        "finetune_status": "unsupported_by_pinned_official_API",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inference_seconds_two_synthetic_forwards": inference_seconds,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
