"""Manual TTM CPU compatibility probe; excluded from the default test suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
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
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ttm"))
    args = parser.parse_args()

    import pandas as pd
    import torch
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForDecomposedPrediction

    repository = "ibm-granite/granite-timeseries-ttm-r3"
    revision = "7b070728ff280ee2bb682bd8dfbe71da21ca2c7c"
    model = TinyTimeMixerForDecomposedPrediction.from_pretrained(
        repository,
        revision=revision,
        prediction_filter_length=96,
        cache_dir=args.cache_dir,
    )
    model.eval()
    synthetic = torch.randn(1, 512, 7)
    before = digest(model)
    started = time.perf_counter()
    with torch.no_grad():
        first = model(past_values=synthetic).prediction_outputs
        second = model(past_values=synthetic).prediction_outputs
    inference_seconds = time.perf_counter() - started
    assert first.shape == (1, 96, 7)
    assert torch.isfinite(first).all()
    assert digest(model) == before

    values = pd.read_csv(args.data).drop(columns=["date"]).to_numpy(dtype="float32")
    train_end = int(0.6 * len(values))
    validation_end = int(0.8 * len(values))
    origin = train_end + 512
    assert origin + 96 <= validation_end
    past = torch.from_numpy(values[origin - 512 : origin]).unsqueeze(0)
    future = torch.from_numpy(values[origin : origin + 96]).unsqueeze(0)
    model.eval()
    validation_before = digest(model)
    with torch.no_grad():
        validation_prediction = model(past_values=past).prediction_outputs
    assert validation_prediction.shape == future.shape
    assert torch.isfinite(validation_prediction).all()
    assert digest(model) == validation_before

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(past_values=past, future_values=future)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    assert digest(model) != validation_before
    trained_digest = digest(model)
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "training_state.pt"
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 1},
            state_path,
        )
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        assert saved["step"] == 1 and digest(model) == trained_digest

    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "repository": repository,
        "revision": revision,
        "device": "cpu",
        "dtype": str(first.dtype),
        "synthetic_shape": list(first.shape),
        "synthetic_finite": True,
        "synthetic_bitwise_repeat": torch.equal(first, second),
        "zero_shot_parameter_hash_unchanged": True,
        "validation_shape": list(validation_prediction.shape),
        "validation_finite": True,
        "validation_window_origin": origin,
        "test_split_used": False,
        "finetune_steps": 1,
        "finetune_loss_finite": True,
        "parameters_updated": True,
        "training_state_round_trip": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "inference_seconds_two_synthetic_forwards": inference_seconds,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
