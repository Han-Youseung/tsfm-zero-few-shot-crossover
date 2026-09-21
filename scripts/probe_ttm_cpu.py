"""Manual TTM CPU compatibility probe; excluded from the default test suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path


def gpu_probe(args):
    import torch

    from tsfm_crossover.models.gpu_smoke import run

    def factory(i, cache, training):
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerForDecomposedPrediction,
            TinyTimeMixerForPrediction,
        )

        cls = (
            TinyTimeMixerForPrediction
            if i["horizon"] == 720
            else TinyTimeMixerForDecomposedPrediction
        )
        return cls.from_pretrained(
            i["repository"],
            revision=i["revision"],
            cache_dir=cache,
            prediction_filter_length=i["horizon"],
        )

    def inputs(x, i):
        # Official get_model zeropad contract requires caller-side left zero padding.
        if i["horizon"] == 720:
            x = torch.nn.functional.pad(x, (0, 0, 512, 0))
        return x

    def predict(model, x, i):
        return model(past_values=inputs(x, i)).prediction_outputs

    def loss(model, x, y, i):
        return model(past_values=inputs(x, i), future_values=y).loss

    def channels(model, x, i):
        perm = torch.tensor([6, 4, 2, 0, 1, 3, 5], device=x.device)
        a, b = predict(model, x, i), predict(model, x[..., perm], i)[..., torch.argsort(perm)]
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        return float((a - b).abs().max())

    return run(
        args,
        "ttm",
        factory,
        predict,
        loss,
        lambda m: torch.optim.AdamW(m.parameters(), lr=1e-6),
        channels,
    )


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
    parser.add_argument("--output", type=Path)
    from tsfm_crossover.models.gpu_smoke import arguments

    arguments(parser, 1)
    args = parser.parse_args()
    if args.gpu_gate:
        if not args.output or not args.expected_commit:
            parser.error("--gpu-gate requires --output and --expected-commit")
        return gpu_probe(args)

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
    # Legacy CPU smoke also must train only on the train split.
    train_past = torch.from_numpy(values[:512]).unsqueeze(0)
    train_future = torch.from_numpy(values[512:608]).unsqueeze(0)
    output = model(past_values=train_past, future_values=train_future)
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
