"""Streaming channel-macro validation errors; never scale model inputs."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainScale:
    std: tuple[float | None, ...]
    counts: tuple[int, ...]
    near_constant_threshold: float = 1e-8


def train_scale(values, train_end, *, split="train", threshold=1e-8):
    if split != "train" or not 0 < train_end <= len(values):
        raise ValueError("scale statistics require the declared train prefix")
    width = len(values[0])
    counts, means, m2 = [0] * width, [0.0] * width, [0.0] * width
    for index in range(train_end):
        row = values[index]
        if len(row) != width:
            raise ValueError("channel width mismatch")
        for c, value in enumerate(row):
            value = float(value)
            if math.isfinite(value):
                counts[c] += 1
                delta = value - means[c]
                means[c] += delta / counts[c]
                m2[c] += delta * (value - means[c])
    return TrainScale(
        tuple(math.sqrt(max(0.0, s / n)) if n else None for n, s in zip(counts, m2, strict=True)),
        tuple(counts),
        threshold,
    )


def macro(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    return math.fsum(finite) / len(finite) if finite else None


class StreamingMetrics:
    """Consume rows of channel errors, weighted by valid elements, not batch means."""

    def __init__(self, scale: TrainScale):
        self.scale = scale
        self.count = [0] * len(scale.std)
        self.absolute = [0.0] * len(scale.std)
        self.squared = [0.0] * len(scale.std)

    def update(self, predictions, targets):
        for pred, target in zip(predictions, targets, strict=True):
            if len(pred) != len(self.count) or len(target) != len(self.count):
                raise ValueError("metric channel width mismatch")
            for c, (p, y) in enumerate(zip(pred, target, strict=True)):
                if not math.isfinite(float(y)):
                    continue
                if not math.isfinite(float(p)):
                    raise ValueError("nonfinite prediction on a valid target")
                error = float(p) - float(y)
                self.count[c] += 1
                self.absolute[c] += abs(error)
                self.squared[c] += error * error

    def compute(self):
        per_channel = []
        for n, a, s, sd in zip(
            self.count, self.absolute, self.squared, self.scale.std, strict=True
        ):
            valid_scale = sd is not None and sd > self.scale.near_constant_threshold
            per_channel.append(
                {
                    "count": n,
                    "mae": a / n if n else None,
                    "mse": s / n if n else None,
                    "normalized_mae": a / n / sd if n and valid_scale else None,
                    "normalized_mse": s / n / (sd * sd) if n and valid_scale else None,
                    "normalization_eligible": valid_scale,
                }
            )
        names = ("mae", "mse", "normalized_mae", "normalized_mse")
        return {
            "per_channel": per_channel,
            "macro": {k: macro([c[k] for c in per_channel]) for k in names},
            "normalization": "train population std; normalize errors only",
            "excluded_normalized_channels": sum(
                not c["normalization_eligible"] for c in per_channel
            ),
        }

    def update_array(self, predictions, targets):
        """Vectorized FP64 sums for large multivariate test sets (optional NumPy)."""
        import numpy as np

        pred, target = (
            np.asarray(predictions, dtype="float64"),
            np.asarray(targets, dtype="float64"),
        )
        if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != len(self.count):
            raise ValueError("metric array shape mismatch")
        mask = np.isfinite(target)
        if not np.isfinite(pred[mask]).all():
            raise ValueError("nonfinite prediction on a valid target")
        error = np.where(mask, pred - target, 0.0)
        for c, (n, a, s) in enumerate(
            zip(mask.sum(0), np.abs(error).sum(0), np.square(error).sum(0), strict=True)
        ):
            self.count[c] += int(n)
            self.absolute[c] += float(a)
            self.squared[c] += float(s)

    def state(self):
        from dataclasses import asdict

        return {
            "scale": asdict(self.scale),
            "count": self.count.copy(),
            "absolute": self.absolute.copy(),
            "squared": self.squared.copy(),
        }

    def restore(self, state):
        from tsfm_crossover.data.common import stable_hash

        if stable_hash(state["scale"]) != stable_hash(self.state()["scale"]):
            raise ValueError("metric train scale mismatch")
        for key in ("count", "absolute", "squared"):
            values = state[key]
            if len(values) != len(self.count) or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("corrupt streaming metric state")
            if key == "count" and any(type(v) is not int for v in values):
                raise ValueError("metric counts must be integers")
            setattr(self, key, values.copy())


def relative_improvement(zero, few):
    # A zero denominator is undefined, not a manufactured perfect improvement.
    if zero is None or few is None or zero <= 0:
        return None
    return (zero - few) / zero


def dataset_macro(results):
    return {
        key: macro([r["macro"][key] for r in results])
        for key in ("mae", "mse", "normalized_mae", "normalized_mse")
    }
