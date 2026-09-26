"""Small predeclared train-only descriptors, not selected using test correlations."""

import argparse
import hashlib
import math
import subprocess
from pathlib import Path

from tsfm_crossover.data.pilot_data import load_pilot_values
from tsfm_crossover.evaluation.metrics import macro
from tsfm_crossover.experiments.main_plan import load_plan
from tsfm_crossover.tracking.atomic import write_json_atomic


def features(values, train_end):
    if not 1 < train_end <= len(values):
        raise ValueError("train prefix required")
    width = len(values[0])
    missing = constants = 0
    correlations, trends = [], []
    for channel in range(width):
        observed = [
            (t, float(values[t][channel]))
            for t in range(train_end)
            if math.isfinite(values[t][channel])
        ]
        n = len(observed)
        missing += train_end - n
        if n < 2:
            continue
        mean = math.fsum(v for _, v in observed) / n
        variance = math.fsum((v - mean) ** 2 for _, v in observed) / n
        if variance <= 1e-16:
            constants += 1
            continue
        mean_t = math.fsum(t for t, _ in observed) / n
        denom = math.fsum((t - mean_t) ** 2 for t, _ in observed)
        slope = math.fsum((t - mean_t) * (v - mean) for t, v in observed) / denom
        trends.append(abs(slope) * (train_end - 1) / math.sqrt(variance))
        pairs = [
            (a[1], b[1]) for a, b in zip(observed, observed[1:], strict=False) if b[0] == a[0] + 1
        ]
        if len(pairs) > 1:
            mx, my = (math.fsum(p[i] for p in pairs) / len(pairs) for i in (0, 1))
            dx = math.fsum((x - mx) ** 2 for x, _ in pairs)
            dy = math.fsum((y - my) ** 2 for _, y in pairs)
            if dx * dy > 0:
                correlations.append(
                    math.fsum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(dx * dy)
                )
    return {
        "scope": "train_only_raw_observations",
        "train_rows": train_end,
        "channels": width,
        "missing_rate": missing / (train_end * width),
        "constant_channels": constants,
        "channel_macro_lag1_correlation": macro(correlations),
        "channel_macro_absolute_linear_trend_in_train_std": macro(trends),
        "seasonality": "not_estimated_by_this_descriptor_set",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd()
    _, prepared = load_plan(root, root / "configs/study/main.yaml")
    result = {}
    for name, entry in prepared.items():
        values, _, split = load_pilot_values(root, entry)
        result[name] = features(values, split.train.end) | {"fingerprint": entry["qc"]["sha256"]}
    write_json_atomic(
        args.output,
        {
            "features": result,
            "test_values_read": False,
            "descriptor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "repository_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "tracked_source_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], text=True
                ).strip()
            ),
            "status": "cpu_train_descriptors_computed_not_main_performance",
        },
    )


if __name__ == "__main__":
    main()
