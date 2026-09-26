"""CPU design preflight: count train/validation opportunities, never parse test values."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_crossover.data.missing import POLICY, causal_context, eligible_windows
from tsfm_crossover.data.pilot_data import load_pilot_values
from tsfm_crossover.data.sampling import selected_count
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.tracking.atomic import write_json_atomic


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optimizer: Literal["AdamW"]
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    samples: int = Field(ge=1)
    point_statistic: Literal["prediction_outputs", "torch_sample_median"]


class StudyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["primary9_bounded_compute_v1"]
    prepared_manifest: Literal["results/manifests/pilot/prepared_primary9.json"]
    context: Literal[512]
    horizons: tuple[int, ...]
    sampling_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    max_optimizer_steps: Literal[1000]
    eval_every_steps: Literal[100]
    patience_evals: Literal[3]
    validation_windows: Literal[64]
    batch_size: Literal[1]
    gradient_accumulation: Literal[1]
    gradient_clip: None
    drop_last: Literal[False]
    dtype: Literal["float32"]
    external_scaler: Literal[False]
    primary_metric: Literal["channel_macro_train_std_normalized_mae"]
    test_stride: Literal[1]
    models: dict[str, ModelSettings]
    test_evaluation: Literal[False]
    protocol_frozen: Literal[False]
    main_experiment_allowed: Literal[False]

    @model_validator(mode="after")
    def fixed_design(self):
        if self.horizons != (96, 192, 336, 720) or self.sampling_rates != (
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
        ):
            raise ValueError("approved common grid required")
        if self.seeds != (1729, 2718, 31415) or set(self.models) != {"ttm", "moirai1"}:
            raise ValueError("approved seeds and both models required")
        for family, lr, decay, samples, point in (
            ("ttm", 1e-4, 0.01, 1, "prediction_outputs"),
            ("moirai1", 5e-6, 0.1, 100, "torch_sample_median"),
        ):
            m = self.models[family]
            if (m.learning_rate, m.weight_decay, m.samples, m.point_statistic) != (
                lr,
                decay,
                samples,
                point,
            ):
                raise ValueError("model settings differ from reviewed validation decision")
        return self


def preflight(root, config_path):
    config = StudyPlan.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    prepared_path = root / config.prepared_manifest
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if set(prepared) != {
        "ETTh1",
        "ETTh2",
        "ETTm1",
        "ETTm2",
        "Electricity",
        "Solar",
        "Weather",
        "Tetouan",
        "Traffic",
    }:
        raise ValueError("primary9 set mismatch")
    rows = []
    for name, entry in prepared.items():
        values, channels, split = load_pilot_values(root, entry)
        context = causal_context(values) if entry.get("missing_policy") == POLICY else values
        for horizon in config.horizons:
            train = generate_train_windows(
                entry["variant"], entry["qc"]["sha256"], split.train, 512, horizon
            )
            raw_count = len(train)
            train = eligible_windows(train, values, context, training=True)
            valid = generate_rolling_windows(
                entry["variant"],
                entry["qc"]["sha256"],
                "validation",
                split.validation,
                512,
                horizon,
            )
            valid = eligible_windows(valid, values, context, training=False)
            if not train or len(valid) < config.validation_windows:
                raise ValueError("insufficient common train/validation windows")
            rows.append(
                {
                    "dataset": name,
                    "horizon": horizon,
                    "channels": len(channels),
                    "data_sha256": entry["qc"]["sha256"],
                    "split": split.as_dict(),
                    "raw_train_candidates": raw_count,
                    "eligible_train_candidates": len(train),
                    "selected_counts": {
                        str(r): selected_count(r, len(train)) for r in config.sampling_rates
                    },
                    "eligible_validation_windows": len(valid),
                    "validation_subset_size": 64,
                    "test_window_count_nominal": split.test.end - split.test.start - horizon + 1,
                    "test_values_read": False,
                }
            )
        print(name, "train/validation preflight passed", flush=True)
    return {
        "status": "cpu_design_preflight_passed",
        "config": config.model_dump(mode="json"),
        "source_sha256": {
            "config": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "prepared": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
            "planner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "windows": rows,
        "zero_shot_seed_evaluations": 216,
        "few_shot_runs": 1728,
        "total_conditions": 1944,
        "maximum_optimizer_steps_total": 1728000,
        "test_counts": "metadata-only nominal; missing targets masked only at final evaluation",
        "test_evaluation": False,
        "protocol_frozen": False,
        "main_experiment_allowed": False,
        "remaining_gate": "main_runner_and_analysis_integration_not_validated",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/study/preexperiment.yaml"))
    p.add_argument(
        "--output", type=Path, default=Path("results/manifests/pilot/study_preflight.json")
    )
    args = p.parse_args()
    result = preflight(Path.cwd(), args.config)
    if args.output.exists():
        # JSON serializes SplitManifest.ratios tuples as lists.
        if json.loads(args.output.read_text(encoding="utf-8")) != json.loads(json.dumps(result)):
            raise ValueError("refusing to replace preflight evidence")
    else:
        write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
