"""An explicit finite execution plan, independent of test performance."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.pilot_data import TARGETS


class PilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = "phase6_validation_only"
    context: Literal[512] = 512
    horizons: tuple[Literal[96, 192, 336, 720], ...] = (96, 192, 336, 720)
    representatives: tuple[str, ...] = ("ETTh1", "Electricity")
    sampling_rate: float = Field(default=0.05, gt=0, le=1)
    seed: int = 1729
    max_steps: int = Field(default=200, ge=1, le=200)
    eval_every: int = Field(default=50, ge=1)
    patience_evals: int = Field(default=4, ge=1)
    checkpoint_every: int = Field(default=10, ge=1)
    validation_windows: int = Field(default=64, ge=1, le=64)
    training_batch: int = Field(default=1, ge=1, le=64)
    evaluation_batch: Literal[1] = 1
    accumulation: Literal[1] = 1
    learning_rates: dict[str, list[float]] = {
        "ttm": [1e-6, 1e-5, 1e-4],
        "moirai1": [5e-7, 5e-6, 5e-5],
    }
    num_samples: Literal[100] = 100
    batch_candidates: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    memory_fraction: float = Field(default=0.75, gt=0, lt=1)
    amp_probe: bool = False
    selection_metric: Literal["normalized_mae"] = "normalized_mae"
    dtype: Literal["float32"] = "float32"
    test_evaluation: Literal[False] = False
    protocol_frozen: Literal[False] = False

    @model_validator(mode="after")
    def bounded(self):
        if set(self.learning_rates) != {"ttm", "moirai1"}:
            raise ValueError("both models required")
        if any(
            not 1 <= len(v) <= 3
            or len(set(v)) != len(v)
            or any(not math.isfinite(x) or x <= 0 for x in v)
            for v in self.learning_rates.values()
        ):
            raise ValueError("one to three finite positive LR candidates per model")
        if tuple(self.batch_candidates) != (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("bounded batch ladder required")
        if len(self.representatives) != 2 or any(x not in TARGETS for x in self.representatives):
            raise ValueError("two named representatives required")
        return self


def token_risk(channels, horizon):
    time_tokens = 8 + math.ceil(horizon / 64)
    tokens = channels * time_tokens
    return {
        "moirai_time_tokens": time_tokens,
        "moirai_joint_tokens": tokens,
        "moirai_attention_pairs_per_sample": tokens * tokens,
        "moirai_max_seq_len_config": 512,
        "hard_joint_512_token_limit": False,
        "interpretation": "RoPE uses time_id and grows; dense attention remains quadratic",
        "ttm_channels": channels,
        "ttm_external_channel_split": False,
        "runtime_status": "pending_gpu",
    }


def execution_plan(config, prepared, commit):
    rows = []
    for name in TARGETS:
        entry = prepared.get(name, {})
        ready = entry.get("status") == "ready_with_warnings"
        for family in config.learning_rates:
            for horizon in config.horizons:
                row = {
                    "kind": "feasibility",
                    "dataset": name,
                    "family": family,
                    "horizon": horizon,
                    "status": "planned" if ready else "blocked_data",
                }
                rows.append(row)
            if name in config.representatives:
                for lr in config.learning_rates[family]:
                    rows.append(
                        {
                            "kind": "stability",
                            "dataset": name,
                            "family": family,
                            "horizon": 96,
                            "learning_rate": lr,
                            "status": "planned" if ready else "blocked_data",
                        }
                    )
    for row in rows:
        entry = prepared.get(row["dataset"], {})
        row["id"] = stable_hash(
            {
                "condition": row,
                "config": config.model_dump(mode="json"),
                "commit": commit,
                "data": entry.get("qc", {}).get("sha256"),
            }
        )
    return {
        "execution_commit": commit,
        "config": config.model_dump(mode="json"),
        "planned_groups": sum(r["status"] == "planned" for r in rows),
        "blocked_groups": sum(r["status"] != "planned" for r in rows),
        "maximum_batch_attempts_per_feasibility_group": 14 + (2 if config.amp_probe else 0),
        "conditions": rows,
        "protocol_frozen": False,
    }
