"""Nested, temporally stratified few-shot window sampling."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from .common import stable_hash
from .windows import WindowIndex

DEFAULT_RATES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def selected_count(rate: float, total: int) -> int:
    if not 0 <= rate <= 1:
        raise ValueError("sampling rates must be in [0, 1]")
    return 0 if rate == 0 else max(1, math.floor(rate * total))


def _spread_order(size: int, seed: int) -> list[int]:
    """Return a seeded low-discrepancy permutation of contiguous strata."""
    if size < 1:
        return []
    bits = (size - 1).bit_length()
    mask = (1 << bits) - 1
    scramble = random.Random(seed).randrange(mask + 1)
    ranked = []
    for value in range(mask + 1):
        reversed_bits = int(f"{value:0{bits}b}"[::-1], 2) if bits else 0
        candidate = reversed_bits ^ scramble
        if candidate < size:
            ranked.append(candidate)
    return ranked


def temporal_master_order(
    windows: list[WindowIndex], seed: int, temporal_strata: int | None = None
) -> list[int]:
    total = len(windows)
    if total == 0:
        return []
    strata_count = min(temporal_strata or max(1, math.ceil(math.sqrt(total))), total)
    strata: list[list[int]] = [[] for _ in range(strata_count)]
    for index in range(total):
        strata[min(strata_count - 1, index * strata_count // total)].append(index)
    for stratum_id, members in enumerate(strata):
        random.Random(stable_hash([seed, stratum_id])).shuffle(members)
    stratum_order = _spread_order(strata_count, seed)
    result: list[int] = []
    for round_index in range(max(map(len, strata))):
        round_order = (
            stratum_order[round_index % strata_count :]
            + stratum_order[: round_index % strata_count]
        )
        for stratum_id in round_order:
            if round_index < len(strata[stratum_id]):
                result.append(strata[stratum_id][round_index])
    if len(result) != total or len(set(result)) != total:
        raise AssertionError("master ordering is not a permutation")
    return result


@dataclass(frozen=True)
class RateSelection:
    requested_rate: float
    selected_count: int
    effective_rate: float
    selected_window_ids: tuple[str, ...]


@dataclass(frozen=True)
class SamplingManifest:
    dataset: str
    dataset_sha256: str
    split_config_hash: str
    context_length: int
    horizon: int
    sampling_strategy: str
    sampling_seed: int
    temporal_strata: int
    total_train_windows: int
    requested_sampling_rates: tuple[float, ...]
    selections: tuple[RateSelection, ...]
    nestedness_valid: bool
    generated_at: str
    code_commit_sha: str
    manifest_hash: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_sampling_manifest(
    windows: list[WindowIndex],
    *,
    dataset_sha256: str,
    split_config_hash: str,
    seed: int,
    code_commit_sha: str,
    rates: Iterable[float] = DEFAULT_RATES,
    temporal_strata: int | None = None,
    generated_at: str | None = None,
) -> SamplingManifest:
    if not windows:
        raise ValueError("cannot sample an empty train-window collection")
    normalized_rates = tuple(sorted(set(rates)))
    strata = min(temporal_strata or max(1, math.ceil(math.sqrt(len(windows)))), len(windows))
    order = temporal_master_order(windows, seed, strata)
    selections = tuple(
        RateSelection(
            rate,
            (count := selected_count(rate, len(windows))),
            count / len(windows),
            tuple(windows[i].window_id for i in order[:count]),
        )
        for rate in normalized_rates
    )
    nested = all(
        set(left.selected_window_ids).issubset(right.selected_window_ids)
        for left, right in zip(selections, selections[1:], strict=False)
    )
    payload = {
        "dataset": windows[0].dataset,
        "dataset_sha256": dataset_sha256,
        "split_config_hash": split_config_hash,
        "context_length": windows[0].context_length,
        "horizon": windows[0].horizon,
        "sampling_strategy": "nested_temporally_stratified_master_prefix_v1",
        "sampling_seed": seed,
        "temporal_strata": strata,
        "total_train_windows": len(windows),
        "requested_sampling_rates": normalized_rates,
        "selections": [asdict(item) for item in selections],
        "nestedness_valid": nested,
        "code_commit_sha": code_commit_sha,
    }
    return SamplingManifest(
        dataset=windows[0].dataset,
        dataset_sha256=dataset_sha256,
        split_config_hash=split_config_hash,
        context_length=windows[0].context_length,
        horizon=windows[0].horizon,
        sampling_strategy="nested_temporally_stratified_master_prefix_v1",
        sampling_seed=seed,
        temporal_strata=strata,
        total_train_windows=len(windows),
        requested_sampling_rates=normalized_rates,
        selections=selections,
        nestedness_valid=nested,
        code_commit_sha=code_commit_sha,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        manifest_hash=stable_hash(payload),
    )
