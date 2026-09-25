"""Connect existing WindowIndex and SamplingManifest to a model-neutral batch."""

from dataclasses import dataclass
from typing import Any

from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.data.sampling import SamplingManifest
from tsfm_crossover.data.splits import SplitManifest
from tsfm_crossover.data.windows import WindowIndex


@dataclass(frozen=True)
class WindowBatch:
    past: Any
    future: Any | None
    split: str
    channel_names: tuple[str, ...]
    window_ids: tuple[str, ...]
    dataset_fingerprint: str
    sampling_manifest_hash: str


def select_train_windows(
    windows: list[WindowIndex], manifest: SamplingManifest, rate: float
) -> list[WindowIndex]:
    payload = manifest.as_dict()
    payload.pop("manifest_hash")
    payload.pop("generated_at")
    if stable_hash(payload) != manifest.manifest_hash:
        raise ValueError("sampling manifest hash mismatch")
    if len(windows) != manifest.total_train_windows or any(w.split != "train" for w in windows):
        raise ValueError("train candidates do not match sampling manifest")
    by_id = {w.window_id: w for w in windows}
    if len(by_id) != len(windows):
        raise ValueError("duplicate window IDs")
    chosen = next((s for s in manifest.selections if s.requested_rate == rate), None)
    if chosen is None:
        raise ValueError("rate absent from sampling manifest")
    if chosen.selected_count != len(chosen.selected_window_ids):
        raise ValueError("selected count mismatch")
    try:
        return [by_id[i] for i in chosen.selected_window_ids]
    except KeyError as exc:
        raise ValueError("selected window absent from candidates") from exc


def make_window_batch(
    values,
    windows: list[WindowIndex],
    *,
    splits: SplitManifest,
    channel_names: tuple[str, ...],
    fingerprint: str,
    sampling_manifest_hash: str,
    context_values=None,
) -> WindowBatch:
    """Only train/validation engineering windows; no scaler or new sampling logic."""
    if not windows or len({w.split for w in windows}) != 1:
        raise ValueError("a nonempty batch from one split is required")
    split = windows[0].split
    if split not in {"train", "validation"}:
        raise ValueError("test split is blocked in adapter engineering")
    bounds = getattr(splits, split)
    for w in windows:
        if not (
            0 <= w.context_start < w.context_end == w.target_start < w.target_end <= bounds.end
            and bounds.start <= w.target_start
            and w.context_end - w.context_start == w.context_length
            and w.target_end - w.target_start == w.horizon
            and (split != "train" or w.context_start >= bounds.start)
        ):
            raise ValueError("window crosses split boundary or has invalid lengths")
        if w.target_end > len(values):
            raise ValueError("insufficient observed rows")
        expected = stable_hash(
            {
                "dataset_fingerprint": fingerprint,
                "split": split,
                "context_length": w.context_length,
                "horizon": w.horizon,
                "context_start": w.context_start,
                "target_start": w.target_start,
            }
        )
        if expected != w.window_id:
            raise ValueError("window fingerprint/ID mismatch")
    inputs = values if context_values is None else context_values
    past = [inputs[w.context_start : w.context_end] for w in windows]
    future = [values[w.target_start : w.target_end] for w in windows]
    return WindowBatch(
        past,
        future,
        split,
        channel_names,
        tuple(w.window_id for w in windows),
        fingerprint,
        sampling_manifest_hash,
    )
