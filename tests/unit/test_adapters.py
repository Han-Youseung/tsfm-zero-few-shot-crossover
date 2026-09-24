import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from tsfm_crossover.data.sampling import build_sampling_manifest
from tsfm_crossover.data.splits import chronological_split
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows
from tsfm_crossover.models.adapter_config import AdapterConfig
from tsfm_crossover.models.window_batch import make_window_batch, select_train_windows

ROOT = Path(__file__).resolve().parents[2]


def adapter_config(**changes):
    settings = dict(
        family="ttm",
        condition_id="fixture-rate-005",
        dataset_fingerprint="a" * 64,
        sampling_manifest_hash="b" * 64,
        channel_names=("a", "b", "c"),
        horizon=96,
        num_samples=1,
        prediction_seed=11,
        training_seed=12,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_optimizer_steps=2,
        point_statistic="prediction_outputs",
    )
    return AdapterConfig(**(settings | changes))


def test_optional_import_does_not_load_vendor_or_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tsfm_crossover.models.adapters import create_adapter; "
            "assert not any(x in sys.modules for x in ['torch','uni2ts','tsfm_public'])",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"channel_names": ("a", "a")},
        {"num_samples": 8},
        {"dtype": "float16"},
        {"external_scaler": True},
        {"point_statistic": "torch_sample_median"},
    ],
)
def test_invalid_adapter_config(changes):
    with pytest.raises(ValueError):
        adapter_config(**changes)


def test_existing_sampling_and_windows_connect_without_dropping_last():
    split = chronological_split(1500)
    windows = generate_train_windows("fixture", "a" * 64, split.train, 512, 96)
    manifest = build_sampling_manifest(
        windows,
        dataset_sha256="a" * 64,
        split_config_hash=split.split_config_hash,
        seed=7,
        code_commit_sha="c" * 40,
        rates=[0.005, 0.01],
    )
    selected = select_train_windows(windows, manifest, 0.005)
    assert len(selected) == 1
    assert set(w.window_id for w in selected) <= set(
        w.window_id for w in select_train_windows(windows, manifest, 0.01)
    )
    values = [[1.0, 2.0, 3.0]] * split.validation.end
    batch = make_window_batch(
        values,
        selected,
        splits=split,
        channel_names=("a", "b", "c"),
        fingerprint="a" * 64,
        sampling_manifest_hash=manifest.manifest_hash,
    )
    assert len(batch.past) == 1 and len(batch.future[0]) == 96
    bad = dataclasses.replace(selected[0], target_end=split.train.end + 1)
    with pytest.raises(ValueError, match="boundary"):
        make_window_batch(
            values,
            [bad],
            splits=split,
            channel_names=batch.channel_names,
            fingerprint="a" * 64,
            sampling_manifest_hash=manifest.manifest_hash,
        )
    test = generate_rolling_windows("fixture", "a" * 64, "test", split.test, 512, 96)[0]
    with pytest.raises(ValueError, match="test split"):
        make_window_batch(
            values,
            [test],
            splits=split,
            channel_names=batch.channel_names,
            fingerprint="a" * 64,
            sampling_manifest_hash=manifest.manifest_hash,
        )
    with pytest.raises(ValueError, match="hash"):
        select_train_windows(windows, dataclasses.replace(manifest, manifest_hash="0" * 64), 0.005)
