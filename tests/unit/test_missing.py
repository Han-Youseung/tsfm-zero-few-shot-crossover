import math

import pytest

from tsfm_crossover.data.missing import causal_context, eligible_windows
from tsfm_crossover.data.splits import Interval
from tsfm_crossover.data.windows import generate_train_windows


def test_forward_only_and_original_targets_unchanged():
    nan = float("nan")
    raw = [[nan, 1], [2, nan], [nan, 3], [999, 999]]
    out = causal_context(raw)
    assert math.isnan(out[0][0])
    assert out[1:3] == [[2, 1], [2, 3]]
    assert math.isnan(raw[2][0])
    prefix = causal_context(raw[:3])
    assert math.isnan(prefix[0][0]) and prefix[1:] == out[1:3]


def test_filter_before_sampling_and_leading_blocked():
    raw = [[float("nan")], [1], [2], [float("nan")], [4], [5], [6], [7]]
    context = causal_context(raw)
    windows = generate_train_windows("fixture", "a" * 64, Interval(0, 8), 2, 2)
    train = eligible_windows(windows, raw, context, training=True)
    assert [w.target_start for w in train] == [4, 5, 6]
    valid = eligible_windows(windows, raw, context, training=False)
    assert [w.target_start for w in valid] == [3, 4, 5, 6]


def test_no_implicit_infinite_imputation():
    with pytest.raises(ValueError, match="infinite"):
        causal_context([[1], [float("inf")]])


def test_loader_never_parses_test_targets(tmp_path):
    from tsfm_crossover.data.missing import POLICY
    from tsfm_crossover.data.pilot_data import digest, load_pilot_values

    path = tmp_path / "fixture.csv"
    # 60/20/20: last two target values deliberately cannot be parsed.
    path.write_text("date,x\n" + "\n".join(f"{i},{i if i < 8 else 'TEST'}" for i in range(10)))
    entry = {
        "status": "ready_with_warnings",
        "variant": "fixture",
        "path": "fixture.csv",
        "missing_policy": POLICY,
        "qc": {"sha256": digest(path), "rows": 10, "channel_names": ["x"]},
    }
    values, _, split = load_pilot_values(tmp_path, entry)
    assert len(values) == split.validation.end == 8
