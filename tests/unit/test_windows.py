import pytest

from tsfm_crossover.data.splits import Interval
from tsfm_crossover.data.windows import generate_rolling_windows, generate_train_windows


def test_train_window_count_and_boundaries() -> None:
    windows = generate_train_windows("D", "sha", Interval(0, 60), 10, 5)
    assert len(windows) == 60 - 10 - 5 + 1
    assert all(w.context_start >= 0 and w.target_end <= 60 for w in windows)
    assert all(w.context_end == w.target_start for w in windows)


def test_validation_and_test_rolling_boundaries() -> None:
    validation = generate_rolling_windows("D", "sha", "validation", Interval(60, 80), 10, 5)
    test = generate_rolling_windows("D", "sha", "test", Interval(80, 100), 10, 5)
    assert validation[0].context_start == 50
    assert validation[0].target_start == 60
    assert validation[-1].target_end <= 80
    assert test[1].context_end == test[1].target_start == 81
    assert test[-1].target_end <= 100


def test_window_id_is_reproducible_and_fingerprint_sensitive() -> None:
    a = generate_train_windows("D", "sha", Interval(0, 20), 3, 2)[0]
    b = generate_train_windows("D", "sha", Interval(0, 20), 3, 2)[0]
    c = generate_train_windows("D", "other", Interval(0, 20), 3, 2)[0]
    assert a.window_id == b.window_id
    assert a.window_id != c.window_id


@pytest.mark.parametrize("kind", ["train", "validation"])
def test_too_long_raises(kind: str) -> None:
    with pytest.raises(ValueError, match="no .*windows"):
        if kind == "train":
            generate_train_windows("D", "sha", Interval(0, 5), 4, 2)
        else:
            generate_rolling_windows("D", "sha", "validation", Interval(2, 5), 4, 2)
