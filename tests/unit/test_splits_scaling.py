from datetime import datetime, timedelta

import pytest

from tsfm_crossover.data.scaling import TrainOnlyStandardScaler
from tsfm_crossover.data.splits import Interval, chronological_split


@pytest.mark.parametrize(("size", "ends"), [(100, (60, 80)), (11, (6, 8))])
def test_floor_split_boundaries(size: int, ends: tuple[int, int]) -> None:
    split = chronological_split(size)
    assert (split.train.end, split.validation.end) == ends
    assert split.train.end == split.validation.start
    assert split.validation.end == split.test.start
    assert split.train.rows + split.validation.rows + split.test.rows == size


def test_split_boundary_timestamps() -> None:
    stamps = tuple(datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10))
    split = chronological_split(10, stamps)
    assert split.boundary_timestamps["validation_start"] == stamps[6].isoformat()


def test_scaler_train_only_and_inverse() -> None:
    first = ((0.0, 5.0), (2.0, 5.0), (1000.0, 7.0), (2000.0, 8.0))
    changed = first[:2] + ((-9999.0, 50.0), (9999.0, 80.0))
    one = TrainOnlyStandardScaler.fit(first, Interval(0, 2))
    two = TrainOnlyStandardScaler.fit(changed, Interval(0, 2))
    assert one.means == two.means == (1.0, 5.0)
    assert one.constant_channels == (1,)
    transformed = one.transform(first)
    assert one.inverse_transform(transformed) == first


def test_scaler_rejects_empty_train() -> None:
    with pytest.raises(ValueError, match="no rows"):
        TrainOnlyStandardScaler.fit(((1.0,),), Interval(0, 0))
