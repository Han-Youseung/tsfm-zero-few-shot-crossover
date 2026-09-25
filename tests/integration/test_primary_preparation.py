import pytest

pd = pytest.importorskip("pandas")
from scripts.prepare_primary_candidates import regularize  # noqa: E402


def test_exact_duplicates_grid_and_mask():
    d = pd.DataFrame(
        {"date": ["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:20"], "x": [1, 1, -9999]}
    )
    result, audit = regularize(d, "10min", {"x": -9999})
    assert audit["rows_before"] == 3 and audit["rows_after_exact_dedup"] == 2
    assert audit["inserted_timestamps"] == 1
    assert result.x.isna().tolist() == [False, True, True]
    assert d.x.tolist() == [1, 1, -9999]


def test_conflicts_not_deleted():
    d = pd.DataFrame({"date": ["2020-01-01", "2020-01-01"], "x": [1, 2]})
    result, audit = regularize(d, "10min")
    assert result is None and audit["timestamp_conflict_rows"] == 2
