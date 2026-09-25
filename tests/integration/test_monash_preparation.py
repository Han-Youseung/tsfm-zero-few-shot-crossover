"""Optional NumPy parsing checks; no download and no actual dataset required."""

import pytest

pytest.importorskip("numpy")

from scripts.prepare_monash_variants import parse_tsf  # noqa: E402

HEADER = """@attribute series_name string
@attribute start_timestamp date
@missing false
@equallength true
@data
"""


def test_all_series_order_and_start_preserved():
    names, start, values = parse_tsf(
        HEADER + "a:2006-01-01 00-00-01:1,2\nb:2006-01-01 00-00-01:3,4"
    )
    assert names == ["a", "b"]
    assert start == "2006-01-01 00-00-01"
    assert values.tolist() == [[1, 3], [2, 4]]


@pytest.mark.parametrize("data", ["a:x:1,2\na:x:3,4", "a:x:1,2\nb:y:3,4", "a:x:nan,2", "a:x:1x,2"])
def test_invalid_tsf_rejected(data):
    with pytest.raises(ValueError):
        parse_tsf(HEADER + data)
