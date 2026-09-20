import math

import pytest

from tsfm_crossover.data.canonical import (
    CanonicalizationError,
    canonical_fingerprint,
    canonicalize_long_records,
)
from tsfm_crossover.data.provenance import (
    ProvenanceEvidence,
    compare_numeric,
    diagnose_affine,
    validate_manifest_safety,
    validate_provenance_decision,
    validate_single_source_variant,
)


def records(channels=("a", "b")):
    return [
        {"date": date, "cols": channel, "data": value}
        for channel, values in zip(channels, ((1, 2), (3, 4)), strict=True)
        for date, value in zip(("2020-01-01", "2020-01-02"), values, strict=True)
    ]


def test_normal_long_to_wide_recovery():
    result = canonicalize_long_records(records())
    assert result.values == ((1.0, 3.0), (2.0, 4.0))


def test_composite_key_duplicate_is_blocked():
    rows = records() + [records()[0]]
    with pytest.raises(CanonicalizationError, match="identical duplicate"):
        canonicalize_long_records(rows)


def test_conflicting_duplicate_is_blocked():
    rows = records() + [{"date": "2020-01-01", "cols": "a", "data": 99}]
    with pytest.raises(CanonicalizationError, match="conflicting duplicate"):
        canonicalize_long_records(rows)


def test_missing_channel_time_is_blocked():
    with pytest.raises(CanonicalizationError, match="missing channel-time"):
        canonicalize_long_records(records()[:-1])


def test_channel_timestamp_mismatch_is_blocked():
    rows = records()
    rows[-1] = {"date": "2020-01-03", "cols": "b", "data": 4}
    with pytest.raises(CanonicalizationError, match="missing channel-time"):
        canonicalize_long_records(rows)


def test_element_count_and_multiset_are_preserved():
    result = canonicalize_long_records(records())
    assert result.validation_report["input_elements"] == 4
    assert result.validation_report["output_elements"] == 4
    assert result.validation_report["value_multiset_preserved"] is True


def test_first_appearance_channel_order_is_preserved():
    result = canonicalize_long_records(records(("z", "a")))
    assert result.channel_names == ("z", "a")


def test_explicit_official_channel_order_is_honored():
    result = canonicalize_long_records(records(), channel_order=("b", "a"))
    assert result.values == ((3.0, 1.0), (4.0, 2.0))


def test_fingerprint_is_stable():
    left = canonicalize_long_records(records(), channel_order=("a", "b"))
    right = canonicalize_long_records(list(reversed(records())), channel_order=("a", "b"))
    assert left.canonical_data_fingerprint == right.canonical_data_fingerprint


def test_fingerprint_changes_when_one_value_changes():
    rows = records()
    changed = [dict(row) for row in rows]
    changed[0]["data"] = 1.1
    assert (
        canonicalize_long_records(rows).canonical_data_fingerprint
        != canonicalize_long_records(changed).canonical_data_fingerprint
    )


def test_fingerprint_changes_with_channel_order():
    values = ((1.0, 3.0), (2.0, 4.0))
    timestamps = ("1", "2")
    assert canonical_fingerprint(timestamps, ("a", "b"), values) != canonical_fingerprint(
        timestamps, ("b", "a"), values
    )


def test_row_order_only_does_not_change_canonical_result():
    left = canonicalize_long_records(records(), channel_order=("a", "b"))
    shuffled = [records()[2], records()[0], records()[3], records()[1]]
    right = canonicalize_long_records(shuffled, channel_order=("a", "b"))
    assert left.timestamps == right.timestamps
    assert left.values == right.values


def test_unpadded_slash_timestamps_are_sorted_chronologically():
    rows = [
        {"date": "2020/10/1 0:00", "cols": "a", "data": 10},
        {"date": "2020/2/1 0:00", "cols": "a", "data": 2},
    ]
    result = canonicalize_long_records(rows)
    assert result.timestamps == ("2020/2/1 0:00", "2020/10/1 0:00")


def test_no_aggregation_is_performed():
    result = canonicalize_long_records(records())
    assert result.validation_report["aggregation_performed"] is False


def test_source_variant_mixing_is_blocked():
    with pytest.raises(ValueError, match="exactly one source variant"):
        validate_single_source_variant(["ETTh1__bundle_long", "ETTh1__official_raw"])


def test_one_source_variant_is_accepted():
    assert validate_single_source_variant(["ETTh1__official_raw"] * 2) == "ETTh1__official_raw"


def test_grade_a_requires_exact_official_evidence():
    with pytest.raises(ValueError, match="grade A"):
        validate_provenance_decision("A", "ready", ProvenanceEvidence())


def test_grade_b_requires_code_and_fingerprint_evidence():
    with pytest.raises(ValueError, match="grade B"):
        validate_provenance_decision("B", "ready", ProvenanceEvidence(code_verified_lossless=True))


def test_blocked_release_requires_a_or_b():
    with pytest.raises(ValueError, match="cannot be released"):
        validate_provenance_decision("C", "ready", ProvenanceEvidence())


def test_weather_shaped_21_channel_long_fixture_is_not_duplicate():
    rows = [
        {"date": date, "cols": f"c{channel}", "data": channel + offset}
        for channel in range(21)
        for offset, date in enumerate(("2020-01-01", "2020-01-02"))
    ]
    result = canonicalize_long_records(rows)
    assert len(result.channel_names) == 21
    assert result.validation_report["input_elements"] == 42


def test_exact_numeric_match():
    result = compare_numeric([1.0, 2.0], [1.0, 2.0])
    assert result.exact and result.allclose


def test_dtype_tolerance_allclose():
    result = compare_numeric([1.0], [1.0 + 1e-9])
    assert not result.exact and result.allclose


def test_affine_transform_diagnosis():
    result = diagnose_affine([1.0, 2.0, 3.0], [5.0, 7.0, 9.0])
    assert math.isclose(result.slope, 2.0)
    assert math.isclose(result.intercept, 3.0)
    assert result.max_absolute_residual == 0.0


def test_manifest_rejects_raw_value_payload():
    with pytest.raises(ValueError, match="raw value"):
        validate_manifest_safety({"raw_values": [1.0, 2.0]})


def test_manifest_rejects_absolute_path():
    with pytest.raises(ValueError, match="absolute path"):
        validate_manifest_safety({"source": "C:\\private\\data.csv"})


def test_manifest_accepts_hashes_and_aggregate_statistics():
    validate_manifest_safety({"sha256": "abc", "mean": 1.2, "source": "data/file.csv"})
