from dataclasses import replace

from tsfm_crossover.data.coverage import compute_coverage
from tsfm_crossover.data.sampling import (
    DEFAULT_RATES,
    build_sampling_manifest,
    temporal_master_order,
)
from tsfm_crossover.data.splits import Interval
from tsfm_crossover.data.windows import generate_train_windows


def windows(count_basis: int = 1000):
    return generate_train_windows("D", "sha", Interval(0, count_basis + 3), 2, 2)


def test_counts_nestedness_duplicates_and_reproducibility() -> None:
    items = windows()
    kwargs = dict(
        dataset_sha256="sha",
        split_config_hash="split",
        seed=42,
        code_commit_sha="code",
        generated_at="fixed",
    )
    first = build_sampling_manifest(items, **kwargs)
    second = build_sampling_manifest(items, **kwargs)
    assert first.manifest_hash == second.manifest_hash
    assert first.selections[0].selected_count == 0
    assert first.selections[-1].selected_count == len(items)
    assert first.nestedness_valid
    for selection in first.selections:
        assert len(selection.selected_window_ids) == len(set(selection.selected_window_ids))


def test_floor_counts_minimum_and_equal_k() -> None:
    manifest = build_sampling_manifest(
        windows(5),
        dataset_sha256="sha",
        split_config_hash="s",
        seed=1,
        code_commit_sha="c",
        generated_at="fixed",
    )
    positive = manifest.selections[1:]
    assert positive[0].selected_count == 1
    assert positive[0].selected_window_ids == positive[1].selected_window_ids


def test_different_seed_and_temporal_representativeness() -> None:
    items = windows(1000)
    a = temporal_master_order(items, 1, 20)
    b = temporal_master_order(items, 2, 20)
    assert a[:20] != b[:20]
    positions = sorted(a[:20])
    assert positions[0] < 100
    assert positions[-1] > 900


def test_coverage_disjoint_partial_and_complete_overlap() -> None:
    base = windows(20)[0]
    disjoint = [
        base,
        replace(base, context_start=10, context_end=12, target_start=12, target_end=14),
    ]
    stats = compute_coverage(disjoint, (0, 20))
    assert stats.unique_total_observed_time_points == 8
    assert stats.maximum_time_point_exposure_count == 1
    partial = [base, replace(base, context_start=1, context_end=3, target_start=3, target_end=5)]
    stats = compute_coverage(partial, (0, 10))
    assert stats.unique_total_observed_time_points == 5
    assert stats.maximum_time_point_exposure_count == 2
    same = compute_coverage([base, base], (0, 10))
    assert same.unique_total_observed_time_points == 4
    assert same.average_time_point_exposure_count == 2


def test_large_sampling_structure_100k() -> None:
    items = windows(100_000)
    order = temporal_master_order(items, 42)
    assert len(order) == 100_000
    assert len(set(order)) == 100_000
    stats = compute_coverage((items[i] for i in order[:1000]), (0, 100_003))
    assert stats.unique_total_observed_time_points > 0
    assert DEFAULT_RATES[-1] == 1.0
