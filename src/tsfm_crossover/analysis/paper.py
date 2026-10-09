"""Evidence-bound two-model H96 analysis; never trains or selects a checkpoint."""

import argparse
import hashlib
import itertools
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

from tsfm_crossover.analysis.crossover import interval, source_group
from tsfm_crossover.data.common import stable_hash
from tsfm_crossover.experiments.main_review import compact_result
from tsfm_crossover.experiments.pilot_review import read_archive, require
from tsfm_crossover.tracking.atomic import write_json_atomic

METRICS = ("normalized_mae", "normalized_mse", "mae", "mse")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_scope(path):
    scope = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(scope["schema_version"] == 1, "analysis schema")
    require(scope["families"] == ["ttm", "moirai1"], "two-model scope required")
    require(scope["horizon"] == 96 and scope["context"] == 512, "analysis horizon/context")
    require(scope["primary_metric"] == "normalized_mae", "primary metric cannot change")
    require(not scope["timer_included"] and not scope["execution_protocol_changed"], "scope")
    require(scope["rates"][0] == 0 and scope["rates"] == sorted(set(scope["rates"])), "rates")
    require(len(scope["seeds"]) == 3 and len(set(scope["seeds"])) == 3, "three unique seeds")
    return scope


def expected_keys(scope):
    return set(
        itertools.product(scope["families"], scope["datasets"], scope["seeds"], scope["rates"])
    )


def condition_key(record):
    c = record["identity"]["condition"]
    return c["family"], c["dataset"], c["seed"], c["rate"]


def load_verified(scope, root, archives):
    """Bind pinned canonical review bytes to unchanged original archive/result bytes."""
    compact, expected_archives, raw = {}, {}, {}
    for ref in scope["reviews"]:
        path = root / ref["path"]
        require(digest(path) == ref["sha256"], "review bytes changed")
        review = read(path)
        require(review["status"].startswith("validated_"), "review not validated")
        require(review["execution_commit"] == scope["execution_commit"], "review commit")
        require(review["validated_conditions"] == len(review["conditions"]), "review count")
        for item in review["archives"]:
            name, value = item["name"], item["sha256"]
            require(name not in expected_archives or expected_archives[name] == value, "ZIP hash")
            expected_archives[name] = value
        for row in review["conditions"]:
            cid = row["identity"]["condition"]["id"]
            require(cid not in compact, "duplicate review condition")
            compact[cid] = row
    require(len({p.name for p in archives}) == len(archives), "duplicate source ZIP name")
    require({p.name for p in archives} == set(expected_archives), "source ZIP inventory")
    for path in archives:
        require(digest(path) == expected_archives[path.name], "source ZIP bytes changed")
        for name, row in read_archive(path).items():
            if not name.endswith("/result.json"):
                continue
            cid = row["identity"]["condition"]["id"]
            require(name == cid + "/result.json", "member identity")
            require(cid not in raw or raw[cid] == row, "conflicting raw duplicate")
            require(cid in compact, "unreviewed source condition")
            require(compact_result(row) == compact[cid], "raw/review content mismatch")
            raw[cid] = row
    require(set(raw) == set(compact), "missing raw condition")
    validate_records(list(raw.values()), scope)
    return list(raw.values()), expected_archives


def validate_records(records, scope):
    keys = [condition_key(r) for r in records]
    require(len(keys) == len(set(keys)), "duplicate condition")
    require(set(keys) == expected_keys(scope), "incomplete or out-of-scope grid")
    for r in records:
        identity, test = r["identity"], r["test"]
        require(identity["commit"] == scope["execution_commit"], "execution commit")
        require(identity["condition"]["horizon"] == scope["horizon"], "horizon")
        require(r["status"] == "completed" and r["test_evaluation"] is True, "not final test")
        require(r["protocol_frozen"] is True and r["external_scaler"] is False, "protocol")
        require(
            test["no_update_verified"] is True
            and test["parameter_hash"] == r["selection"]["parameter_hash"],
            "test no-update evidence",
        )
        require(test["stride"] == 1, "rolling stride")
        for metric in METRICS:
            value = test["metrics"]["macro"][metric]
            require(value is not None and math.isfinite(value) and value >= 0, "invalid metric")
    for field in ("prepared_hash", "protocol_hash"):
        require(len({r["identity"][field] for r in records}) == 1, "shared identity mismatch")
    lookup = {condition_key(r): r for r in records}
    for d in scope["datasets"]:
        rows = [r for r in records if condition_key(r)[1] == d]
        signatures = {
            (
                r["identity"]["data_sha256"],
                r["test"]["window_hash"],
                r["test"]["number_of_test_windows"],
                tuple(c["count"] for c in r["test"]["metrics"]["per_channel"]),
                stable_hash(r["provenance"]["metric_train_scale"]),
            )
            for r in rows
        }
        require(len(signatures) == 1, "unpaired windows/masks/metric scale")
        for seed, rate in itertools.product(scope["seeds"], scope["rates"]):
            a, b = (lookup[f, d, seed, rate]["provenance"] for f in scope["families"])
            for name in ("sampling_manifest_hash", "selected_train_windows", "total_train_windows"):
                require(a[name] == b[name], "unpaired train sampling")


def paired_curve(lookup, family, dataset, seeds, rates, metric, channel=None):
    def metric_of(record):
        values = record["test"]["metrics"]
        return (values["macro"] if channel is None else values["per_channel"][channel])[metric]

    curve = []
    for rate in rates:
        pairs = []
        for seed in seeds:
            zero = metric_of(lookup[family, dataset, seed, 0.0])
            few = metric_of(lookup[family, dataset, seed, rate])
            require(
                zero is not None
                and math.isfinite(zero)
                and zero > 0
                and few is not None
                and math.isfinite(few)
                and few >= 0,
                "undefined baseline or invalid paired metric",
            )
            pairs.append(
                {
                    "seed": seed,
                    "zero": zero,
                    "few": few,
                    "delta": zero - few,
                    "relative_improvement": (zero - few) / zero,
                }
            )
        deltas = [p["delta"] for p in pairs]
        relative = [p["relative_improvement"] for p in pairs]
        curve.append(
            {
                "rate": rate,
                "paired": pairs,
                "zero_mean": st.fmean(p["zero"] for p in pairs),
                "few_mean": st.fmean(p["few"] for p in pairs),
                "mean_delta": st.fmean(deltas),
                "mean_relative_improvement": st.fmean(relative),
                "seed_min_relative_improvement": min(relative),
                "seed_max_relative_improvement": max(relative),
                "improved_seed_count": sum(v > 0 for v in deltas),
            }
        )
    return curve


def boundaries(curve):
    rates = [c["rate"] for c in curve]
    return {
        "mean": interval(rates, [c["mean_delta"] > 0 for c in curve]),
        "all_seeds": interval(rates, [c["seed_min_relative_improvement"] > 0 for c in curve]),
    }


def budget_summary(rows):
    few = [r for r in rows if condition_key(r)[3] > 0]
    full = [r for r in few if condition_key(r)[3] == 1.0]

    def ranges(values):
        return {"min": min(values), "mean": st.fmean(values), "max": max(values)}

    return {
        "fine_tuned_conditions": len(few),
        "reached_max_steps": sum(r["training"]["actual_optimizer_steps"] == 1000 for r in few),
        "best_step_at_budget_cap": sum(r["training"]["best_validation_step"] == 1000 for r in few),
        "full_pool_conditions": [
            {
                "seed": condition_key(r)[2],
                "selected_pool": r["provenance"]["selected_train_windows"],
                "actual_steps": r["training"]["actual_optimizer_steps"],
                "actual_unique_windows": r["training"]["actual_unique_train_windows"],
                "unique_fraction_of_selected_pool": r["training"]["actual_unique_train_windows"]
                / r["provenance"]["selected_train_windows"],
                "equivalent_epochs": r["training"]["equivalent_epochs"],
            }
            for r in full
        ],
        "full_pool_unique_fraction": ranges(
            [
                r["training"]["actual_unique_train_windows"]
                / r["provenance"]["selected_train_windows"]
                for r in full
            ]
        ),
        "actual_steps": ranges([r["training"]["actual_optimizer_steps"] for r in few]),
        "convergence_established": False,
    }


def source_aggregate(groups, scope):
    results = []
    for family in scope["families"]:
        for rate in scope["rates"][1:]:
            source_values = defaultdict(list)
            for g in groups:
                if g["family"] == family:
                    value = next(
                        c["mean_relative_improvement"]
                        for c in g["metrics"]["normalized_mae"]["curve"]
                        if c["rate"] == rate
                    )
                    source_values[source_group(g["dataset"])].append(value)
            means = {s: st.fmean(v) for s, v in source_values.items()}
            require(len(means) == 3, "expected three distinct source groups")
            results.append(
                {
                    "family": family,
                    "rate": rate,
                    "source_group_means": means,
                    "source_equal_weight_mean": st.fmean(means.values()),
                    "dataset_equal_weight_mean": st.fmean(
                        v for vs in source_values.values() for v in vs
                    ),
                    "leave_one_source_out_means": {
                        s: st.fmean(v for k, v in means.items() if k != s) for s in means
                    },
                    "is_confidence_interval": False,
                }
            )
    return results


def analyze(records, scope, descriptors):
    validate_records(records, scope)
    lookup = {condition_key(r): r for r in records}
    groups = []
    for family, dataset in itertools.product(scope["families"], scope["datasets"]):
        rows = [r for r in records if condition_key(r)[:2] == (family, dataset)]
        base = lookup[family, dataset, scope["seeds"][0], 0.0]
        metrics = {}
        for metric in METRICS:
            curve = paired_curve(
                lookup, family, dataset, scope["seeds"], scope["rates"][1:], metric
            )
            metrics[metric] = {"curve": curve, **boundaries(curve)}
        primary = metrics[scope["primary_metric"]]
        leave_seed = []
        for omit in scope["seeds"]:
            seeds = [s for s in scope["seeds"] if s != omit]
            curve = paired_curve(
                lookup, family, dataset, seeds, scope["rates"][1:], "normalized_mae"
            )
            leave_seed.append({"omitted_seed": omit, "remaining_seeds": seeds, **boundaries(curve)})
        margins = []
        for margin in scope["diagnostic_relative_margins"]:
            margins.append(
                {
                    "relative_margin": margin,
                    "mean_relative": interval(
                        scope["rates"][1:],
                        [c["mean_relative_improvement"] > margin for c in primary["curve"]],
                    ),
                    "all_seeds_relative": interval(
                        scope["rates"][1:],
                        [c["seed_min_relative_improvement"] > margin for c in primary["curve"]],
                    ),
                }
            )
        channels = []
        for i, name in enumerate(base["adapter_settings"]["channel_names"]):
            curve = paired_curve(
                lookup, family, dataset, scope["seeds"], scope["rates"][1:], "normalized_mae", i
            )
            channels.append({"channel": name, "endpoint": curve[-1], **boundaries(curve)})
        descriptor = next(d for d in descriptors["descriptors"] if d["dataset"] == dataset)
        require(descriptor["dataset_sha256"] == base["identity"]["data_sha256"], "descriptor data")
        groups.append(
            {
                "family": family,
                "dataset": dataset,
                "horizon": scope["horizon"],
                "source_group": source_group(dataset),
                "metrics": metrics,
                "posthoc_leave_one_seed_out": leave_seed,
                "posthoc_relative_margins": margins,
                "channels": channels,
                "budget": budget_summary(rows),
                "train_descriptor": descriptor,
                "test_windows": base["test"]["number_of_test_windows"],
                "test_window_hash": base["test"]["window_hash"],
            }
        )
    return {
        "schema_version": 1,
        "scope": scope,
        "status": "two_model_restricted_scope_analysis_completed_not_full_original_study",
        "conditions": len(records),
        "groups": groups,
        "source_aggregates": source_aggregate(groups, scope),
        "primary_rule": "mean paired raw delta in normalized MAE > 0; all-three-seed companion",
        "additional_analyses": (
            "posthoc descriptive; do not replace primary metric or choose settings"
        ),
        "uncertainty": "three paired seed ranges; not CI or p-values",
        "iid_window_bootstrap_used": False,
        "time_block_bootstrap_performed": False,
        "equal_epoch_gpu_robustness_performed": False,
        "new_gpu_execution": False,
        "test_re_evaluation": False,
        "test_metrics_recomputed_from_predictions": False,
        "timer_results_deleted": False,
        "timer_included": False,
        "generalization": "two pinned models, five dataset variants, three source groups, H96",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/analysis/primary5_h96.yaml"))
    p.add_argument("--archive-dir", type=Path, action="append", default=[])
    p.add_argument("--archive", type=Path, action="append", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    scope, root = load_scope(args.config), Path.cwd()
    require(not args.output_dir.exists(), "use a new analysis output directory")
    archives = args.archive + [p for d in args.archive_dir for p in sorted(d.glob("*.zip"))]
    records, hashes = load_verified(scope, root, archives)
    ref = scope["descriptors"]
    require(digest(root / ref["path"]) == ref["sha256"], "descriptor artifact bytes changed")
    descriptors = read(root / ref["path"])
    require(descriptors["future_values_parsed_for_descriptors"] is False, "descriptor scope")
    require(
        descriptors["descriptor_source_sha256"]
        == digest(root / "src/tsfm_crossover/analysis/train_features.py"),
        "descriptor implementation changed",
    )
    result = analyze(records, scope, descriptors)
    result["evidence"] = {
        "archives": hashes,
        "raw_records_bound_to_pinned_canonical_reviews": len(records),
        "analysis_config_sha256": digest(args.config),
        "analysis_code_sha256": digest(Path(__file__)),
        "supporting_source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("crossover.py", "paper_report.py", "paper_template.html")
        },
        "execution_commit": scope["execution_commit"],
    }
    write_json_atomic(args.output_dir / "analysis.json", result)
    from tsfm_crossover.analysis.paper_report import render

    render(result, args.output_dir)
    write_json_atomic(
        args.output_dir / "verification.json",
        {
            "status": "source_binding_and_analysis_checks_passed",
            "conditions": len(records),
            "source_archives": len(hashes),
            "execution_commit": scope["execution_commit"],
            "analysis_code_sha256": result["evidence"]["analysis_code_sha256"],
            "checks": {
                "canonical_review_bytes": True,
                "original_zip_bytes": True,
                "raw_results_match_canonical_reviews": True,
                "complete_two_model_grid": True,
                "shared_data_windows_masks_scale_and_sampling": True,
                "train_only_descriptor_provenance": True,
            },
            "new_gpu_execution": False,
            "test_metric_recomputation_from_predictions": False,
            "generated_file_sha256": {
                path.name: digest(path)
                for path in sorted(args.output_dir.iterdir())
                if path.is_file()
            },
        },
    )
    print(
        json.dumps(
            {
                "conditions": len(records),
                "model_dataset_groups": len(result["groups"]),
                "new_gpu_execution": False,
                "output": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
