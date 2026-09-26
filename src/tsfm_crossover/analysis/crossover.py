"""Paired-seed descriptive intervals; no IID rolling-window bootstrap or p-values."""

from tsfm_crossover.evaluation.metrics import macro, relative_improvement


def source_group(dataset):
    return {
        "ETTh1": "ETT-transformer1",
        "ETTm1": "ETT-transformer1",
        "ETTh2": "ETT-transformer2",
        "ETTm2": "ETT-transformer2",
    }.get(dataset, dataset)


def interval(rates, improved):
    first = next((i for i, value in enumerate(improved) if value), None)
    sustained = next((i for i in range(len(rates)) if all(improved[i:])), None)

    def bounds(i):
        return (
            None
            if i is None
            else {"lower_exclusive": 0.0 if i == 0 else rates[i - 1], "upper_inclusive": rates[i]}
        )

    return {
        "first_improvement": bounds(first),
        "sustained_improvement": bounds(sustained),
        "nonmonotone_sign": any(a and not b for a, b in zip(improved, improved[1:], strict=False)),
        "status": "not_observed_on_grid" if first is None else "observed_on_grid",
    }


def summarize(records, rates, seeds):
    groups = {}
    for record in records:
        row = record["identity"]["condition"]
        key = (row["family"], row["dataset"], row["horizon"])
        condition = (row["seed"], row["rate"])
        if condition in groups.setdefault(key, {}):
            raise ValueError("duplicate analysis condition")
        groups[key][condition] = record
    output = []
    for (family, dataset, horizon), runs in sorted(groups.items()):
        expected = {(s, r) for s in seeds for r in (0.0, *rates)}
        item = {
            "family": family,
            "dataset": dataset,
            "horizon": horizon,
            "source_group": source_group(dataset),
        }
        if set(runs) != expected:
            output.append(
                item | {"status": "incomplete", "available": len(runs), "required": len(expected)}
            )
            continue
        # Same exact rolling origins/mask semantics across every rate/seed.
        if len({r["test"]["window_hash"] for r in runs.values()}) != 1:
            raise ValueError("unpaired test windows")
        if (
            len(
                {
                    tuple(c["count"] for c in r["test"]["metrics"]["per_channel"])
                    for r in runs.values()
                }
            )
            != 1
        ):
            raise ValueError("unpaired target masks")
        curve = []
        for rate in rates:
            pairs = []
            for seed in seeds:
                zero = runs[seed, 0.0]["test"]["metrics"]["macro"]["normalized_mae"]
                few = runs[seed, rate]["test"]["metrics"]["macro"]["normalized_mae"]
                if zero is None or few is None:
                    raise ValueError("undefined primary metric")
                pairs.append(
                    {
                        "seed": seed,
                        "zero_minus_few": zero - few,
                        "relative_improvement": relative_improvement(zero, few),
                    }
                )
            deltas = [p["zero_minus_few"] for p in pairs]
            curve.append(
                {
                    "rate": rate,
                    "paired": pairs,
                    "mean_delta": macro(deltas),
                    "seed_min": min(deltas),
                    "seed_max": max(deltas),
                    "mean_relative_improvement": macro([p["relative_improvement"] for p in pairs]),
                }
            )
        output.append(
            item
            | {
                "status": "complete",
                "curve": curve,
                "mean": interval(rates, [p["mean_delta"] > 0 for p in curve]),
                "all_three_seeds": interval(rates, [p["seed_min"] > 0 for p in curve]),
                "uncertainty": "three paired seed range; descriptive, not a CI",
            }
        )
    # No aggregate across an incomplete data/horizon group. Give every source group
    # equal weight, averaging ETT hourly/minutely within its transformer first.
    aggregates = []
    for family in sorted({r["family"] for r in output}):
        complete = [r for r in output if r["family"] == family and r["status"] == "complete"]
        if len(complete) != 36:
            continue
        for i, rate in enumerate(rates):
            source_values = {}
            for r in complete:
                source_values.setdefault(r["source_group"], []).append(
                    r["curve"][i]["mean_relative_improvement"]
                )
            aggregates.append(
                {
                    "family": family,
                    "rate": rate,
                    "source_macro_relative_improvement": macro(
                        [macro(v) for v in source_values.values()]
                    ),
                    "source_groups": len(source_values),
                }
            )
    return {
        "groups": output,
        "source_group_aggregates": aggregates,
        "inference_limit": "seed variation only; no population CI or causal feature claims",
    }
