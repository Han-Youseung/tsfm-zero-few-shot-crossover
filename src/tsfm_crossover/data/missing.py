"""Causal context preparation; original targets are never imputed."""

import math

POLICY = "causal_ffill_complete_training_targets_v1"


def causal_context(values):
    if not values:
        raise ValueError("empty observations")
    last = [float("nan")] * len(values[0])
    result = []
    for row in values:
        if len(row) != len(last):
            raise ValueError("inconsistent channel count")
        for channel, value in enumerate(row):
            if math.isinf(value):
                raise ValueError("infinite observation")
            if math.isfinite(value):
                last[channel] = value
        result.append(last.copy())
    return result


def eligible_windows(windows, original, context, *, training):
    """Common model-independent eligibility, before nested sampling."""

    def prefix(rows, bad):
        result = [0]
        for row in rows:
            result.append(result[-1] + int(bad(row)))
        return result

    bad_context = prefix(context, lambda r: not all(map(math.isfinite, r)))
    bad_target = prefix(original, lambda r: not all(map(math.isfinite, r)))
    valid_target = [0]
    for row in original:
        valid_target.append(valid_target[-1] + sum(map(math.isfinite, row)))
    return [
        w
        for w in windows
        if bad_context[w.context_end] == bad_context[w.context_start]
        and (
            bad_target[w.target_end] == bad_target[w.target_start]
            if training
            else valid_target[w.target_end] > valid_target[w.target_start]
        )
    ]
