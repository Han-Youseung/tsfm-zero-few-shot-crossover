# Common data protocol

All models share one model-independent pipeline. For `T` ordered rows, boundaries are `floor(0.6T)` and `floor(0.8T)`: train `[0, train_end)`, validation `[train_end, validation_end)`, and test `[validation_end, T)`. The intervals have no gaps or overlaps.

The channel-wise scaler uses population standard deviation (`ddof=0`) and is fit only to raw train points. A zero-variance channel receives scale 1 and is recorded. Validation and test are transform-only. Missing values, infinity, duplicate/non-monotonic timestamps, and parsing failures are reported; no sorting, deletion, interpolation, aggregation, or imputation occurs.

Train windows keep both context and target inside train. Validation/test rolling-origin windows keep the complete target inside their split, while context may use only observations strictly before that target. Thus later origins may use actual values already observed earlier in validation/test, but never the current target or future values. This differs from fixed-origin evaluation. Default stride is 1.

The split, scaler fit range, dataset fingerprint, and window positions are recorded. These boundaries and window IDs are common to TTM and MOIRAI.

Real bundle audits use a streaming long-CSV reader to avoid materializing very large matrices. `expected_*` belongs to configuration, `observed_*` is computed from a checksum-pinned local file, and `official_*` stays null until supported by an authoritative source. A pre-normalized bundle cannot pass the leakage gate unless the preprocessing fit range is documented as train-only.
