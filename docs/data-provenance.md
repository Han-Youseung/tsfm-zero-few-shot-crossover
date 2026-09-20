# Dataset provenance resolution

Checked 2026-09-20. This work investigates the origin of the user-supplied bundle; it does
not reproduce or adopt TSFM-Bench's protocol.

## Bundle route and published evidence

The bundle layout and `FORECAST_META.csv` correspond to the datasets distributed from
[TSFM-Bench](https://github.com/decisionintelligence/TSFM-Bench). The inspected repository
revision is `f9bb69402fe5c57ff6dec7ac254aabaab38239ff`.

- `README.md`, section `Prepare Datasets`, links a Google Drive archive described as
  “well pre-processed.” It does not state how the files were created.
- `ts_benchmark/data/utils.py`, function `read_data`, reads `date,data,cols`, takes channel
  order from first appearance, and slices consecutive channel blocks back into a wide
  frame. It performs no scaling, imputation, interpolation, or resampling at that point.
- `ts_benchmark/data/data_source.py`, `LocalDataSource._load_series`, delegates to that
  reader without explaining upstream preparation.
- Repository history and tracked files did not expose a conversion script that creates
  the distributed forecasting CSVs. Consequently there is no code-to-file fingerprint
  chain and no proof that upstream conversion was reshape-only.

The loader establishes the expected storage convention, not the provenance of the values.
Model-level normalization code elsewhere in the benchmark is not evidence for how these
CSV files were prepared.

## Official ETT comparison

The four official wide CSVs were downloaded from the
[ETDataset repository](https://github.com/zhouhaoyi/ETDataset) at commit
`1d16c8f4f943005d613b5bc962e9eeb06058cf07`. Downloads are held only under ignored
`data/official_raw/`. Their SHA256 values and aggregate comparison results are in
`raw_comparison_summary.csv`.

For ETTh1, ETTh2, ETTm1, and ETTm2, timestamps, shape, channel names, and channel order
match the bundle exactly. Values are not exact or allclose to raw. Per channel, however,
the bundle is an affine transform of official raw with maximum residual below `3e-15`.
The slope and intercept exactly match population-standardization statistics fitted to the
official first 12 months: 8,640 rows for hourly ETT and 34,560 rows for 15-minute ETT.
Thus these bundle variants are reversible but are not raw or reshape-only. They remain
grade D and blocked; the official variants are grade A and `ready_with_warnings` because
the repository declares CC BY-ND 4.0 and raw files must not be redistributed here.

This legacy prefix lies within the first 60% of each ETT series, so the comparison did not
find future-test leakage for ETT. It still violates this study's rule that preprocessing be
performed independently and consistently from the declared 60% train split.

## What remains unknown

For Electricity, Traffic, PEMS08, Solar, Wind, Weather, AQShunyi, Exchange, ZafNoo, and
CzeLan, upstream normalization, imputation, interpolation, resampling, timestamp changes,
channel filtering, and unit changes remain unknown. Diagnostic moments are not used to
turn an unknown into a fact. These bundle variants remain grade D and blocked until an
official raw source or fingerprint-linked creation code is available.

The full evidence tables are under `results/manifests/provenance_resolution/`. They contain
only hashes, counts, decisions, and aggregate statistics—never complete time-series values.
