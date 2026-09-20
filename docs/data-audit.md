# Real-data audit

## Placement and commands

Put the user-supplied archive at `data/incoming/TSFM_dataset.zip`. Inspect it before extraction:

```bash
python -m tsfm_crossover.data.archive data/incoming/TSFM_dataset.zip
```

Only a safe archive may be extracted beneath `data/raw/`. The audit command is:

```bash
python -m tsfm_crossover.data.audit \
  --registry configs/datasets/registry.yaml \
  --data-root data/raw \
  --context-lengths 256 512 \
  --horizons 96 192 336 720 \
  --sampling-rates 0 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1 \
  --sampling-seed 42 \
  --output results/manifests/datasets
```

Use `--dry-run` to avoid writes, `--dataset NAME` to select a subset, `--overwrite` only for an intentional replacement, and `--fail-on-warning` in strict automation. Identical input signatures are skipped. Raw data remain ignored by Git because source licenses and files can be large; only small metadata summaries are committed.

## September 2026 audit outcome

The archive contained 24 entries (22 files), expanded to 1,971,892,973 bytes, and passed traversal, absolute-path, symlink, encryption, duplicate-path, executable, and CRC checks. Its SHA256 is `8e7fca31165755daa838ef14a79290f6195503a425fe08c89dc2ce08dcb4ec22`.

All 14 study files were uniquely identified by exact canonical filename. They use a processed long layout (`date`, `data`, `cols`; AQShunyi also has `name`) rather than the original wide formats. The values appear transformed; the bundle does not document whether this transformation was fitted using train-only observations. Consequently every dataset is currently `blocked` for leakage-safe model experiments even though all context/horizon window combinations are structurally feasible.

Weather has 21 duplicate timestamps/rows and 21 irregular intervals per channel alignment. The other 13 files had no parsed NaN, infinity, duplicate timestamp, non-monotonic timestamp, or irregular interval. The audit never sorts, imputes, resamples, deduplicates, or rewrites source files.

`ready` means all technical, provenance, and preprocessing checks passed; `ready_with_warnings` is technically usable with non-blocking provenance warnings; `blocked` has a correctness or leakage blocker; `missing` has no exact file; and `ambiguous` has multiple exact candidates.

Observed metadata describes only these local bytes. Expected metadata is the configured loading contract. Official metadata must be supported by an original provider; observing a row count locally does not make it official.
