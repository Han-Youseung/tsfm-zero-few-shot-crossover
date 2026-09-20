# Dataset registry

The canonical registry is `configs/datasets/registry.yaml`. It contains all 14 study names and exact aliases; `CzenLan` resolves to `CzeLan`. Fuzzy matching is deliberately disabled.

No dataset is downloaded by this project. Place a locally obtained CSV at the registry path after independently checking its provenance and license. Unknown row counts, channel counts, frequency, timestamp fields, source fields, and licenses remain `null` or `pending`; they are not inferred from benchmark conventions. The current timestamp declarations for the four ETT files are provisional and must be confirmed before a real-data run.

Validation reports discrepancies and never changes the registry or data. Only explicitly selected targets, or columns admitted by `all_numeric_targets` after exclusions, become model inputs. Timestamp-free files use a positional index; no synthetic dates are created.
