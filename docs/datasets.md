# Dataset registry

The canonical registry is `configs/datasets/registry.yaml`. It contains all 14 study names and exact aliases; `CzenLan` resolves to `CzeLan`. Fuzzy matching is deliberately disabled.

No dataset is downloaded by this project. Place a locally obtained CSV at the registry path after independently checking its provenance and license. Unknown row counts, channel counts, frequency, timestamp fields, source fields, and licenses remain `null` or `pending`; they are not inferred from benchmark conventions. The current timestamp declarations for the four ETT files are provisional and must be confirmed before a real-data run.

The audited local bundle contains all 14 datasets in long format. Every file has a `date` column, so the loading contract was updated from unknown to required `date`; this is an observed file rule, not an assertion about an original provider's format. The bundle has already transformed values and its preprocessing fit range is unknown, so `local_preprocessing_status` remains `unverified`.

Validation reports discrepancies and never changes the registry or data. Only explicitly selected targets, or columns admitted by `all_numeric_targets` after exclusions, become model inputs. Timestamp-free files use a positional index; no synthetic dates are created.
