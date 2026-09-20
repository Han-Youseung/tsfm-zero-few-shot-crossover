# Result and manifest schema

Phase 1 defines atomic experiment result records. Phase 2 adds split, scaler, window, sampling, and coverage records. A sampling manifest contains dataset SHA256, split hash, context, horizon, seed, strata, requested/effective rates, selected window IDs, nestedness, code commit, timestamp, and a stable content hash.

`generated_at` is audit metadata and is excluded from the stable manifest hash, so identical scientific inputs hash identically across runs. Large real-data ID lists may later be separated into JSONL/CSV with a referenced checksum; this phase creates no real manifests.
