# Canonical time-series format

The independent canonical representation contains ordered timestamps, ordered channel
names, a rectangular time-by-channel numeric matrix, a missing-value mask, source SHA256,
canonicalization-config hash, data fingerprint, and validation report.

Canonicalization is aggregation-free:

1. `(timestamp, channel)` must be unique. Identical and conflicting duplicates both fail.
2. Every channel must contain the same timestamp set.
3. Every timestamp must contain every expected channel.
4. Timestamps are sorted ascending. Channel order is chosen from official order, then
   verified bundle metadata order, then first appearance. Alphabetic order is not a default.
5. Input and output element counts and the float-value multiset must match.

The in-memory implementation is `canonicalize_long_records`. It never edits its source and
does not write the full canonical matrix. Real-bundle inspection uses the channel-major
storage convention only after verifying that channel blocks are contiguous and aligned.

## Fingerprint serialization

`tsfm-canonical-v1` uses SHA256, UTF-8 strings prefixed by unsigned 64-bit big-endian byte
lengths, missing flags as one byte, and non-missing values as IEEE-754 float64 big-endian.
It hashes an ordered timestamp component and one value/mask component per ordered channel,
then hashes their binary digests with the dimensions and channel names. This permits
streaming without building a large string. Row order in a long input does not matter after
canonicalization; timestamp order, channel order, missingness, or any value change does.

`source_variant` is mandatory in experiment configuration. A comparison group must pass
`validate_single_source_variant`; mixing names such as `ETTh1__bundle_long` and
`ETTh1__official_raw` is rejected.
