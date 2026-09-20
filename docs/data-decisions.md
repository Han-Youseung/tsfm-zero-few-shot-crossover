# Data decisions after provenance resolution

Checked 2026-09-20. A/B may be used subject to license warnings; C is blocked pending a
research decision; D is blocked. No C or D dataset was released merely because it parses.

| Dataset variant | Grade | Status | Decision |
|---|---:|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2 `__bundle_long` | D | blocked | Replace with official raw; legacy 12-month prefix standardization is verified. |
| ETTh1/ETTh2/ETTm1/ETTm2 `__official_raw` | A | ready_with_warnings | Eligible for independent preparation; observe CC BY-ND 4.0 and do not redistribute. |
| Electricity `__bundle_long` | D | blocked | Obtain official raw and independently define the documented 321-channel hourly derivation. |
| Traffic `__bundle_long` | D | blocked | Obtain authoritative source and derivation; account/terms may require user action. |
| PEMS08 `__bundle_long` | D | blocked | Obtain authoritative source and derivation; account/terms may require user action. |
| Solar `__bundle_long` | D | blocked | Resolve source license and independently reproduce the 137-channel series. |
| Wind `__bundle_long` | D | blocked | Identify and verify the original provider and preparation. |
| Weather `__bundle_long` | D | blocked | Compare official station raw before deciding whether the 21 identical duplicates may be removed. |
| AQShunyi `__bundle_long` | D | blocked | Compare UCI official raw and document any imputation of original missing observations. |
| Exchange `__bundle_long` | D | blocked | Identify authoritative raw source and verify the affine-looking transformed values. |
| ZafNoo `__bundle_long` | D | blocked | Verify the site/provider, license, units, and preparation. |
| CzeLan `__bundle_long` | D | blocked | Verify the site/provider, license, units, and preparation. |

Thirteen bundle files are structurally lossless long-to-wide candidates. Weather is not:
timestamp `2020/5/12 6:00` is duplicated once in every one of 21 channels, yielding 21 identical
composite-key/value duplicates and no conflicting duplicates. Repeated timestamps across
different channels are normal and are not counted as composite-key duplicates. No row is
automatically deleted, selected, or averaged.

User decisions still required before expanding the study are whether to acquire and convert
the ten remaining official sources, whether any license-pending source should remain in
scope, and whether the study should proceed initially with the four verified ETT variants
or wait for a larger verified subset.
