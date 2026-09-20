# Nested temporally stratified sampling

A single master permutation is generated per dataset–context–horizon–seed. The ordered train windows are divided into contiguous temporal strata (default `ceil(sqrt(N))`, capped at `N`). Each stratum is independently shuffled with deterministic seed-derived state. A seeded bit-reversal order spreads early selections across the timeline, and round-robin passes drain every stratum. Each requested set is a prefix of that one permutation, so nestedness is structural rather than repaired afterward.

For positive rate `r`, `k=max(1,floor(rN))`; zero selects none and 100% selects each window once. Requested and effective (`k/N`) rates can differ, especially for small `N` or when rates share the same `k`.

Construction costs `O(N + S log S)` space/time aside from output ordering (`S` strata; sorting is not over windows). Coverage uses an event sweep with `O(N log N)` time and `O(N)` events, not an `N × time` matrix. The method improves temporal spread but is not a random uniform subset and does not guarantee optimal discrepancy for every irregular window distribution. Raw-time-point coverage is reported because heavily overlapping windows can expose far more observations than their window percentage suggests.

Sampling manifests are model-independent: TTM and MOIRAI reuse the same dataset fingerprint, split hash, window IDs, and prefixes.
