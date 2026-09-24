# Phase 6: fourteen-dataset readiness

Checked 2026-09-25. The legacy registry remains untouched and every `__bundle_long`
variant stays blocked. The executable overlay is `results/manifests/pilot/prepared_data.json`.
`dataset_readiness.json/csv` has all fourteen rows, exact SHA256, shape, frequency,
duplicate/order/gap counts, missing values, train-only constant counts and source links.
Null means unverified, never zero. Previous audits remain immutable.

| Dataset | Acquired shape (rows x targets) | Status / next action |
|---|---:|---|
| ETTh1 | 17,420 x 7 | Official raw ready with CC BY-ND warning; no redistribution |
| ETTh2 | 17,420 x 7 | Official raw ready with CC BY-ND warning |
| ETTm1 | 69,680 x 7 | Official raw ready with CC BY-ND warning |
| ETTm2 | 69,680 x 7 | Official raw ready with CC BY-ND warning |
| Electricity | 26,304 x **370** | Independent UCI hourly variant ready; not the 321-channel bundle |
| Traffic | 17,544 x 862 | Author matrix acquired; timestamps/preparation/terms unresolved |
| PEMS08 | 17,856 x 170 x 3 measures | Author NPZ acquired; flow selection/calendar/terms need verification |
| Solar | 52,560 x 137 | Author matrix acquired; calendar and complete preparation/terms need verification |
| Wind | 48,673 x 7 (legacy only) | Original provider/variant unidentified; do not replace with arbitrary wind data |
| AQShunyi | 35,064 x 11 | UCI raw acquired; **8,040 missing values**, imputation policy pending |
| Exchange | 7,588 x 8 | Author matrix acquired; dated observations, original provider/terms unresolved |
| Weather | 52,696 x 21 | MPI raw acquired; **1 duplicate timestamp, 2 irregular intervals**, no automatic deletion |
| ZafNoo | 19,225 x 11 (legacy only) | SAPFLUXNET lead; site/variables/units and exact span unresolved |
| CzeLan | 19,934 x 11 (legacy only) | SAPFLUXNET lead; site/variables/units and exact span unresolved |

## Verified sources and independent conversion

- ETT: [official repository](https://github.com/zhouhaoyi/ETDataset/tree/1d16c8f4f943005d613b5bc962e9eeb06058cf07),
  exact hashes from earlier provenance work. Identity copy, no normalization or trimming.
- Electricity: [UCI DOI 10.24432/C58C86](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014),
  Artur Trindade, CC BY 4.0. All 370 clients are retained. Select quarter-hour timestamps
  2012-01-01 00:15 through 2015-01-01 00:00; each hour-ending output averages exactly
  four preceding kW observations. No interpolation, zero replacement or channel selection.
  The provider describes zero values for not-yet-created clients and daylight-saving
  conventions. Those values are preserved, not reinterpreted as measured demand.
  There are 22 train-constant channels; raw metrics retain them, normalized metrics
  identify/exclude them. This is an explicitly named alternative variant, not proof of
  the old 321-channel derivation. Final-paper variant acceptance remains a research decision.
- AQShunyi: [UCI DOI 10.24432/C5RK5G](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data),
  Song Chen, CC BY 4.0. Select Shunyi and the eleven numeric pollutant/meteorological
  measures, excluding categorical wind direction and metadata. Preserve all NA values.
- Weather: [MPI station portal](https://www.bgc-jena.mpg.de/wetter/weather_data.html),
  CC BY 4.0; rooftop `mpi_roof_2020a.zip` and `mpi_roof_2020b.zip`. Concatenate in
  published half-year order, decode timestamps/column encoding, retain every measurement.
  The duplicate is present in official data, not merely an artifact of the supplied bundle.
  Native Windows TLS validation succeeded; Python's local certificate chain failed for
  this host. No certificate verification was disabled. Source ZIP hashes are recorded.
- Solar/Traffic/Exchange: [LSTNet authors' distribution](https://github.com/laiguokun/multivariate-time-series-data/tree/7f402f185cc2435b5e66aed13a3b560ed142e023).
  This is source investigation only, not reuse of any benchmark protocol or code.
  The matrices do not include timestamp columns. Public hosting alone does not settle
  original data-use terms or the complete preprocessing chain.
- PEMS08: [ASTGCN authors' distribution](https://github.com/Davidham3/ASTGCN/tree/d5d40645fa45471f2b473fc49a29ec3f0a993c6d/data/PEMS08).
  Their README describes 170 detectors, July-August 2016 and flow/occupancy/speed.
  We downloaded the NPZ with `allow_pickle=False`; no arbitrary channel projection
  or invented timestamp was added to the pilot. Caltrans account/terms may need user action.
- ZafNoo/CzeLan: [SAPFLUXNET 0.1.5](https://zenodo.org/records/3971689) is an open-access
  harmonized source lead, not yet fingerprint-linked to these local variants. Its whole
  archive is 3.2 GB with multiple unit representations and quality flags. Bulk download
  was deferred until site/variable/units mapping is established; names alone are insufficient.

## Reproduction and release policy

`python -m tsfm_crossover.data.pilot_data` prepares the four ETTs, Electricity and
AQShunyi into ignored `data/`. `--names Weather --output NEW_MANIFEST.json` prepares
the station variant after its ZIPs are available. `scripts/prepare_source_leads.py`
acquires the four author matrices for provenance work only. Every source and output
fingerprint is retained. `--verify-against` checks the published fingerprints; a changed
download fails rather than quietly replacing evidence. pandas/NumPy are preparation-only
dependencies already present in the separate model environments, not base test requirements.

Derived CSVs use LF newlines and ten significant decimal digits (`%.10g`); ETT copies
retain their exact original bytes. The independently prepared Electricity output was
reproduced with both local model environments (pandas 3.x and 2.1.4) and matched the
same pinned SHA256. This numeric serialization is part of the conversion, not a scaler.

Full-file hashes and integrity counts inspect raw bytes/format; they are not test metrics.
All learned/normalization statistics use train only. Pilot loading stops at validation.end
and never parses test target values. Original raw files, intermediate CSVs and ZIPs are
never committed. None of the fourteen intended study datasets has been removed.

Ready data can proceed now. For blocked data the next actions are specific: authorize a
documented causal missing-value policy for AQShunyi; resolve Weather duplicates/gaps;
verify the four author-distributed variants' calendar/preparation/terms; identify Wind;
map SAPFLUXNET sites/variables and units. No future interpolation or backward fill is enabled.
