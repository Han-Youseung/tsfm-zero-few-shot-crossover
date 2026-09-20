# 라이선스와 출처

> 확인일: 2026-09-20. `pending`은 추가 확인이 필요하며 라이선스를 추측하지 않았다는 뜻입니다.

| 프로젝트/모델 | 코드 저장소 | 모델 가중치/데이터셋 | 라이선스 | 사용 방식 | 고지 | 상태 |
|---|---|---|---|---|---|---|
| IBM Granite TSFM | https://github.com/ibm-granite/granite-tsfm | — | Apache-2.0 | TTM 공식 API 후보 | 필요 | 확인 |
| Granite TTM R3 | 상동 | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3 | Apache-2.0 | 파일럿 후 revision 고정 | 필요 | 확인 |
| Salesforce uni2ts | https://github.com/SalesforceAIResearch/uni2ts | — | Apache-2.0 | MOIRAI 공식 API 후보 | 필요 | 확인 |
| MOIRAI 2.0 R-small | 상동 | https://huggingface.co/Salesforce/moirai-2.0-R-small | CC BY-NC 4.0 | 비상업 연구, 파일럿 후 revision 고정 | 저작자 표시·비상업 조건 | 확인 |
| TSFM-Bench | https://github.com/decisionintelligence/TSFM-Bench | 사용하지 않음 | CC BY-NC-ND 4.0 | 관련 연구로만 인용 | 인용 | 확인 |
| ETT 4종 공식 원본 | https://github.com/zhouhaoyi/ETDataset | 공식 wide CSV | CC BY-ND 4.0 | 독립 전처리 후보 | 표시·변경/재배포 제한 확인 | ready_with_warnings |
| 나머지 10개 데이터셋 | 개별 공식 출처 확인 중 | 데이터셋 파일 | pending | 출처·checksum 고정 전 사용 차단 | pending | pending |

본 저장소에 TSFM-Bench 코드를 포함하거나 수정해 배포하지 않으며, 해당 프로젝트의 구조를 번안하지 않습니다. 모델 코드와 가중치의 라이선스는 따로 관리하며 revision 고정 시 다시 확인합니다.
# Dataset licenses and sources

The registry supports official name, source URL, author/provider, download URL, license or terms, redistribution status, verification status/date, SHA256, and local filename. All 14 entries currently remain **pending** except for their study labels and local filenames. This is intentional: benchmark hosting is not assumed to be an authoritative source or to grant redistribution rights.

Before real data are used, each entry must be checked against an original provider or authoritative publication, dated, and pinned by local SHA256. Until then, data must not be redistributed or committed.

## Verified source leads checked 2026-09-20

- ETT: the [official ETDataset repository](https://github.com/zhouhaoyi/ETDataset) documents the original wide columns and cites Informer. Its repository license is CC BY-ND 4.0. Four official files were pinned at commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07`; checksums are recorded in the provenance manifests. The bundle is an exactly affine, legacy-prefix-standardized derivative and remains blocked, while official raw is `ready_with_warnings`.
- Electricity: [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112) reports 370 clients, 15-minute measurements, DOI `10.24432/C58C86`, and CC BY 4.0. The local file has 321 hourly channels, hence is a derived bundle whose transformation terms are not established.
- PEMS08/Traffic: [Caltrans PeMS](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source) is the original sensor system and requires an account. The exact derivation and redistribution permission for these local matrices remain pending.
- Weather: the [Max Planck Institute for Biogeochemistry weather portal](https://www.bgc-jena.mpg.de/wetter/weather_data.html) publishes station files under CC BY 4.0. Exact correspondence of the local 21-channel transformed file remains pending.
- AQShunyi: [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality) attributes air-quality observations to the Beijing Municipal Environmental Monitoring Center and meteorology to the China Meteorological Administration. Local preprocessing and redistribution terms remain pending.
- Solar: [Monash/Zenodo record 4656144](https://zenodo.org/records/4656144) describes 137 Alabama series at ten-minute intervals and traces them to NREL/Lai et al.; the displayed record does not provide a clear license value, so permission remains pending.
- Wind, Exchange, ZafNoo, and CzeLan: original-source identity and redistribution permission remain pending. Public availability is not treated as permission.

`original_source`, the route used to download a derivative, and the user-supplied local bundle are separate concepts. Using a TSFM benchmark bundle as a transport source does not adopt that benchmark's experimental protocol.
