# 라이선스와 출처

> 확인일: 2026-09-20. `pending`은 추가 확인이 필요하며 라이선스를 추측하지 않았다는 뜻입니다.

| 프로젝트/모델 | 코드 저장소 | 모델 가중치/데이터셋 | 라이선스 | 사용 방식 | 고지 | 상태 |
|---|---|---|---|---|---|---|
| IBM Granite TSFM | https://github.com/ibm-granite/granite-tsfm | — | Apache-2.0 | TTM 공식 API 후보 | 필요 | 확인 |
| Granite TTM R3 | 상동 | https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3 | Apache-2.0 | 파일럿 후 revision 고정 | 필요 | 확인 |
| Salesforce uni2ts | https://github.com/SalesforceAIResearch/uni2ts | — | Apache-2.0 | MOIRAI 공식 API 후보 | 필요 | 확인 |
| MOIRAI 2.0 R-small | 상동 | https://huggingface.co/Salesforce/moirai-2.0-R-small | CC BY-NC 4.0 | 비상업 연구, 파일럿 후 revision 고정 | 저작자 표시·비상업 조건 | 확인 |
| TSFM-Bench | https://github.com/decisionintelligence/TSFM-Bench | 사용하지 않음 | CC BY-NC-ND 4.0 | 관련 연구로만 인용 | 인용 | 확인 |
| 14개 실험 데이터셋 | 개별 공식 출처 예정 | 데이터셋 파일 | pending | 다음 단계에서 출처·checksum 고정 | pending | pending |

본 저장소에 TSFM-Bench 코드를 포함하거나 수정해 배포하지 않으며, 해당 프로젝트의 구조를 번안하지 않습니다. 모델 코드와 가중치의 라이선스는 따로 관리하며 revision 고정 시 다시 확인합니다.
# Dataset licenses and sources

The registry supports official name, source URL, author/provider, download URL, license or terms, redistribution status, verification status/date, SHA256, and local filename. All 14 entries currently remain **pending** except for their study labels and local filenames. This is intentional: benchmark hosting is not assumed to be an authoritative source or to grant redistribution rights.

Before real data are used, each entry must be checked against an original provider or authoritative publication, dated, and pinned by local SHA256. Until then, data must not be redistributed or committed.
