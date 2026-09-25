# 본 실험 전 준비: 승인된 데이터 권고안 적용

## 상태와 경계

본 실험은 실행하지 않았다. 원본, 과거 CPU/GPU 결과 및 실패 기록은 보존한다.
현재 `prepared_primary.json`의 사용 준비 후보는 **9개**이다. 이는 최종 연구 세트
확정이나 9개 전부의 GPU 검증 완료를 의미하지 않는다.

| 후보 | 현재 처리 | 남은 조건 |
|---|---|---|
| ETTh1, ETTh2, ETTm1, ETTm2, Electricity | 기존 준비 및 A100 pilot 증거 유지 | 최종 공통 budget/protocol 결정 |
| Solar, Traffic | 공개 Monash 배포본과 author matrix 대조 완료 | 신규 A100 feasibility |
| Weather | MPI 2020 원본에서 별도 quality variant 생성 | 신규 A100 feasibility |
| Tetouan | UCI 849 대체 후보, 8채널 전체 유지 | 신규 A100 feasibility; 변수 단위 문서 정밀화 |
| PEMS08 | conditional, primary 실행 차단 유지 | 배포 이용조건, calendar 및 upstream 처리 확인 |
| Exchange, ZafNoo, CzeLan | conditional, primary 실행 차단 유지 | 실제 원본·시간축·처리 과정 대응 확인 |
| Wind | primary 제외 | pred_* 생성 시점 및 원 관측 변수 출처 미확인 |
| AQShunyi | 현재 네-horizon primary 제외 | H336/H720 완전 target train window가 0개 |

10개 이상은 목표이지 품질 기준을 낮출 이유가 아니다. PEMS08의 조건이 해결되면
10번째 후보로 검증할 수 있다. 해결되지 않으면 9개로 가능한 연구 범위를 먼저 평가한다.
ETT 4종은 독립적인 4개 도메인이 아니므로 데이터셋 개수와 원 출처/도메인 수를
구분하여 보고한다. Solar는 원시 실측 센서가 아닌 시뮬레이션 기반 연구 배포본이며,
Solar/Traffic의 timestamp는 배포본의 calendar와 1초 offset을 그대로 보존한다.

## 중복·결측 정책

1. 모든 컬럼이 같은 행만 별도 variant에서 제거한다. 동일 timestamp에 다른 값이
   남으면 준비를 차단하고 conflict를 기록한다. 기존 raw 파일은 수정하지 않는다.
2. 명시적 주기로 누락 timestamp를 NaN 행으로 복원한다. target 값은 채우지 않는다.
3. 입력 context만 시점별 과거 관측값으로 forward-fill한다. leading missing은 그대로
   남으며 해당 context는 제외한다. backfill, interpolation, 미래 통계는 사용하지 않는다.
4. 고정된 TTM R3 공식 loss 경로는 모든 경우에 future mask를 반영하지 않으므로,
   **두 모델 모두 완전 관측 target을 갖는 train window만 사용한다.** MOIRAI만 별도의
   masked loss 표본을 추가하지 않는다. 공식 objective와 vendor 코드는 변경하지 않는다.
5. eligibility 필터 이후의 후보 수를 N으로 삼아 기존 nested sampling을 적용한다.
   원 후보 수, 적격 후보 수, 실제 선택 수, 원 window ID/manifest를 함께 보존한다.
6. validation은 finite target이 하나 이상 있고 사용 가능한 context를 갖는 동일 window를
   사용한다. 예측 API는 future target을 보지 않으며 공통 StreamingMetrics가 원 target의
   NaN 위치를 제외한다. 공식 loss 진단은 완전 target validation window가 있을 때만 한다.
7. 모델 내부 scaling만 사용한다. 외부 StandardScaler는 사용하지 않으며 normalized metric의
   통계만 원 train의 관측값에서 계산한다. pilot loader는 validation 끝까지만 값을 읽는다.

### Weather 변환 기록

MPI 2020: 52,696행 → 완전 중복 1행 제거 → 52,695행 → 누락 timestamp 9개 복원 →
52,704행, 21채널 전체 유지. 시간 충돌 0개, leading missing 0개.
물리적으로 유효하지 않은 -9999를 다음 세 컬럼에 한정하여 NaN으로 표시했다:
`wv (m/s)` 1개, `max. PAR (µmol/m²/s)` 30개, `CO2 (ppm)` 50개.
이는 해당 값에 대한 명시적 물리적 유효성 규칙이며 제공자의 missing-code 정의를
확인했다고 주장하지 않는다. grid의 결측까지 합쳐 270개 target cell이 결측이다.

| Horizon | 원 train 후보 | 완전 target 적격 후보 | pilot 5% 선택 | validation subset |
|---|---:|---:|---:|---:|
| 96 | 31,015 | 29,412 | 1,470 | 64 |
| 192 | 30,919 | 27,935 | 1,396 | 64 |
| 336 | 30,775 | 25,970 | 1,298 | 64 |
| 720 | 30,391 | 21,427 | 1,071 | 64 |

### Tetouan 변환 기록

[UCI 849](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city),
CC-BY-4.0. 고정 ZIP SHA-256:
`3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe`.
실제 CSV는 **52,416행**이며 카탈로그의 52,417과 다르다. 2017-01-01 00:00부터
2017-12-30 23:50까지 10분 간격이다. 날짜나 행을 임의 추가하지 않았다.
5개 기상/환경 변수와 3개 전력 변수, 총 8채널 모두 사용한다.
결측·중복·timestamp conflict·간격 누락은 0개이다.
H96/192/336/720 적격 train 후보는 각각 30,842/30,746/30,602/30,218개이다.

두 신규 variant는 pandas 3.x/2.1.4 환경에서 재생성하여 동일 파일/manifest hash를
확인했다. 원본과 가공 CSV는 Git에 올리지 않는다.

## 다음 실행: 신규 데이터 feasibility만

`notebooks/31_new_dataset_feasibility.ipynb`를 **모델마다 새 A100 세션**에서 실행한다.
Python 3.11 또는 3.12를 사용하고 공개된 이번 준비 commit SHA를 입력한다.
실제 Colab 설치와 신규 GPU 결과는 아직 `installation_pending` / `pending_gpu`이다.

셀 순서:

1. GPU/runtime 확인, `ttm` 또는 `moirai1`, 실행 commit 입력.
2. 고정 commit checkout (실행 도중 pull 금지).
3. 사용자가 Drive 연결 승인: 원본 캐시와 조건별 결과를 영속 저장.
4. 모델별 환경 설치, pip check, subprocess interpreter/CUDA 확인.
5. pinned 공개 배포본 다운로드, 준비 결과 fingerprint 대조.
6. 정확히 16개 조건 확인: Solar/Traffic/Weather/Tetouan × 네 horizon.
7. `RUN_NEW` 입력. Traffic H720부터 실행하여 가장 큰 조건의 제약을 조기에 확인.
8. JSON/CSV 요약 archive 다운로드. 실패도 삭제하지 않는다.

기존 52개 A100 pilot 조건과 stability/LR 탐색은 이 notebook에서 재실행하지 않는다.
기존 고정 config의 batch ladder를 사용하되 신규 stability run은 만들지 않는다.
완료된 조건은 같은 commit/config에서 건너뛴다. 중단된 프로세스의 lock은 실제 종료를
확인한 뒤 해당 lock만 수동 처리한다. Drive가 아닌 임시 디스크만으로 영속성을 주장하지 않는다.

반환 파일: `ttm-new-data-feasibility.zip`, `moirai1-new-data-feasibility.zip`.
정상 판단은 exit code뿐 아니라 각 조건 `fp32_batch1_supported`, CUDA 장치,
공식 revision, 실행 commit, data hash, 선택/validation window hash 확인을 포함한다.
batch 1 실패와 더 큰 batch OOM을 구분한다. 높은 채널 수 때문에 실패해도 채널 분할,
임의 제거, 모델 변경을 자동 적용하지 않는다.

기존 `pilot_review`는 이전 52개 조건 전용이다. 신규 archive를 이전 importer에 넣거나
과거 성공으로 덮어쓰지 않는다. 반환 후 이번 commit의 eligibility와 계획으로 재생성 대조하고,
별도의 review 기록으로 반영해야 한다. 검증 전 archive 내용으로 gate를 자동 열지 않는다.

## 본 실험 전 남은 단계

1. 신규 GPU 32개 조건 실행·회수·검증, 특히 Traffic의 공동 다변량 메모리.
2. PEMS08 등 출처 확인 결과와 GPU 제약을 합쳐 최종 dataset set 제안.
3. 기존 validation-only pilot 근거로 모델별 설정안 확정 검토. 기존 200-step 실험은
   수렴 증거가 아니며 최종 budget으로 자동 승격하지 않는다. 필요한 추가 검증은 별도 설계한다.
4. primary MSE/MAE 및 point statistic, channel/dataset 집계, seed/sample 수,
   모델별 LR/step budget과 정지 규칙, 시간 block 불확실성, 데이터 특성 정의를 명문화한다.
5. 최종 체크포인트·프로토콜 lock 및 dry-run을 검토하되 **본 실험 실행은 하지 않는다**.

현재 `protocol_frozen=false`, `main_experiment_allowed=false`를 유지한다.
