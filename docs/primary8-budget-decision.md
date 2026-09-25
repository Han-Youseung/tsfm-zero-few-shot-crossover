# Primary 8 선정 및 학습 예산 확인

## 결정

사용자가 자원 제약에 따른 제외를 승인하고 판단을 위임했다. 이번 primary는
**ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Solar, Weather, Tetouan**의 8개로 정한다.
이 선택은 test 점수가 아니라 출처·결측 처리 및 동일 40GB 자원에서 두 모델/네 horizon의
공통 실행 가능성에 근거한다. 64개 model-dataset-horizon의 batch-1 GPU 증거가 있다.
long-run 수렴/최종 성능이나 전체 본 실험 성공을 보장하는 증거는 아니다.

Traffic은 **두 모델과 모든 horizon의 primary에서 함께 제외**한다. 작은 horizon만 남겨
불균형한 primary 표를 만들지 않는다. 데이터 삭제는 하지 않으며, 기존 H720의 MOIRAI
학습 OOM과 나머지 성공 기록을 그대로 유지한다. 추후 별도 resource sensitivity 대상으로만
검토할 수 있으며 지금 자동 실행하지 않는다. 80GB 재시도는 더 이상 필수 gate가 아니다.

80GB preflight는 사용자 보고상 A100-SXM4-40GB, 39.4935 GiB가 배정되어 중단되었다.
본체/ZIP 생성은 실행되지 않았다. 이를 검증된 새 GPU 실험 실패로 기록하지 않고
`user_report_not_imported_gpu_result` / `not_run_gpu_not_allocated`로 구분한다.

PEMS08, Exchange, ZafNoo, CzeLan은 출처/시간축 조건 미해결로 미포함;
Wind와 AQShunyi의 기존 제외 사유도 유지한다. 숫자를 맞추기 위한 대체 데이터 추가는
중단한다. 8개 데이터셋이 독립적인 8개 도메인이라는 주장은 하지 않는다. ETT는
변압기별로 hourly/minutely 변형을 묶어 보고하고, 다른 출처로의 일반화는 제한적으로 해석한다.

## 모델별 선택과 남은 예산 확인

기존 validation-only LR 비교에서 TTM `1e-4`가 ETTh1/Electricity 모두 우세했다.
MOIRAI `5e-6`은 두 대표에서 모두 개선했으며, `5e-5`의 ETTh1 악화와 데이터셋 간
tradeoff를 피하는 보수적 선택이다. 이는 새로운 사전적 선택 규칙이 아니라 이미 관측한
validation 결과에 근거한 결정임을 명시한다. test 결과는 사용하지 않았다.

이 LR은 모델 안에서 데이터셋/학습 비율별로 바꾸지 않는 후보로 고정한다.
FP32, batch 1, accumulation 1, full-parameter, 기존 공식 AdamW 구성/weight decay
(TTM 0.01, MOIRAI 0.1), gradient clipping 없음, 외부 scaler 없음을 유지한다.
MOIRAI 100 samples와 torch sample median, TTM prediction_outputs를 유지한다.
모델별 공식 objective는 동일하지 않으며 변경하지 않는다.

기존 200-step pilot에서 최적 checkpoint가 상한에 도달한 조건이 있어, 상한을 최종
예산으로 즉시 채택하지 않는다. 다음은 **4개 조건만** 추가 확인한다:

- ETTh1/Electricity × TTM/MOIRAI, H96, 5%, seed 1729.
- 모델마다 LR 1개, FP32 batch 1, 최대 1,000 optimizer steps.
- validation 64개 동일 window, 100 steps마다 평가, 3회 연속 비개선 시 중단.
- 25 steps마다 resume checkpoint; drop_last=False; 원 pretrained에서 독립 시작.
- 이전 200-step 결과를 새 코드의 실행으로 병합하거나 checkpoint 이어붙이지 않는다.

이는 초기 작은 budget의 과소학습 위험을 점검하는 제한된 validation 확인이며
LR sweep, crossover 분석 또는 본 실험이 아니다. 1,000은 **이번 pilot의 계산 상한**으로
정한 값이며 수렴 보장이나 확정된 본 실험 step 수가 아니다. 3회 patience도 validation
100-step 간격 기준이며 TSFM-Bench의 epoch patience를 가져온 것이 아니다.

반환 후 모델별로 두 대표의 normalized MAE 추이, best step, 마지막 평가 개선폭,
중단 사유/시간/메모리/실제 window exposure를 함께 검토한다. 1,000에서도 개선 중이면
수렴을 주장하지 않고 bounded-compute 연구로 명시할지 판단한다. 자동 추가 탐색은 하지 않는다.

## 평가/설계 결정안

primary metric 후보는 train std로 error만 정규화한 channel-macro MAE다. sample median과
목표가 맞고 서로 다른 채널 단위를 섞는 문제를 줄인다. raw-unit MAE/MSE와 normalized
MSE는 보조 보고한다. train-constant 채널은 학습/원 단위 평가에 유지하고 정규화 metric에서만
기존 명시적 기준으로 제외·개수를 보고한다. 모든 비율에서 같은 mask/window/statistic을 쓴다.
dataset별 결과가 주이며 전체 raw MSE 단순 평균은 주 결론으로 사용하지 않는다.
ETT 출처 묶음을 고려한 집계와 seed/시간 block 불확실성을 명문화한 뒤 protocol을 잠근다.

현재 dry-run 설계는 8 dataset × 2 model × 4 horizon × (Zero-Shot + 8 rates) × 3 seeds
= **1,728조건**이다. Few-Shot 학습은 1,536회이다. rates는
0.5/1/2/5/10/20/50/100%, seeds는 1729/2718/31415로 계획한다.
이는 실행 횟수 산술 계획이지 성능 결과·runtime 보장 또는 실행 허가가 아니다.
동일 seed 안에서 nested 표본을 유지하고 동일 test window에서 paired 비교한다.
Zero-Shot은 학습 seed 변동이 없으며 MOIRAI의 예측 seed 변동과 학습 변동을 구분한다.

## 실행 방법

`notebooks/32_budget_confirmation.ipynb`를 새 세션에서 위부터 실행한다.
**A100 40GB면 충분한 기존 대표 조건**을 사용하며 80GB를 요구하지 않는다.
TTM과 MOIRAI는 별도 세션으로 설치한다. 실행할 **이번 준비 commit** 전체 SHA를 입력한다.
설치/새 1,000-step GPU 실행은 아직 pending이고, 1-step 메모리 통과가 long-run 보장은 아니다.

각 모델에서 정확히 ETTh1/Electricity 2개 조건이 표시되는지 확인한 뒤 `RUN_BUDGET`을
입력한다. 성공한 기존 16조건/Traffic/전체 feasibility는 실행하지 않는다.
Drive 연결은 사용자가 승인한다. 결과·checkpoint는 Drive에 보존하며 ZIP에는 소용량
JSON/CSV만 포함한다. `ttm-budget-confirmation.zip`, `moirai1-budget-confirmation.zip`을 반환한다.
프로세스가 죽은 경우만 stale running.lock을 확인 후 수동 처리하고 임의 결과 삭제는 하지 않는다.

반환 후 이번 실행 commit/config/data/model/window identity와 RNG/resume 상태를 검증한다.
과거 32/52조건 importer를 그대로 적용하지 않는다. 그 다음 최종 optimizer budget과
분석 계획/실행량을 검토해 본 실험 전 checklist를 완료한다.
지금은 `protocol_frozen=false`, `main_experiment_allowed=false`, test 실행 금지를 유지한다.
