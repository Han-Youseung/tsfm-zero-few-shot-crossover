# Validation 예산 확인 결과와 bounded-compute 결정

후속 상태: 사용자 승인에 따른 본 실험 v1은 [main-study](main-study.md)에 별도로 고정했다.
아래의 실행 금지/통합 gate 대기 설명은 budget 검토 당시 기록이며 당시 manifest는 보존한다.

## 실제 결과

실행 commit `d6c63f812b36d6c17df56016e6b9415558e11cdc`의 두 archive를 검증했다.
각 2조건, 총 4조건이며 모두 NVIDIA A100-SXM4-40GB에서 실행했다.
`budget_review`는 실행 계획, 공식 model/source revision, 실제 CUDA/FP32, data hash,
train sampling/validation windows를 대조하고 early stopping을 history로 재계산한다.
실제 optimizer step, batch, 노출 수 및 equivalent epoch도 대조한다.
결과는 `results/manifests/pilot/budget_review.json`에 보존한다.

| 모델 | 데이터 | Zero-Shot NMAE | Best validation NMAE | Best step | Stop step | 정지 사유 |
|---|---|---:|---:|---:|---:|---|
| TTM | ETTh1 | 0.374348 | 0.372952 | 100 | 400 | 3회 비개선 |
| TTM | Electricity | 0.227423 | 0.220192 | 900 | 1000 | 최대 step |
| MOIRAI | ETTh1 | 0.386524 | 0.384896 | 400 | 700 | 3회 비개선 |
| MOIRAI | Electricity | 0.290667 | 0.263194 | 1000 | 1000 | 최대 step |

이 표는 H96/5%/seed1729/64 validation windows의 pilot이다. test 성능이나 crossover
결과가 아니며 모든 데이터/horizon/rate에 같은 개선을 보장하지 않는다.
MOIRAI/Electricity는 마지막 step에서도 개선 중이다. **수렴은 입증되지 않았다.**
이전 200-step pilot은 50-step 평가, 이번은 100-step 평가이므로 best checkpoint
지표를 동일 평가 횟수의 직접 비교로 해석하지 않는다.

## 선택한 학습 예산

추가 탐색을 무한히 연장하지 않고 **bounded-compute crossover**를 주 연구 대상으로 삼는다.
TTM LR1e-4, MOIRAI LR5e-6, 각각 최대1000 optimizer steps를 사용한다.
두 모델 모두 batch1/accumulation1, FP32, full-parameter, eval100steps,
patience3evaluations, drop_last=False, clipping 없음, 기존 공식 optimizer 경로를 유지한다.
모델별 모든 dataset/rate에 동일한 규칙을 적용한다. 실제 step은 early stopping으로 달라질 수 있다.
Checkpoint는 fine-tuning 후 평가한 checkpoint 중 validation NMAE 최저를 선택하며,
test로 고르거나 악화됐다는 이유로 Zero-Shot checkpoint로 교체하지 않는다.

1,000은 수렴 기준이 아니라 검증된 계산 상한으로 선택한 값이다. 결과는 이 예산에
조건부이며 더 긴 학습의 crossover와 달라질 수 있다. 동일 최대 epoch 강건성 분석은
주 분석 이후 전환 후보 주변에서 별도로 설계하며 지금 실행하지 않는다.

중요: batch1/최대1000step이면 최대 training window exposure는 1000회다.
선택된 window pool이 더 크면 모든 선택 window를 실제 방문하지 않을 수 있다.
100%는 **train 후보 pool 100% 접근 허용**이지 1 full epoch 또는 모든 window 사용이 아니다.
selected window 수와 실제 unique visited/exposure/등가 epoch를 반드시 분리 기록한다.

## 공통 설계

`configs/study/preexperiment.yaml`은 연구 설정안이며 실행 허용 config가 아니다.
Primary9, context512, H96/192/336/720, nested rates0.5/1/2/5/10/20/50/100%,
seeds1729/2718/31415. 원 checkpoint·native scaling·point rule은 Zero/Few-Shot 사이에 고정한다.
모델별 seed는 sampling/training/stochastic prediction을 함께 정하는 paired replicate이며
각 변동 원인의 독립적 효과로 해석하지 않는다. TTM Zero-Shot의 동일 결과 재사용은 실행
계층에서 검증한 뒤 적용한다. 현재 counts에는 seed별 평가를 명시적으로 포함한다.

주 지표는 train population std로 error만 정규화한 channel-macro MAE이다.
원 단위 예측을 유지하고 normalized MSE 및 원 단위 MAE/MSE를 보조 보고한다.
MOIRAI point는 100 samples의 torch median이며 TTM은 prediction_outputs이다.
train-constant 채널은 모델/원 단위 평가에서 보존하고 normalized metric에서만 명시적으로 제외한다.
결측 input은 causal ffill, leading missing context 차단, target 미보간,
train은 완전 target window만 두 모델에 공통 적용하며 metric은 동일 관측 mask를 사용한다.

60/20/20 시간순 분할, train-only 통계, validation64개의 사전 고정 시간층화 subset,
최종 test stride1 rolling-origin을 유지한다. test 관측값은 이후 context로만 사용하며
통계/학습/checkpoint 선택에 사용하지 않는다. 이번 사전 점검은 test **값을 읽지 않는다**.
원본 전체 checksum/format audit와 test 점수 평가는 구분한다.

## 사전 점검 및 남은 구현

`python -m tsfm_crossover.experiments.study_preflight`는 GPU 없이 각 prepared 파일 hash,
적격 train/validation window, 비율별 정확한 k 및 nominal test window 수를 확인한다.
test 개수는 split 길이 metadata 기반이며 결측 mask 적용 후 최종 유효 관측 개수와 다르다.
주 실행 계획은 Zero-Shot216 + Few-Shot1728 = 1944조건,
최대 optimizer step 합은 1,728,000이다. early stopping 전 상한이며 실행시간 예측이 아니다.
Traffic MOIRAI H720 FP32 학습은 80GB급 검증 자원이 필요하다. 미배정 시 임의 축소/AMP전환을
하지 않고 해당 조건을 대기 상태로 둔다. 장치가 다른 wall time은 직접 속도 순위로 비교하지 않는다.

다음 코딩 단계는 **본 실험 실행기·분석 계층의 통합 검증**이다. 다음은 아직 완료되지 않았다:

- 전체 rate/seed manifest를 사용하는 실행기, 최적 checkpoint 복원 후 공통 test 평가 연결.
- 중단/재개·완료 건너뛰기·partial failure·GPU 자원 대기 통합 테스트.
- 최초/지속 crossover 구간 및 미관측/비단조 규칙 구현.
- paired seed 및 시간 block 불확실성, ETT 출처 묶음별 보고/집계.
- train-only 데이터 특성 추출과 데이터셋 수가 적은 특성별 탐색 분석의 한계 명시.

이 통합 gate 전에는 `protocol_frozen=false`, `main_experiment_allowed=false`이다.
추가 GPU pilot을 지금 요청하지 않는다. 이후 필요한 engineering smoke만 별도로 구분한다.
본 실험이나 test metric은 아직 실행하지 않았다.
