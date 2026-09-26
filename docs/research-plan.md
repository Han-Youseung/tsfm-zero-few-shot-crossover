# 연구계획 요약

## 연구 질문과 범위

동일한 사전학습 모델·데이터셋·예측 길이에서 대상 train 데이터로 Fine-Tuning할 때 Zero-Shot 성능을 안정적으로 넘어서는 전환 구간을 측정하고, 이 구간과 시계열 특성의 관계를 분석합니다.

TSFM-Bench는 관련 선행연구로 인용할 뿐, 결과 재현·성능 개선·동일 조건의 직접 비교를 주장하지 않습니다. 코드와 프로토콜도 복사·번안하지 않습니다.

## 모델과 버전 고정

선택 모델은 `ibm-granite/granite-timeseries-ttm-r3`와 `Salesforce/moirai-1.1-R-small`입니다. MOIRAI 2.0의 실패 기록은 보존합니다. 호환성·파일럿 후 사용자 승인으로 [본 실험 v1](main-study.md)을 고정합니다. 새 본 실험의 실제 GPU 결과는 아직 없으며 [검증 범위와 실제 objective](gpu-evidence-review.md)와 구분합니다.

## 공통 실험 프로토콜

### 데이터와 분할

- Primary9: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Solar, Weather, Tetouan, Traffic. 기존14종을 숫자에 맞춰 강제로 포함하지 않습니다.
- 모든 데이터셋에 시간순 train 60%, validation 20%, test 20%를 적용합니다.
- 현재 두 모델은 native scaling만 사용하며 외부 scaler를 비활성화합니다. 향후 외부 scaler를 채택한다면 train에서만 fit해야 합니다.
- validation은 early stopping과 사전 모델 설정 확인에만, test는 최종 평가에만 사용합니다.
- test는 학습, scaler 갱신, early stopping, hyperparameter/checkpoint 선택에 사용하지 않습니다.

### 입력과 예측 길이

검증 후 context512와 horizons96/192/336/720을 고정했습니다. 자원 부족 때문에 임의로 축소하지 않고 해당 조건을 대기시킵니다.

### Few-Shot 표본

주 분석은 **nested temporally stratified window sampling**을 사용합니다. Train 후보 window 수를 `N`, 요청 비율을 `r`이라고 할 때 `k=max(1, floor(r×N))`개를 정확히 선택하고 다음 중첩을 보장합니다.

```text
S_0.5% ⊆ S_1% ⊆ S_2% ⊆ S_5% ⊆ S_10% ⊆ S_20% ⊆ S_50% ⊆ S_100%
```

전체 train 시간 범위를 고르게 대표하는 하나의 중첩 순서를 생성하고 비율별 prefix를 사용합니다. 비율별 독립 random sample을 다시 뽑지 않으며 `drop_last=False`를 적용합니다. 요청·실효 비율, 전체·선택 window 수, index manifest, 고유 관측 시점, seed와 manifest hash를 필수로 기록합니다. Prefix/suffix는 향후 민감도 분석용 interface로만 분리합니다.

### Fine-Tuning 예산

주 분석은 모델 내 동일한 최대 optimizer-step 예산을 사용합니다. 모델별 optimizer, learning rate, max steps, validation 주기, patience, batch size, gradient accumulation/clipping, AMP와 fine-tuning 대상은 test를 보지 않고 파일럿 validation·메모리 결과로 한 번만 결정합니다. TTM과 MOIRAI의 설정은 다를 수 있지만 한 모델 안에서는 모든 데이터셋·비율에 동일한 규칙을 적용합니다.

Actual steps, equivalent epochs, average window exposures, best validation/stopping step, 시간, peak GPU memory, trainable/total parameters를 기록합니다. 동일 최대 epoch 방식은 전환 후보 주변의 강건성 분석으로만 사용합니다.

### 평가와 공정성

- 동일 model–dataset–horizon의 Zero-Shot과 모든 Few-Shot은 같은 validation/test window를 사용합니다.
- 주 평가는 stride 1 rolling-origin이며 해당 시점까지 새로 관측된 test 값은 이후 context에 포함할 수 있습니다.
- Test 값은 모델/scaler 갱신, early stopping, hyperparameter, checkpoint 선택에 사용하지 않습니다.
- 인접 window는 독립이 아니므로 단순 window bootstrap을 사용하지 않고 seed 변동, 시간 block 또는 데이터셋 단위 불확실성을 사용합니다.
- 주 지표는 train-std normalized channel-macro MAE, 보조 지표는 raw MAE/MSE와 normalized MSE입니다. TTM prediction_outputs, MOIRAI100 samples의 torch median을 고정합니다.

Pretrained checkpoint, context, split, normalization, validation/test windows, point forecast, metric 계산과 optimizer-step 예산 규칙을 model–dataset–horizon 단위에서 고정합니다. Zero-Shot과 Few-Shot 사이의 핵심 차이는 train 데이터로 파라미터를 갱신했는지와 선택 window 수입니다.

## 전환 구간

```text
개선률(r) = [Zero-Shot 오차 - Few-Shot 오차(r)] / Zero-Shot 오차
```

사전 고정한3개 paired seeds 전체 grid에서 최초/지속적 역전, 미관측, 비단조를 보고합니다. Seed 범위는 기술통계이며 신뢰구간이 아닙니다. 관측 grid 사이의 연속 단조성이나 일반화된 유의성을 주장하지 않습니다. 전환 후보 주변의 동일-epoch 강건성 분석은 후속 별도 protocol입니다.

## 실행과 보존

- GPU 실험은 Colab Pro+에서, 설정·데이터 검증·테스트·분석은 로컬 CPU에서 실행합니다.
- 설정, seed, 패키지, commit, 장치, 시간, window/step 수와 환경 정보를 기록합니다.
- 원자적 상태 저장과 experiment ID로 완료된 조건을 재실행하지 않습니다.
- 데이터, 모델 가중치, cache, 대용량 예측 배열과 원본 로그는 Git에 저장하지 않습니다.

## 현재 단계

사용자의 본 실험 실행 승인 이후 현재 기준은 [main-study](main-study.md)와
`configs/study/main.yaml`이다. 실행기·resume·test 차단 및 분석의 CPU 통합 검증을 수행하고,
실제 main GPU 실행은 Colab notebook40으로 진행한다. 아래는 이전 단계 기록이며
당시 manifest의 `protocol_frozen=false`를 소급 수정하지 않는다.

최신 상태는 [budget 확인 및 bounded-compute 결정](budget-decision.md)이다.
아래 phase6 준비 설명은 이전 단계 기록이다. 현재 primary9와 학습 설정을 선택했으나
main runner/분석 통합 검증 전이며 test 실행과 protocol freeze는 아직 허용하지 않는다.

공통 adapter의 8개 GPU 조건을 확인했고, 현재는 [6단계 validation-only pilot](validation-pilot.md) 준비 단계입니다. 실제 pilot GPU 실행과 최종 protocol 확정은 pending입니다. [14개 데이터셋 준비 상태](phase6-data-readiness.md)를 별도 관리하며 기존 bundle을 자동 허용하지 않습니다. Electricity의 준비된 UCI 370채널 variant는 기존 321채널과 구분합니다. 전체 실험과 crossover 탐색은 실행하지 않습니다. TTM의 공식 channel-independent 구조와 MOIRAI의 공동 target 처리를 구분하며, 학습 objective와 평가 metric은 별개입니다.

Pilot은 기존 60:20:20 분할을 그대로 사용하며 train 내부 validation을 추가하지 않습니다. 학습 비율은 기존 중첩 grid 중 5% 한 조건만 사용합니다. 새 요청의 2/5/10/20/50/100%는 기존 grid의 부분집합으로 취급하고, 본 실험의 0.5/1% 제외를 자동 확정하지 않습니다. 후보 선택 지표는 사전에 선언한 채널 macro normalized MAE이며 최종 지표는 pilot 검토 후 결정합니다.
