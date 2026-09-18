# 연구계획 요약

## 가제

**시계열 데이터 특성에 따른 TSFM의 Zero-Shot–Few-Shot 성능 전환점 분석**

영문 가제: *When Does Fine-Tuning Beat Zero-Shot Forecasting? Data-Characteristic-Dependent Crossover Points of Time-Series Foundation Models*

## 목적

TSFM-Bench의 Zero-Shot 및 5% Few-Shot 평가를 출발점으로 삼아 학습 데이터 비율을 더 촘촘히 변화시키고, Few-Shot Fine-Tuning이 Zero-Shot보다 우수해지는 전환 구간을 데이터셋·모델·예측 길이별로 분석합니다.

## 최종 모델

연구 모델은 다음 두 개로 제한합니다.

1. **TTM (Tiny Time Mixers)**
   - 비교적 작은 사전학습 시계열 모델로 반복 Fine-Tuning 실험의 계산 부담이 상대적으로 낮습니다.
   - 제한된 학습 데이터에서 데이터 효율성과 전환 구간을 측정하기 적합합니다.

2. **MOIRAI**
   - 다양한 주기와 다변량 시계열을 지원하는 확률 예측 기반 모델입니다.
   - 구조와 예측 방식이 TTM과 달라 전환점이 모델에 따라 달라지는지 비교할 수 있습니다.

기준논문의 과거 모델을 재현하는 대신, 실험 시작 시점의 최신 공식 세대를 사용합니다. TTM은 `ibm-granite/granite-timeseries-ttm-r3`, MOIRAI는 `Salesforce/moirai-2.0-R-small`을 1차 후보로 사용합니다. 파일럿에서 4개 horizon의 Zero-Shot과 full-parameter Fine-Tuning 지원 여부를 확인한 뒤 최종 확정하며, 모델 가중치 revision과 공식 코드 commit을 고정합니다. 실험 도중 새 버전이 공개되어도 교체하지 않습니다.

## 전환점 정의

동일한 테스트 구간에서 다음 개선률을 계산합니다.

```text
개선률(r) = [Zero-Shot 손실 - Few-Shot 손실(r)] / Zero-Shot 손실
```

복수 시드와 불확실성 분석에서 개선이 안정적으로 관측되는 최초 학습량을 실용적 전환점으로 정의합니다.

- 끝까지 개선되지 않으면: 전환점 미관측
- 순위가 여러 차례 바뀌면: 최초 역전과 지속적 역전을 구분
- 관측 비율 사이에서 역전되면: 하나의 값이 아니라 구간으로 보고
- 비율과 함께 실제 학습 윈도 수를 보고

## 실험 범위

### 데이터셋

ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, PEMS08, Solar, Wind, Weather, AQShunyi, Exchange, ZafNoo, CzeLan의 14개 데이터셋을 사용합니다.

### 학습 비율

`0%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%`

0%는 대상 데이터로 파라미터를 업데이트하지 않는 Zero-Shot입니다. 나머지는 시간상 훈련 구간에서 지정된 비율의 학습 윈도를 사용한 Few-Shot Fine-Tuning입니다.

### 예측 및 평가

- 입력 길이 기본값: 512
- 예측 길이: 96, 192, 336, 720
- 평가 지표: MAE, MSE
- rolling stride: 1
- 1단계: 전체 조건을 1개 시드로 탐색
- 2단계: 전환 후보 주변을 3개 이상 시드로 검증
- 동일한 모델·데이터셋·예측 길이에서는 모든 학습 비율이 동일한 validation/test 구간을 사용

테스트 평가는 각 rolling origin에서 직전 관측값을 context로 사용합니다. 따라서 앞선 테스트 시점에서 새로 관측된 실제 값이 다음 예측 window의 context에 포함될 수 있습니다. 이는 재귀적으로 예측값을 이어 붙이는 고정-origin 평가가 아니라 rolling-origin 평가이며 논문에 명시합니다.

## 확정 실험 프로토콜

### 데이터 분할과 정규화

- 데이터셋별 TSFM-Bench 분할을 재현합니다.
- ETTh1을 포함한 6:2:2 데이터셋은 앞 60%를 train, 다음 20%를 validation, 마지막 20%를 test로 사용합니다.
- 7:1:2 데이터셋은 앞 70%를 train, 다음 10%를 validation, 마지막 20%를 test로 사용합니다.
- `StandardScaler`는 train 구간에만 fit하고 validation/test에는 transform만 적용합니다.
- 테스트 구간은 학습 함수에 전달하지 않습니다.

### Few-Shot 표본 정의

주 분석은 TSFM-Bench와의 비교 가능성을 위해 **train 구간의 전체 후보 window에서 균등하게 선택한 uniform window sampling**을 사용합니다.

기존 구현의 `int(1 // sampling_rate)` 간격 방식은 사용하지 않습니다. 각 조건에서 다음을 명시적으로 계산합니다.

```text
k = max(1, floor(sampling_rate × 전체 train window 수))
```

- 중복 없이 정확히 k개의 window index를 전체 train 구간에 균등 배치합니다.
- 요청 비율, 전체 후보 window 수, 실제 선택 window 수, 실효 비율을 모두 기록합니다.
- `drop_last=False`를 사용해 작은 비율에서도 학습 batch가 0개가 되지 않게 합니다.
- 시간순 prefix sampling은 주 분석이 아니라 배포 상황을 모사하는 강건성 분석 후보로 둡니다.
- random sampling을 사용할 경우 별도 강건성 분석으로 구분하고 Python, NumPy, PyTorch, DataLoader seed를 모두 기록합니다.
- window 중첩으로 실제 고유 시점 노출량이 비율보다 클 수 있으므로 unique observed time points도 가능하면 기록합니다.

### Fine-Tuning 예산

주 분석은 TSFM-Bench 재현성을 위해 다음을 사용합니다.

- optimizer: Adam
- loss: MSE
- 최대 20 epochs
- validation loss 기반 early stopping
- patience: 3
- full-parameter fine-tuning

데이터 비율이 커지면 epoch당 optimizer step도 늘어나므로, 학습 데이터량과 계산량의 효과가 함께 변할 수 있습니다. 이를 투명하게 하기 위해 실제 epoch, optimizer step, 학습 시간과 GPU 메모리를 모두 기록합니다.

강건성 분석에서는 전환 후보 주변의 일부 조건에 대해 **동일 optimizer-step 예산**을 적용해 결론이 계산량 차이에만 의존하는지 확인합니다. 주 분석과 강건성 분석 결과를 혼합하지 않습니다.

### 모델별 예측과 비결정성

- MOIRAI 2.0의 patch/tokenization, quantile 출력과 point forecast 변환은 최신 공식 구현을 따라 파일럿에서 확정하며, MOIRAI 1.x의 patch size 64·100개 확률 표본 중앙값 설정을 그대로 가정하지 않습니다.
- 확률 또는 quantile 예측에 난수가 사용되면 관련 seed와 point forecast 집계 규칙을 명시적으로 고정하고 기록합니다.
- TTM-R3의 모델 revision, context length, prediction length 및 모델 변형을 결과에 기록합니다.
- CUDA 및 확률 sampling 때문에 bitwise 재현이 보장되지 않을 수 있으므로 deterministic 설정 적용 여부와 비결정성 경고를 환경 정보에 저장합니다.
- 모델별 batch size와 AMP는 메모리에 맞게 다를 수 있지만, 데이터 분할·표본 index·평가 구간·지표는 동일하게 유지합니다.

## 실험 설계

1. 공식 전처리 데이터의 행·열·결측·시간 열을 기준 논문과 대조합니다.
2. 시간순 분할을 재현하고 정규화 통계는 훈련 구간에서만 계산합니다.
3. TTM과 MOIRAI 각각 한 데이터셋에서 Zero-Shot 및 5% Few-Shot을 먼저 재현합니다.
4. 세 데이터셋 파일럿에서 두 모델의 데이터 처리·평가 파이프라인을 검증합니다.
5. 학습 비율을 0.5/1/2/5/10/20/50/100%로 확장합니다.
6. 비율별 표본 index, 시드, 실제 optimizer step, 조기 종료와 계산 예산을 기록합니다.
7. MAE·MSE, 실제 학습 윈도 수, 학습·추론 시간과 가능하면 GPU 메모리를 측정합니다.
8. 데이터 특성은 훈련 구간에서만 계산합니다.
9. 전체 조건을 단일 시드로 탐색한 뒤 전환 후보 주변만 다중 시드로 재검증합니다.
10. 전환 후보 주변 일부 조건에서 동일 optimizer-step 강건성 분석을 수행합니다.

## 파일럿

계산량과 오류 추적을 고려해 다음 순서로 진행합니다.

1. TTM: ETTh1, horizon 96의 Zero-Shot 및 5% Few-Shot
2. MOIRAI: 동일 조건
3. TTM·MOIRAI × ETTh1·ETTm1·Electricity의 단계적 파일럿
4. 파일럿 통과 후 14개 데이터셋 본 실험

파일럿에서는 두 모델에 동일한 데이터 분할, 선택된 train window index, 테스트 구간과 평가 지표를 적용합니다. 모델 고유의 context length, patch size, 확률 예측 집계 방식은 별도 설정과 메타데이터로 관리합니다.

## 독립 구현과 라이선스

TSFM-Bench의 분할 비율, rolling 평가 개념, 공개 설정값과 모델 API 사용 방식을 참고하되 코드는 직접 복사하거나 수정해 포함하지 않고 독립적으로 작성합니다.

- TSFM-Bench 코드 및 자료의 라이선스 조건을 별도로 기록합니다.
- IBM `granite-tsfm` 코드와 TTM 모델의 라이선스를 확인하고 고지합니다.
- Salesforce `uni2ts` 코드 라이선스와 MOIRAI 가중치 라이선스를 분리해 기록합니다.
- 사용한 외부 패키지, 모델 가중치, 데이터셋의 출처·버전·라이선스를 manifest로 관리합니다.

## 실행 환경과 결과 보존

- GPU 실험은 Google Colab Pro+에서 수행합니다.
- 런타임 중단에 대비해 실험별 상태와 소용량 결과를 GitHub에 지속적으로 저장합니다.
- 완료된 실험은 재실행하지 않도록 resume 기능을 사용합니다.
- 데이터셋, 모델 체크포인트, 캐시와 대용량 예측 배열은 Git에 저장하지 않습니다.
- 모든 결과에는 모델, checkpoint/revision, 데이터셋, horizon, 요청 비율, 실제 학습 윈도 수, 실효 비율, 시드, epoch, optimizer step, 실행 시간, 환경 정보와 Git commit SHA를 기록합니다.
- 병렬 Colab 실행이 필요하면 실행 단위별 결과 branch를 사용하고 main에 직접 동시에 push하지 않습니다.

## 해석 범위

14개 데이터셋에는 ETT 4종과 수액 흐름 2종처럼 유사 계열이 포함되므로 데이터셋들이 완전히 독립적이라고 간주하지 않습니다. 계열별 민감도 분석을 병행하고, 산업 전체에 통용되는 임계값이나 인과관계는 주장하지 않습니다.
