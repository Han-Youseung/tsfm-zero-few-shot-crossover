# TSFM Zero-Shot–Few-Shot Crossover

시계열 데이터의 특성과 학습 데이터 양에 따라 Time-Series Foundation Model(TSFM)의 **Zero-Shot 성능을 Few-Shot Fine-Tuning이 언제 넘어서는지** 분석하는 연구 저장소입니다.

> 현재 단계: 연구 주제와 실험 설계 확정 / 데이터 검증 및 실험 환경 구축 전

## 연구 질문

새로운 시계열을 예측할 때 사전학습 모델을 그대로 사용하는 Zero-Shot과, 대상 데이터 일부로 추가 학습하는 Few-Shot Fine-Tuning 중 어느 방법이 더 좋은가? 성능 순위가 바뀌는 전환 구간은 데이터 특성·모델·예측 길이에 따라 어떻게 달라지는가?

단일 비율을 정확한 전환점으로 단정하지 않고, 관측한 학습량 사이의 **전환 구간**으로 보고합니다. 예를 들어 2%에서는 Zero-Shot이 우수하고 5%에서는 Fine-Tuning이 우수하다면 전환 구간은 `2% 초과~5% 이하`입니다.

## 연구 범위

### 데이터셋 14개

| 계열 | 데이터셋 |
|---|---|
| 변압기 | ETTh1, ETTh2, ETTm1, ETTm2 |
| 전력·교통 | Electricity, Traffic, PEMS08 |
| 에너지·기상 | Solar, Wind, Weather |
| 대기질·환율 | AQShunyi, Exchange |
| 수액 흐름·환경 | ZafNoo, CzeLan |

TSFM-Bench에서 Zero-Shot과 5% Few-Shot 결과가 모두 보고된 장기 예측 데이터셋을 대상으로 합니다.

### 최종 모델 2개

- **TTM (Tiny Time Mixers)**: 비교적 작은 모델로, 제한된 계산 자원에서 학습량별 Fine-Tuning 실험을 반복하기 적합합니다.
- **MOIRAI**: 다양한 주기와 다변량 시계열을 다루는 범용 확률 예측 모델로, TTM과 구조 및 예측 방식이 달라 모델별 전환점 차이를 비교할 수 있습니다.

파일럿에서는 두 모델 모두 동일한 데이터 분할·학습 비율·평가 지표를 사용합니다. 먼저 각 모델에 대해 한 데이터셋의 Zero-Shot과 5% Few-Shot을 재현한 뒤 3개 데이터셋 파일럿과 14개 데이터셋 본 실험으로 확대합니다.

### 학습량 후보

`0% (Zero-Shot), 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%`

비율뿐 아니라 실제 학습 윈도 수도 함께 기록합니다.

### 평가

- 예측 길이: 96, 192, 336, 720
- 지표: MAE, MSE
- 반복: 1개 시드로 전체 조건 탐색 후 전환 후보 주변을 3개 이상 시드로 검증
- 주요 분석: Zero-Shot 대비 오차 개선률, 최초 역전 구간, 지속적 역전, 전환점 미관측 사례
- 데이터 특성: 계절성, 추세, 정상성, 변화·전환, 분포 이동, 채널 상관, 비정규성

## 진행 계획

- [x] 연구 질문 및 범위 설정
- [x] 14개 데이터셋 선정
- [x] 최종 모델 TTM·MOIRAI 선정
- [x] 실험 설계 초안 작성
- [ ] 공식 데이터 다운로드 및 파일·형식 검증
- [ ] TTM·MOIRAI 실행 환경 구축
- [ ] 모델별 Zero-Shot 및 5% Few-Shot 단일 조건 재현
- [ ] 2개 모델 × 3개 데이터셋 파일럿 실험
- [ ] 14개 데이터셋 전체 학습량 스윕
- [ ] 전환 후보 주변 다중 시드 검증
- [ ] 데이터 특성과 전환 구간 분석
- [ ] 논문 작성

## 실행 환경 및 저장 원칙

- GPU가 필요한 모델 실행은 Google Colab Pro+에서 수행합니다.
- 코드·설정·소용량 결과·실험 메타데이터는 이 저장소에 지속적으로 커밋합니다.
- 데이터셋, 모델 체크포인트, Hugging Face 캐시 및 대용량 중간 산출물은 Git에 저장하지 않습니다.
- 각 결과에는 모델·데이터셋·예측 길이·학습 비율·시드·실제 학습 윈도 수·Git commit SHA를 기록합니다.
- 완료된 실험은 재실행하지 않도록 상태를 기록하고 중단 후 재개할 수 있게 구현합니다.

## 문서

- [현재 연구계획 요약](docs/research-plan.md)
- [초기 연구계획 PDF](docs/research-proposal.pdf) — 최종 모델을 2개로 축소하기 전의 후보 모델 검토 초안입니다.

## 기준 연구

Li et al., *TSFM-Bench: A Comprehensive and Unified Benchmark of Foundation Models for Time Series Forecasting*, KDD 2025.

- 공식 코드: [decisionintelligence/TSFM-Bench](https://github.com/decisionintelligence/TSFM-Bench)

## 주의사항

본 연구 결과는 선택한 14개 벤치마크 안에서 해석합니다. 특정 데이터셋에서 발견한 임계값이 같은 산업의 다른 사업장에도 그대로 적용된다고 주장하지 않습니다.
