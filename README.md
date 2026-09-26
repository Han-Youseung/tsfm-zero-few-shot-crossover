# TSFM Zero-Shot–Few-Shot Crossover

시계열 데이터 특성과 대상 데이터 학습량에 따라 Time-Series Foundation Model(TSFM)의 Few-Shot Fine-Tuning이 Zero-Shot 성능을 넘어서는 **전환 구간**을 분석하는 연구 저장소입니다.

> 현재 단계: 사용자 승인에 따라 primary9 본 실험 실행기와 분석·재개 경로를 준비했습니다. 실제 본 실험 GPU 실행은 pending입니다. [실행 가이드](docs/main-study.md), `notebooks/40_main_study.ipynb`를 사용합니다. 과거 bundle은 계속 차단합니다.

## 연구 범위

- 선택 모델: `ibm-granite/granite-timeseries-ttm-r3`, `Salesforce/moirai-1.1-R-small`
- Primary9: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Solar, Weather, Tetouan, Traffic
- 학습 비율: `0%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%`
- 예측 길이: `96, 192, 336, 720`
- 주 평가: 동일한 test window의 stride1 rolling-origin, train-std normalized channel-macro MAE. Raw MAE/MSE와 normalized MSE는 보조 지표입니다.

선택 모델의 Hugging Face revision, 공식 코드 commit과 GPU budget에서 검증한 패키지 버전을 고정합니다. 본 실험 v1은 context512, FP32, batch1, full-parameter, 최대1000steps, eval100/patience3입니다. TTM LR1e-4, MOIRAI LR5e-6을 모든 dataset/rate에 적용합니다. 이 예산은 수렴을 보장하지 않으며 test 성능으로 설정을 바꾸지 않습니다.

TSFM-Bench는 관련 선행연구로만 인용합니다. 그 결과·체크포인트·실험 조건을 재현하거나 직접 비교하지 않고, 코드와 프로토콜도 복사·번안하지 않습니다. 구현은 최신 TTM·MOIRAI의 공식 구현과 공개 API를 사용해 독립적으로 작성합니다.

## 설치

코어 프레임워크는 Python 3.11–3.13을 지원하고 GPU 또는 PyTorch 없이 실행할 수 있습니다.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

모델 의존성은 `requirements/ttm.txt`와 `requirements/moirai1.txt`로 분리하며 같은 환경에 함께 설치하지 않습니다. 기본 테스트는 실제 모델을 다운로드하지 않습니다.

4.6단계 FP32 GPU 근거 12개를 검증했습니다. TTM은 A100, MOIRAI는 T4에서
ETTh1·7채널·batch 1·context 512의 네 horizon을 실행했습니다.
[GPU 근거 및 공식 loss 검토](docs/gpu-evidence-review.md)에 범위와 제약을 기록합니다.

공통 adapter의 설정·checkpoint·RNG 정책과 실행 방법은 [adapter 문서](docs/adapters.md)에 있습니다.
다음 GPU 실행은 `notebooks/40_main_study.ipynb`를 모델별 별도 세션에서 실행합니다.
[14개 원 후보 준비 상태](docs/phase6-data-readiness.md)와 [pilot 기록](docs/validation-pilot.md)은 과거 근거로 보존합니다. 본 실험 설정은 `configs/study/main.yaml`이며 이전 pilot 설정의 test 금지는 유지합니다.

## 구조

- `configs/`: dataset, model, experiment, protocol YAML
- `src/tsfm_crossover/config/`: strict 설정 schema, loader, ID 생성
- `src/tsfm_crossover/tracking/`: 결과 schema, 원자적 저장, 실행 환경 메타데이터
- `src/tsfm_crossover/utils/`: seed와 재현성 유틸리티
- `src/tsfm_crossover/{data,models,experiments,evaluation,analysis}/`: 다음 단계의 독립 구현 경계
- `scripts/`: 재실행 가능한 CLI 진입점
- `notebooks/`: 모델별 Colab compatibility probe
- `tests/`: CPU 단위·통합 테스트
- `results/`: 소용량 JSON/CSV, summary, figure, manifest
- `data/`, `checkpoints/`, `logs/`: Git에 올리지 않는 로컬 산출물

세부 연구 프로토콜은 [docs/research-plan.md](docs/research-plan.md), 출처와 라이선스는 [docs/licenses-and-sources.md](docs/licenses-and-sources.md)에서 관리합니다.

데이터 계층은 60:20:20 시간순 분할, train-only 통계와 모델 독립적인 nested sampling manifest를 제공합니다. 본 실험 입력은 native scaling만 사용하며 기존 StandardScaler 기능은 비활성화됩니다. [실행 가이드](docs/main-study.md)의 명시적인 데이터 준비 셀만 승인된 공식 파일을 내려받습니다.

실제 데이터 배치, ZIP 안전 검사, 전체 audit 명령과 현재 차단 사유는 [real-data audit](docs/data-audit.md)에 기록합니다. 원본 데이터는 Git에 추가하지 않으며 `results/manifests/datasets/`에는 checksum과 집계 metadata만 저장합니다.

Bundle 생성 경로, 공식 ETT 값 대조, canonical fingerprint와 데이터셋별 판정은 [data provenance](docs/data-provenance.md), [canonical format](docs/canonical-format.md), [data decisions](docs/data-decisions.md)에 기록합니다.

모델·공식 코드 revision, 라이선스, context/horizon 제약과 당시 검증 기록은 [model compatibility](docs/model-compatibility.md)에 보존합니다. 본 실험 protocol v1 고정과 과거 compatibility manifest의 상태는 별개입니다.

## 저장 원칙

최신 절차는 [본 실험 v1](docs/main-study.md)입니다. Traffic을 포함한 primary9가 현재 기준입니다.
아래는 과거 단계별 판단 기록이며 현재 실행 지침이 아닙니다.

근거: [budget 검증](docs/budget-decision.md), [Traffic80GB 복구](docs/traffic-recovery-result.md).
Traffic은 primary9에 포함되며 MOIRAI H720 학습만 80GB급 자원을 기다립니다.
[Primary8 결정](docs/primary8-budget-decision.md), [복구 준비](docs/traffic-recovery.md),
[초기 준비](docs/pre-experiment-readiness.md)는 변경된 과거 판단의 기록입니다.

소스, YAML, 소용량 결과·요약·그림·manifest와 문서만 Git으로 관리합니다. 데이터셋, 모델 가중치, 캐시, 대용량 예측 배열과 원본 로그는 저장소에 추적하지 않습니다.
