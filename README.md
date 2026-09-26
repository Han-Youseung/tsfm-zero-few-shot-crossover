# TSFM Zero-Shot–Few-Shot Crossover

시계열 데이터 특성과 대상 데이터 학습량에 따라 Time-Series Foundation Model(TSFM)의 Few-Shot Fine-Tuning이 Zero-Shot 성능을 넘어서는 **전환 구간**을 분석하는 연구 저장소입니다.

> 현재 단계: 6단계 validation-only pilot 준비 — adapter GPU 8조건 확인 완료(T4). ETT 4종과 별도 Electricity UCI-370 variant 준비 완료. 실제 A100 pilot은 pending이며 기존 bundle 14종 차단을 유지합니다.

## 연구 범위

- 선택 모델: `ibm-granite/granite-timeseries-ttm-r3`, `Salesforce/moirai-1.1-R-small`
- 데이터셋: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, PEMS08, Solar, Wind, Weather, AQShunyi, Exchange, ZafNoo, CzeLan
- 학습 비율: `0%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%`
- 예측 길이: `96, 192, 336, 720`
- 주 평가: 동일한 test window의 stride 1 rolling-origin MAE·MSE

현재 선택 모델의 Hugging Face revision, 공식 코드 commit과 핵심 모델 패키지 버전은 고정했으며 자동 업그레이드하지 않습니다. Pilot은 context 512와 full-parameter fine-tuning을 사용합니다. 본 실험의 learning rate, step budget, batch size, precision 및 평가 정책은 아직 확정되지 않았으며 test 성능을 보고 선택하지 않습니다.

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
다음 GPU 실행은 `notebooks/30_validation_pilot.ipynb`를 TTM과 MOIRAI의 별도 세션에서 실행합니다.
[14개 데이터셋 준비 상태](docs/phase6-data-readiness.md)와 [pilot 범위·실행 방법](docs/validation-pilot.md)을 먼저 확인합니다. Test 평가·본 실험·최종 설정 확정은 이번 단계 범위가 아닙니다.

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

데이터 계층은 60:20:20 시간순 분할, train-only StandardScaler, validation/test rolling-origin window와 모델 독립적인 nested temporally stratified sampling manifest를 제공합니다. 사용법과 누수 방지 규칙은 [데이터 프로토콜](docs/data-protocol.md), [sampling 설명](docs/sampling.md), [dataset registry](docs/datasets.md)에 있습니다. 실제 데이터는 자동 다운로드하지 않습니다.

실제 데이터 배치, ZIP 안전 검사, 전체 audit 명령과 현재 차단 사유는 [real-data audit](docs/data-audit.md)에 기록합니다. 원본 데이터는 Git에 추가하지 않으며 `results/manifests/datasets/`에는 checksum과 집계 metadata만 저장합니다.

Bundle 생성 경로, 공식 ETT 값 대조, canonical fingerprint와 데이터셋별 판정은 [data provenance](docs/data-provenance.md), [canonical format](docs/canonical-format.md), [data decisions](docs/data-decisions.md)에 기록합니다.

고정한 모델·공식 코드 revision, 라이선스, context/horizon 제약, scaling 및 아직 실행하지 않은 Colab 검증은 [model compatibility](docs/model-compatibility.md)에 기록합니다. 후보는 아직 `frozen`이 아닙니다.

## 저장 원칙

최신 단계: [budget 검증 및 학습 예산 결정](docs/budget-decision.md).
두 모델의 validation-only 예산 확인을 마쳤습니다. 최대1000step bounded-compute 설정을
채택하고 CPU 사전 점검을 수행하되, 본 실험/분석 계층 통합 gate가 남아 실행은 금지합니다.

최신 결과: [Traffic FP32 복구 및 primary9](docs/traffic-recovery-result.md).
Traffic은 80GB GPU에서 일반 FP32 학습/복원을 통과했습니다. notebook33은 종료하고
primary9 overlay를 사용하는 notebook32의 validation-only budget 확인을 재개합니다.
아래 conditional/primary8/보류 설명은 과거 결정 이력입니다.

최신 우선 작업: [Traffic 복구 검증](docs/traffic-recovery.md).
Traffic 제외 결정을 철회하여 conditional로 유지하고, notebook 33에서 실제 Colab GPU/RAM에
맞춰 메모리 경로를 확인합니다. 아래 primary8 및 notebook32 budget 확인은 그동안 보류합니다.

현재 결정: [Primary 8 및 학습 예산 확인](docs/primary8-budget-decision.md).
Traffic은 공통 40GB 자원 제약으로 primary에서 제외했으며 80GB 재시도는 필수가 아닙니다.
다음 실행은 notebook `32_budget_confirmation.ipynb`의 모델별 2개 validation-only 조건입니다.

승인된 데이터 권고안 적용과 신규 4개 데이터셋의 validation-only GPU 실행 절차는
[본 실험 전 준비 상태](docs/pre-experiment-readiness.md)에 정리했습니다.
해당 문서는 9개 후보 준비 당시 이력이며, 현재 primary는 위 결정에 따른 8개입니다.

소스, YAML, 소용량 결과·요약·그림·manifest와 문서만 Git으로 관리합니다. 데이터셋, 모델 가중치, 캐시, 대용량 예측 배열과 원본 로그는 저장소에 추적하지 않습니다.
