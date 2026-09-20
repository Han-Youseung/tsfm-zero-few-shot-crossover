# TSFM Zero-Shot–Few-Shot Crossover

시계열 데이터 특성과 대상 데이터 학습량에 따라 Time-Series Foundation Model(TSFM)의 Few-Shot Fine-Tuning이 Zero-Shot 성능을 넘어서는 **전환 구간**을 분석하는 연구 저장소입니다.

> 현재 단계: 모델 파일럿 전의 독립 실험 프레임워크 구축

## 연구 범위

- 모델 후보: `ibm-granite/granite-timeseries-ttm-r3`, `Salesforce/moirai-2.0-R-small`
- 데이터셋: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, PEMS08, Solar, Wind, Weather, AQShunyi, Exchange, ZafNoo, CzeLan
- 학습 비율: `0%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%`
- 예측 길이: `96, 192, 336, 720`
- 주 평가: 동일한 test window의 stride 1 rolling-origin MAE·MSE

실험 시작 시점의 최신 공식 TTM·MOIRAI를 파일럿으로 검증한 뒤 Hugging Face revision, 공식 코드 commit과 패키지 버전을 고정합니다. Context length, optimizer, learning rate, step budget, batch size와 fine-tuning 대상 파라미터는 아직 확정되지 않았으며 test 성능을 보고 선택하지 않습니다.

TSFM-Bench는 관련 선행연구로만 인용합니다. 그 결과·체크포인트·실험 조건을 재현하거나 직접 비교하지 않고, 코드와 프로토콜도 복사·번안하지 않습니다. 구현은 최신 TTM·MOIRAI의 공식 구현과 공개 API를 사용해 독립적으로 작성합니다.

## 설치

코어 프레임워크는 Python 3.11–3.13을 지원하고 GPU 또는 PyTorch 없이 실행할 수 있습니다.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

모델 의존성은 파일럿 후 모델별 optional dependency로 분리합니다. 현재 코어 환경은 실제 모델을 다운로드하지 않습니다.

## 구조

- `configs/`: dataset, model, experiment, protocol YAML
- `src/tsfm_crossover/config/`: strict 설정 schema, loader, ID 생성
- `src/tsfm_crossover/tracking/`: 결과 schema, 원자적 저장, 실행 환경 메타데이터
- `src/tsfm_crossover/utils/`: seed와 재현성 유틸리티
- `src/tsfm_crossover/{data,models,experiments,evaluation,analysis}/`: 다음 단계의 독립 구현 경계
- `scripts/`: 재실행 가능한 CLI 진입점
- `notebooks/`: 파일럿 후 추가할 Colab 노트북
- `tests/`: CPU 단위·통합 테스트
- `results/`: 소용량 JSON/CSV, summary, figure, manifest
- `data/`, `checkpoints/`, `logs/`: Git에 올리지 않는 로컬 산출물

세부 연구 프로토콜은 [docs/research-plan.md](docs/research-plan.md), 출처와 라이선스는 [docs/licenses-and-sources.md](docs/licenses-and-sources.md)에서 관리합니다.

## 저장 원칙

소스, YAML, 소용량 결과·요약·그림·manifest와 문서만 Git으로 관리합니다. 데이터셋, 모델 가중치, 캐시, 대용량 예측 배열과 원본 로그는 저장소에 추적하지 않습니다.
