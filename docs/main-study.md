# 본 실험 실행: primary9 bounded-compute v1

사용자의 본 실험 실행 승인을 반영한 설정은 `configs/study/main.yaml`이다.
과거 `preexperiment.yaml`과 모든 CPU/GPU/pilot manifest는 당시 상태로 보존한다.
`protocol_frozen=true`는 이 **주 분석 v1의 선택 규칙을 고정**한다는 뜻이며,
GPU 전체 실행 성공, 수렴, crossover 관측 또는 연구 전체 완료를 뜻하지 않는다.
로컬 CUDA는 사용할 수 없다. 새 main notebook의 Colab 설치와 실제 main GPU 실행은 pending이다.

## 고정된 범위

- Primary9: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Solar, Weather, Tetouan, Traffic.
- TTM-R3와 MOIRAI 1.1-R-small의 기존 checkpoint/revision/code SHA를 그대로 사용한다.
- Context512, H96/192/336/720, seeds1729/2718/31415.
- Zero-Shot 및 nested rates0.5/1/2/5/10/20/50/100%.
- 60/20/20, model-native scaling, 외부 scaler 없음. train population std는 **오차 정규화에만** 사용한다.
- 동일 모델의 모든 dataset/rate에 FP32, batch1, accumulation1, full-parameter,
  AdamW, 최대1000steps, eval100steps, patience3evals, clipping 없음, scheduler 없음을 고정한다.
- TTM LR1e-4/decay0.01, MOIRAI LR5e-6/decay0.1. 공식 loss 경로는 모델마다 다르다.
- TTM prediction_outputs; MOIRAI100 samples의 `torch.median` (짝수 표본에서는 낮은 중앙값).
- Python3.12 minor series, 모델별 `requirements/*-main.txt`는 검증된 GPU budget 환경의
  실제 패키지 버전이다. CPU snapshot이 아니다. pip/setuptools/wheel은 bootstrap 도구로
  고정 검증에서 제외하되 실제 버전은 기록한다. Python patch/driver/GPU도 실제 값을 기록한다.
- strict CUDA median determinism 미지원 때문에 deterministic algorithms+warn_only,
  cudnn benchmark=False, TF32=False를 사용한다. 장치 간 bitwise 동등성을 주장하지 않는다.

계획은 Zero-Shot216 + Few-Shot1728 = **1944조건**, 최대1728000 optimizer steps다.
100%는 전체 적격 train pool 접근이며, 최대1000회 노출로 전체 pool 방문을 보장하지 않는다.
실제 step/unique visited/equivalent epochs/시간/메모리를 각각 기록한다.

## 누수 방지와 최종 평가

`main_study`는 기존 train/validation loader와 학습 루프를 재사용한다.
하나의 전체 rate sampling manifest를 dataset–horizon–seed별로 공유하고 모델별로 다시 뽑지 않는다.
manifest의 큰 window-ID 목록은 Drive에 보존하고 작은 결과에는 경로·hash·개수·coverage를 남긴다.
결측 input은 causal ffill, target은 보간하지 않는다. 두 모델 모두 완전한 train target
window만 후보로 쓰며 test/validation metric은 원본 관측 mask로 제외한다.

Fine-Tuning은 validation64개 시간층화 window에서 최저 normalized MAE를 얻은 **학습 후**
checkpoint를 선택한다. 동률이면 더 이른 checkpoint를 유지한다. Zero-Shot으로 교체하지 않는다.
새 adapter/optimizer에 저장 상태를 복원하고 `selection.json`을 먼저 불변 저장한 다음
test 값을 연다. 선택 후에는 optimizer 메모리를 해제한다.

별도 `predict_test`는 target 없는 test context만 허용한다. 기존 `predict`, `train_step`,
`validation_step`의 test 차단은 유지한다. test stride1 전체 rolling-origin을 평가하며
새로 관측된 test 값은 이후 context에만 들어간다. 입력은 모델/원 단위, 주 지표는
channel-macro train-std normalized MAE이고 raw MAE/MSE와 normalized MSE도 보고한다.
train-constant 채널은 모델과 raw metric에 유지하고 normalized metric에서만 제외한다.
관측값 없는 test target 또는 leading-missing context 때문에 제외한 window 수를 기록한다.

## 재개와 저장

- 학습: 기존 sampler/optimizer/RNG/step/loop를 checkpoint25steps 또는 평가 시 원자적으로 저장한다.
- 평가: 100개 test window마다 cursor와 채널별 합계/count를 하나의 JSON에 원자적으로 저장한다.
  마지막 저장 뒤 작업만 다시 계산하며 저장된 window는 이중 집계하지 않는다.
- `result.json`은 test가 끝난 뒤만 completed가 된다. `training.json`의 completed는 학습 단계만 뜻한다.
- selection/checkpoint hash, commit, protocol/config/data/sampling/window identity가 다르면 거부한다.
- 진행 중 조건의 GPU·driver·실제 패키지 환경이 바뀌면 자동 혼합하지 않는다. 기존 결과를
  보존하고 새 run directory에서 명시적으로 재시작할지 검토한다. 완료된 조건은 재실행하지 않는다.
- 비정상 세션 종료로 남은 `running.lock`은 이전 세션/프로세스 종료를 확인한 뒤 해당 lock만
  수동 제거한다. checkpoint/selection/test-progress/failure는 삭제하지 않는다.
- checkpoint와 큰 sampling manifest는 Drive에 남는다. 임시 `/content`만으로는 복구되지 않는다.
- 모든 실패는 별도 JSON으로 보존하며 자동 채널 분할, 모델 교체, AMP 전환은 하지 않는다.

## GPU 자원

A100만 요구하지 않는다. 고정한 CUDA 소프트웨어를 지원하는 실제 Colab Pro+ GPU를 사용한다.
다만 MOIRAI/Traffic/H720 **학습**은 A10080GB에서 약68.39GiB allocated peak를 확인했고
40GB OOM을 보존했으므로 총VRAM>=75GiB가 아니면 `pending_80gb_class_gpu`로 대기한다.
GPU 이름 선택과 실제 배정 GPU는 다를 수 있다. 나머지 조건도 작은 GPU에서 fit을 보장하지 않는다.
GPU가 다르면 wall time을 모델 속도 순위로 직접 비교하지 않는다.

## Colab 실행 순서

노트북: `notebooks/40_main_study.ipynb`.

1. 런타임에서 Python3.12와 사용 가능한 CUDA GPU를 선택한다.
2. `runtime`: 실제 GPU 확인, family=`ttm` 또는 `moirai1`, 공개된 main 실행 commit40자리 입력.
3. `repository`: 해당 commit checkout. 실행 도중 pull/업데이트 금지.
4. `drive`: 사용자가 Drive 인증. 결과는 `MyDrive/tsfm-main-study/<commit>/<family>`.
5. `installation`: 모델 전용 venv 설치, pip check, 실제 subprocess Python/CUDA 확인.
   설치 실패를 무시하거나 최신 버전으로 임의 바꾸지 않는다.
6. `plan`: 첫 실행은 ETTh1 / H96 / seed1729 권장. 해당 조합의 Zero-Shot+8rates를 표시한다.
   이 순서는 환경 확인을 위한 실행 순서일 뿐 결과로 설정을 바꾸는 pilot이 아니다.
7. `data`: 기존 공식 출처/변환 recipe로 해당 dataset만 준비하고 고정 fingerprint 대조.
8. `run`: `RUN_MAIN` 입력. 최대9조건, 세션6시간 이후 새 조건 시작을 중단한다.
   이미 시작한 조건을 시간 때문에 강제 중단하지 않는다. 실행 셀 재실행은 skip/resume한다.
9. `export`: 실패했더라도 이 셀로 작은 JSON ZIP을 내려받고 Drive exports 사본도 보존한다.
10. 나머지 seed/horizon/dataset으로 6–9번 반복. 다른 모델은 **새 세션**에서 1번부터 진행한다.

정상 출력은 `test windows n/N`, 마지막 `completed` 또는 `skipped_completed`이다.
pending은 성공이 아니다. 오류가 나면 다음 조건을 임의 실행하지 말고 ZIP과 오류를 확인한다.
Drive 잔여 용량도 확인한다. best/last optimizer checkpoint는 조건 수에 따라 상당히 누적된다.
사용자가 검증 전 checkpoint를 삭제하도록 자동 안내하거나 자동 정리하지 않는다.

## Colab model-cache recovery

Model weights and HF/Xet caches stay on local `/content` storage, while results,
optimizer checkpoints and public dataset files remain on Drive. The notebook
sets `HF_HOME` and `HF_XET_CACHE` before starting model subprocesses. After runtime
loss, download the same immutable model revision again; do not change the model
or the execution commit to repair a cache. The local weight cache is disposable,
but experiment checkpoints and evidence are not.

On 2026-09-27, the TTM H96 cached `model.safetensors` resolved to a 79-byte
relative-path string instead of the pinned 21,441,800-byte tensor file, causing
`SafetensorError: header too large`. This was a cache/storage failure, not an OOM
or fine-tuning loss failure. Fresh local download matched the manifest's SHA-256
`8372cf7a0be542fd56b047b190e84bb5eea1cb384d7e60aa0ec32080b9c7bd08`, parsed all
742 tensors, and matched the completed Zero-Shot parameter hash. That verification
did not evaluate test performance or change any research setting.

For an existing Drive-backed model cache, stop before training. Preserve the old
cache and failure JSON; do not recursively delete either. Download into a new
local cache, verify config and weight size/hash against the pinned manifest and
parse the tensors. If a completed Zero-Shot exists, also compare the loaded
pretrained parameter hash with its locked selection. Only then redirect the local
`.cache` link, preserving the old link/target, and record the recovery separately.
Keep `OUT`, the execution commit and the fixed protocol unchanged. Re-running the
original runner validates and skips completed conditions. The notebook guard
blocks an old Drive-backed cache for explicit review rather than silently deleting
or replacing it. A repaired notebook can be opened at a newer revision while
entering the original execution SHA; never pull new runner code into an active run.

## CPU 회수·분석

원본 ZIP을 그대로 보존한 채 다음을 실행한다. 기존 파일을 덮어쓰지 않도록 새 output 경로를 사용한다.

```bash
python -m tsfm_crossover.experiments.main_review <shard1.zip> <shard2.zip> \
  --expected-commit <MAIN_EXECUTION_SHA> --output results/manifests/study/review-001.json
```

검증은 ZIP 안전성/CRC, commit/model revision/code SHA, 패키지, fingerprint,
sampling/window/mask 및 validation stopping을 확인한다. GPU 계산을 CPU로 재실행하거나
예측 배열에서 metric을 재계산했다는 주장은 하지 않는다. 계획 밖·과거 pilot·CPU 결과를 합치지 않는다.
여러 ZIP을 함께 주면 동일 completed 조건은 내용이 정확히 같을 때만 중복 제거한다.
sampling manifest는 코드·train 데이터로 재구성해 hash를 확인한다. 결과 ZIP에는 포함하지 않는다.

각 model–dataset–horizon의 3seeds×9rates가 모두 있어야 crossover를 계산한다.
주 규칙은 paired seed 평균의 `Zero-Shot NMAE − Few-Shot NMAE > 0`이다.
최초 양수 구간, 이후 모든 관측 비율에서 양수인 지속 구간, 미관측/비단조를 구분한다.
3개 seed 모두 양수인 구간도 병기한다. seed min/max는 **기술통계**이며 신뢰구간·유의성 검정이 아니다.
관측 grid의 구간이지 연속 비율의 실제 함수가 그 사이에서 단조라는 보장은 없다.
시간 block bootstrap은 현재 미구현이며 iid window bootstrap은 사용하지 않는다.

전체36그룹이 모이면 ETT hourly/minutely를 같은 transformer 출처로 먼저 평균한 뒤
7개 출처 그룹을 동등 가중한 relative-improvement를 보조 집계한다. 원 단위 이종 dataset
오차를 무리하게 합치지 않는다. train-only descriptor는 결측률, 채널 수, 상수 채널,
lag1 correlation, 표준편차로 정규화한 선형 추세다. 계절성/인과관계는 추정했다고 주장하지 않는다.
9개 dataset(7개 출처 그룹)의 특성–crossover 관계는 탐색적 사례 분석이며 일반화 한계가 크다.

## 검증 범위

기본 CPU 테스트와 toy tensor 통합 테스트는 누수 차단, full-parameter 학습 루프,
best 복원 후 test 연결, 중단·재개·completed skip 및 결과 위변경 탐지를 확인한다.
toy는 테스트 전용이며 실제 GPU 근거로 저장하지 않는다. 기존 모델 GPU 근거와 새 main runner의
실제 Colab 실행 여부는 구분한다. 새 GPU 결과를 받기 전 main 성능/crossover 숫자는 없다.
