# Traffic를 유지하기 위한 Colab 자원 기반 검증

최신 결과: [Traffic 일반 FP32 80GB 복구 통과](traffic-recovery-result.md).
Traffic을 포함한 primary9로 복귀하며, 아래 pending/예산 보류는 준비 당시 이력이다.
CPU offload와 BF16은 실행하지 않았다. 다음 실행은 notebook32 budget confirmation이다.

## 진행 상태 저장 오류 수정

최초 준비 commit `ef75e1ef74ef1a8e69d505afde4922ae2df08272`의 probe에는
running JSON 생성 후 진행 상태 갱신 시 `overwrite`를 허용하지 않는 버그가 있었다.
이 때문에 `DuplicateResultError`가 발생했고 마지막 실패 기록도 저장되지 않았다.
HF unauthenticated 경고나 GPU OOM을 이 오류의 원인으로 해석하지 않는다.
수정 버전은 **동일 identity의 running 기록만** 원자적으로 갱신하며, terminal 결과와
다른 identity는 덮어쓰지 않는다. cleanup 오류도 원래 오류를 가리지 않도록 보존한다.
이전 JSON/로그는 삭제하지 않고 새 실행 commit의 폴더에서 다시 확인한다.

## 현재 결정

Traffic의 최종 제외 결정을 철회한다. **기존 8개 + Traffic conditional**이며,
Traffic이 아직 통과한 것으로 표시하지 않는다. 이전 primary8/40GB OOM/80GB 미배정
기록은 역사적 증거로 보존한다. notebook 32의 budget confirmation은 이 검토까지 보류한다.
GPU 이름은 제한하지 않는다. 실제 배정된 CUDA GPU, VRAM, CPU 가용 RAM, BF16 지원,
driver 및 설치된 torch/CUDA를 기록한다. Pro+ 가입이 특정 GPU/VRAM 배정을 보장한다고
가정하지 않으며, Colab에서 보이는 선택지와 실제 배정 결과가 기준이다.

## 공식 지원 조사와 선택

고정 Uni2TS `cfd46d4510ed8896f263116f32928eede05b0a75`의 MoiraiModule,
MoiraiFinetune, TransformerEncoder 구현에서 gradient checkpointing을 활성화하는
공개 설정을 찾지 못했다. 따라서 vendor layer를 monkey-patch하거나 별도 encoder로
교체하지 않는다. 기존 공식 attention도 그대로 둔다.

먼저 PyTorch 공식 `torch.autograd.graph.save_on_cpu(pin_memory=False)`를 사용해
backward에 필요한 저장 tensor를 CPU로 이동하는 경로를 검증한다. FP32 모델/공식 loss와
gradient 계산을 유지하며 host RAM과 전송 시간을 추가로 사용한다.
[공식 API 설명](https://docs.pytorch.org/docs/2.9/autograd.html#torch.autograd.graph.save_on_cpu).
이는 새 PyTorch로 업그레이드한다는 뜻이 아니다. 실제 고정 torch 2.4.1 설치에서도
API 존재/문서를 로컬 확인했다. GPU 동작은 아직 미검증이다.

CPU offload는 forward의 임시 attention 행렬 자체를 없애지 않는다. CPU RAM 부족이나
GPU 임시 메모리 OOM이 여전히 가능하다. **가용 host RAM 64 GiB**는 이번 안전상 진입
기준일 뿐 필요량 실측/충분조건이 아니다. high-RAM runtime을 선택할 수 있다면 사용한다.
실행 중 host RSS는 단계 종료 시 기록하며 host peak로 해석하지 않는다.

## 검사 순서

모든 mode는 원 pretrained에서 독립 시작한다. ETTh1은 7채널 참조 검증이며 Traffic의
862채널을 축소하는 우회가 아니다. 각 mode에서 train window 1개로 2회 optimizer step,
validation window 1개로 예측/복원을 검증한다. 각 profile은 별도 subprocess다.

1. ETTh1/H720 FP32 참조 실행: 설치/API/정상 학습 환경 확인.
2. VRAM 70 GiB 이상인 실제 장치라면 Traffic/H720 일반 FP32를 먼저 확인.
   이 값은 큰 VRAM에서 기존 정밀도를 우선 시도하는 기준이며 GPU 이름/80GB gate가 아니다.
3. 일반 FP32가 미검증/메모리 실패이면 FP32 + CPU saved-tensor offload를
   ETTh1 → Traffic 순으로 확인. 가용 host RAM 미달은 not_run으로 남긴다.
4. 아직 미통과이고 `torch.cuda.is_bf16_supported()`라면 BF16 autocast를
   ETTh1 → Traffic 순으로 확인. FP32 parameter, full optimizer, GradScaler 없음.
5. 그래도 미통과이면 BF16 + CPU offload를 동일 순서로 확인한다.

FP16은 자동 대안으로 사용하지 않는다. 메모리 이외의 오류나 체크 실패는 중단한다.
성공한 후보를 찾으면 후순위 mode는 실행하지 않는다. 최대 8개 소규모 probe이며,
전체 데이터셋/비율 실험 또는 LR search가 아니다. GPU 브랜드가 H100/A100/L4/T4 등
무엇이든 위 실제 기능/자원 조건으로 분기하며 작은 GPU의 성공을 약속하지 않는다.

공통 고정: 동일 model revision/code, context512/H720, patch64, 100 samples/median,
batch1, 채널 공동 입력, native scaling, official MoiraiFinetune/PackedNLLLoss,
동일 train/validation window ID. test는 사용하지 않는다. 채널 삭제/분할, recursive
forecasting, 모델 교체, PEFT 또는 자체 loss는 없다.

검사 항목: CUDA 입력/parameter, finite shape, seed repeat, Zero-Shot no-update,
finite loss/gradient, full optimizer coverage, 실제 update, 2 steps, 학습 이후 validation,
새 model/optimizer의 checkpoint/RNG/parameter hash/prediction 복원.
CUDA synchronize/reset으로 step별 GPU allocated/reserved와 wall time을 기록한다.
첫 inference는 warm-up 없는 cold shape 측정이며 성능 benchmark로 일반화하지 않는다.
BF16은 ETTh1에서 같은 pretrained의 FP32 대비 prediction 및 공식 validation loss
오차도 기록한다. 이 작은 참조가 Traffic의 수치 동등성을 증명하지는 않는다.

## 사용자 실행

`notebooks/33_traffic_memory_probe.ipynb`를 **새 Colab GPU 세션**에서 연다.
가능한 GPU 중 VRAM이 큰 장치를 우선하되 A100을 강제하지 않는다. 가능하면 high-RAM도 켠다.
Python3.11/3.12, 기존 고정 MOIRAI 설치를 사용하고 실행 commit 전체 SHA를 입력한다.
위에서부터 셀을 실행하고 Drive 연결은 사용자가 승인한다.
자원/데이터 fingerprint 확인 후 `PROBE_TRAFFIC`을 입력한다.

결과는 Drive `tsfm-traffic-recovery/<commit>/moirai1/`에 조건마다 원자적 JSON으로 저장한다.
완료/실패 파일 모두 재사용하여 덮어쓰지 않는다. GPU/RAM을 바꿔 다시 시도하려면
Drive 셀의 OUT에 `/retry-2` 같은 **새 하위 경로**를 지정한다. 이전 JSON/로그는 보존한다.
runtime 강제 종료로 status=running이면 성공이 아니며 마지막 stage 기록을 회수한다.
중간 셀이 멈춰도 마지막 export 셀은 따로 실행할 수 있다.
`traffic-memory-probes.zip`을 반환한다. 모델/optimizer checkpoint는 일시 디스크에서
복원 검사 후 정리하고 Git/ZIP에 포함하지 않는다. 로그는 Drive에 남으며 ZIP은 JSON만 포함한다.

## 결과 이후

통과는 memory compatibility 후보일 뿐 자동 primary 복귀/precision 확정이 아니다.
FP32 offload가 통과하면 동일 계산 경로의 수치 검증과 장기 학습 비용을 확인한다.
BF16만 통과하면 MOIRAI 전체 데이터셋/학습비율에 적용할 모델 내 공통 precision 정책과
Zero-Shot/Few-Shot 동일 경로를 별도 검증한 후 정한다. Traffic 한 조건만 몰래 BF16으로
본 실험에 섞지 않는다. TTM과 MOIRAI precision이 다르면 명시한다.
어느 후보도 불가하면 H96/192/336 supplementary 또는 자원 제한 제외를 다시 검토한다.
지금은 installation_pending / pending_gpu, protocol_frozen=false, 본 실험 금지다.
