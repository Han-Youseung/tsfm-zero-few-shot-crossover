# Traffic FP32 복구 결과 및 primary 9 복귀

## 검증된 결과

반환 archive `traffic-memory-probes.zip`의 execution commit은
`a4386f2cdceb1e552cd73d0cd3c89fa31228fd6c`이다. 기존 ef75e1e 미완료 archive와 구분한다.
ETTh1 FP32 참조와 Traffic FP32 두 조건 모두 완료됐다. importer는 ZIP CRC/경로/크기,
정확한 identity, model/source revision, CUDA/FP32, required checks, 실제 loss/step,
원본 fingerprint 및 재생성한 train/validation window/sampling을 대조했다.
소용량 결과는 `results/manifests/pilot/traffic_recovery_review.json`에 있으며
원본 archive와 이전 40GB OOM 결과를 덮어쓰지 않는다.

- GPU: NVIDIA A100-SXM4-80GB, 실제 79.250732 GiB.
- torch 2.4.1+cu121 / CUDA 12.1, driver 580.82.07.
- context512, H720, 공동 862채널, patch64, 100 samples/median, FP32 batch1.
- CPU offload 없음, autocast 없음, BF16/offload 후보는 **not_run**.
- official MoiraiFinetune/PackedNLLLoss, full parameter coverage, 2 optimizer steps.
- finite prediction/loss/gradient, no-update Zero-Shot, parameter update, 반복 예측과 복원 통과.
- checkpoint model/optimizer/step/RNG/profile 및 parameter hash 복원 검증.
- 복원 예측 최대 절대 오차 0, 비교 atol1e-5/rtol1e-4.
- test 미사용, protocol unfrozen, main experiment disabled.

| Traffic 측정 | 초 | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|
| 추론 (warm-up 없음) | 1.2528 | 21,260.78 | 23,640 |
| 학습 step 1 | 2.1074 | 69,923.36 | 71,366 |
| 학습 step 2 | 2.0365 | 70,029.26 | 71,366 |

약 68.39 GiB allocated가 필요했던 실제 조건이다. 40GB 학습 OOM은 모델 기능 실패가
아니라 자원 제약으로 보존한다. 2-step 측정은 long-run peak나 전체 실행시간 보장이 아니다.
연속형 확률밀도의 NLL은 음수가 될 수 있으므로 기록된 음수 loss 자체를 실패로 보지 않는다.

## 선정 및 자원 정책

**Primary 9개**: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Solar, Weather, Tetouan, Traffic.
각 데이터셋의 모든 채널을 유지하고 context/horizon을 축소하지 않는다.
두 모델 × 9개 × 네 horizon = 72조건의 batch1 feasibility 증거가 마련됐다.
다만 MOIRAI/Traffic/H720 FP32 학습은 현재 **80GB급에서 검증된 조건**이다.
GPU 이름을 A100으로 강제하지 않되, 다른 GPU의 동등 실행 가능성을 확인 없이 주장하지 않는다.
본 실험 자원 배치에서 이 조건은 80GB급 장치 확보를 요구하고, 미배정 시 기다리거나 별도
검증된 경로를 사용해야 한다. 40GB에서 재시도/정밀도 변경을 자동 적용하지 않는다.
GPU별 timing은 별도 집계하며 서로 다른 장치 시간으로 모델 속도 우위를 주장하지 않는다.

이전 primary8 결정은 역사적 기록이다. 새 실행 overlay는 `prepared_primary9.json`이다.
PEMS08/Exchange/SAPFLUXNET 등 미해결 후보를 숫자 맞추기로 추가하지 않는다.
ETT frequency 변형을 독립 도메인으로 세지 않는 한계도 유지한다.

## 다음 단계: budget confirmation 재개

notebook 33은 더 실행할 필요가 없다. **notebook 32를 새 세션에서 실행**한다.
TTM과 MOIRAI는 별도 세션이며, 실행 commit은 이번 결과 반영/primary9 준비 commit이다.
실험 대상은 여전히 ETTh1/Electricity, H96, 5%, 모델당 2개뿐이다.
Traffic 또는 기존 feasibility 전체를 다시 실행하지 않는다.

TTM LR1e-4, MOIRAI LR5e-6, FP32 batch1, 최대1000step,
100step마다 validation64window 평가, 3회 연속 비개선 시 정지한다.
이 상한은 수렴/최종 학습 예산이 아니라 제한된 validation 확인이다.
이 대표 조건에는 기존 40GB 지원 증거가 있으므로 80GB를 강제하지 않는다.
고정 공식 패키지와 CUDA 호환 설치가 실제 통과해야 한다.

반환: `ttm-budget-confirmation.zip`, `moirai1-budget-confirmation.zip`.
반환 후 결과 identity와 best/stopping step, 개선 추이, 시간/메모리를 검토해 최종 예산과
분석 프로토콜을 정한다. 본 실험은 자동 실행하지 않는다.
전체 설계 산술은 9×2×4×9(Zero-Shot+8 rates)×3 seeds = 1,944조건,
이 중 Few-Shot 학습 1,728회이다. 아직 실행하거나 소요시간을 확정하지 않았다.
