# 신규 데이터 GPU 결과와 다음 gate

후속 결정: 사용자가 제외를 승인하여 [primary 8 구성](primary8-budget-decision.md)으로
진행한다. 아래 80GB 재시도는 과거 제안이며 더 이상 필수 작업이 아니다. 실패 증거는 보존한다.

실행 commit: `28e8151931560143aa0955f267f9d8e787549b89`.
원본 ZIP은 Git에 포함하지 않는다. `new_data_review`는 archive CRC/경로/크기,
실행 계획, model/code revision, CUDA/FP32, fingerprint, split, train sampling,
validation window 및 개별 attempt 파일을 검증한다. Weather의 eligibility는 로컬
원본 variant에서 causal context와 완전 target 필터를 다시 계산하여 대조한다.

## 판정

장치: 두 모델 모두 NVIDIA A100-SXM4-40GB (reported 40,441.375 MiB).
TTM torch 2.11.0+cu130 / MOIRAI torch 2.4.1+cu121. Driver 580.82.07.

| Dataset | TTM H96/192/336/720 | MOIRAI H96/192/336/720 |
|---|---|---|
| Solar | 모두 FP32 batch 1 추론/학습 통과 | 모두 통과 |
| Traffic | 모두 통과 | 96/192/336 통과; 720 추론 통과, 학습 OOM |
| Weather | 모두 통과 | 모두 통과 |
| Tetouan | 모두 통과 | 모두 통과 |

32개 조건의 결과가 반환됐지만 **양쪽 연산을 통과한 조건은 31개**이다.
완료(`completed`)는 batch ladder 처리가 끝났다는 뜻이며 전체 성공과 다르다.
TTM 실패 attempt 9개는 더 큰 batch에서 발생했다. MOIRAI 실패 attempt 7개에는
Traffic H720의 batch 1 학습 실패가 포함된다. 실패 기록은 요약 manifest에도 보존한다.
MOIRAI는 기존 100-sample 설정이다. AMP는 not_run이며 FP32를 임의 변경하지 않았다.

Traffic H720 학습 오류: 추가 6.64 GiB를 할당하려 할 때 가용 2.39 GiB;
PyTorch allocated 30.68 GiB, reserved-but-unallocated 5.89 GiB로 보고된 OOM이다.
이 메시지는 성공적인 training peak 측정이 아니며 80GB 성공을 보장하지 않는다.
batch를 1보다 줄이거나 gradient accumulation만 바꿔 해결할 수 있는 문제는 아니다.
모델 기능 부재나 provenance 문제로 해석하지 않는다.

## 데이터 구성 권고

기존 ETT4/Electricity 증거와 합치면 8개 데이터셋은 네 horizon의 두 모델
batch-1 feasibility 증거가 있다. **Traffic은 9번째 후보로 유지하되 H720 학습 gate가
미해결**이다. 임의 채널 삭제, 채널별 독립 예측, horizon 축소, AMP/PEFT 전환은 하지 않는다.
PEMS08/Exchange/SAPFLUXNET 후보의 기존 provenance 차단도 그대로 유지한다.
현재 후보를 더 늘리기 전에 Traffic의 실제 자원 제약을 분리하는 것이 우선이다.

## 다음 실행: Traffic H720, MOIRAI만 80GB에서 확인

새 MOIRAI 세션에서 같은 pinned notebook/commit을 사용한다. A100 80GB가 실제 배정된
경우를 우선하며, 다른 GPU에서 실행했다면 그대로 기록한다. 40GB 재배정을 80GB로
표시하면 안 된다. 기존 데이터/설치/계획 생성 셀까지 실행한 뒤 다음 셀을 추가한다.
현재 검증된 config/precision/model/data를 바꾸지 않고 output 경로만 분리한다.

```python
assert FAMILY == "moirai1"
subprocess.run([
    str(PY), "-c",
    "import torch; p=torch.cuda.get_device_properties(0); "
    "print(p.name,p.total_memory/2**30); assert p.total_memory >= 75*2**30"
], check=True)
OUT = PERSIST / COMMIT / "moirai1-traffic-h720-80gb-retry"
OUT.mkdir(parents=True, exist_ok=True)
BASE = [str(PY), "-m", "tsfm_crossover.experiments.pilot",
        "--prepared", str(PINNED), "--output", str(OUT),
        "--expected-commit", COMMIT]
subprocess.run(BASE, check=True)
plan = json.loads((OUT / "plan.json").read_text())
jobs = [r for r in plan["conditions"] if r["family"] == FAMILY
        and r["dataset"] == "Traffic" and r["horizon"] == 720
        and r["kind"] == "feasibility" and r["status"] == "planned"]
assert len(jobs) == 1
subprocess.run(BASE + ["--condition-id", jobs[0]["id"]], check=True)
result = json.loads((OUT / jobs[0]["id"] / "result.json").read_text())
print("FP32 batch 1 inference AND training:", result["fp32_batch1_supported"])
```

이후 다음 export 셀만 실행한다 (기존 16개 실행 셀은 건너뛴다).

```python
from google.colab import files
archive = Path("/content/moirai1-traffic-h720-80gb-retry.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".csv"} and p.stat().st_size < 2_000_000:
            z.write(p, p.relative_to(OUT))
files.download(str(archive))
```

32조건 importer는 이 단일 재시도 ZIP을 완전한 32조건 결과로 받지 않는다.
반환 후 동일 condition identity를 검증하고 다른 장치의 별도 attempt로 기록해야 한다.
40GB 실패 증거를 덮어쓰지 않는다. 80GB가 배정되지 않으면 실행을 중단하고 pending으로 둔다.

## 이후 순서

Traffic 재검증 → dataset/horizon 범위 검토 → validation-only 예산/설정 검토 →
최종 프로토콜 및 실행 규모 dry-run. 기존 200-step pilot은 수렴 보장이 아니므로
최종 step budget으로 자동 확정하지 않는다. 성능 순위/crossover는 계산하지 않았으며
`main_experiment_allowed=false`, `protocol_frozen=false`를 유지한다.
