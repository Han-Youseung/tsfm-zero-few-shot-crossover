# Scripts

재실행 가능한 CLI 진입점입니다.

- `probe_ttm_cpu.py`: 고정 TTM 512/96 CPU inference, ETTh1 validation, 1-step fine-tuning 및 state round-trip probe
- `probe_moirai_cpu.py`: 고정 MOIRAI 2 CPU inference probe; `--channels 7`의 공식 multi-token 실패와 `--channels 1` 성공을 분리 확인

두 스크립트는 수동 compatibility 검사이며 기본 pytest에서 실행되거나 가중치를 자동 다운로드하지 않습니다.
