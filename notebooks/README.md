# Notebooks

모델별 격리 Colab Pro+ compatibility probe를 제공합니다.

- `10_ttm_compatibility.ipynb`: 고정 TTM R3 synthetic/ETTh1 validation inference와 1-step full fine-tuning smoke
- `11_moirai_compatibility.ipynb`: 고정 MOIRAI 2 synthetic/ETTh1 validation inference smoke
- `12_moirai1_compatibility.ipynb`: 고정 MOIRAI 1.1 다변량 네 horizon, 공식 full fine-tuning 및 CUDA memory gate

MOIRAI 2 notebook은 고정 공식 코드에서 완성된 full fine-tuning API를 찾지 못했으므로 학습을 실행하지 않습니다. 두 notebook 모두 test split을 사용하지 않으며 cache와 weight는 Git에 저장하지 않습니다.
