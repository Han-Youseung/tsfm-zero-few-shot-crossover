# Model environments

Use one fresh environment per model. `ttm.txt` and `moirai.txt` intentionally pin the
official source commit while preserving the dependency bounds declared at that commit.
They are reproducible install inputs. `ttm-resolved.txt` and `moirai-resolved.txt` are
the actual Python 3.12.14 Windows CPU resolution snapshots from this phase; local
file URLs were replaced by the same immutable official Git commits. Each Colab run
must also capture its own Linux/CUDA snapshot before a candidate is frozen.

The two files are not co-installable: the pinned TTM source requires Torch 2.10–2.11,
whereas the pinned uni2ts source requires Torch 2.1–2.4 and NumPy 1.26.
