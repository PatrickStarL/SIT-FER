# Legacy Code (Archived)

This directory contains the original, pre-refactor implementation, kept for historical reference only. **None of it is imported or used by the current codebase.**

The functionality here has been superseded by `src/sit_fer/`:

| Legacy file | Replaced by |
|---|---|
| `main.py` | `scripts/train.py` + `src/sit_fer/core/trainer.py` |
| `losses.py` | `src/sit_fer/losses/` |
| `models/backbone.py`, `models/model.py` | `src/sit_fer/models/resnet.py`, `text_encoder.py` |
| `dataset/raf.py`, `dataset/randaugment.py` | `src/sit_fer/data/` |
| `utils/eval.py`, `utils/misc.py` | `src/sit_fer/utils/helpers.py` |
| `utils/__init__.py`, `utils/logger.py` | `setup_logger` + TensorBoard in `src/sit_fer/utils/` |
| `test3.py`, `text.py`, `text2.py` | Scratch/debug scripts; logic now lives in `InstanceBank` and `TextEncoder` |
| `utils/generate_labelset.py` | Unused/dead code in the original repo (imported but never called) |
| `utils/label2txt.py`, `utils/rename.py` | One-off personal scripts with hard-coded local paths; not reusable |

## Known caveat carried over from the original code

`legacy/models/backbone.py`'s `ResNet_18` loads a **face-recognition-pretrained** checkpoint (`resnet18_msceleb.pth`, MS-Celeb-1M pretraining) rather than plain ImageNet weights, and L2-normalizes features before the classifier. The refactored `src/sit_fer/models/resnet.py::ResNet18` currently uses ImageNet-pretrained `torchvision` weights and does **not** normalize features by default. Since the three-level fusion (text/instance similarity) assumes normalized embeddings, this is worth revisiting before running real training — see `TODO.md`.
