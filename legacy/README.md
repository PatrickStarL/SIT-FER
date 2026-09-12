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

## Resolved: MS-Celeb-1M pretrained backbone

`legacy/models/backbone.py`'s `ResNet_18` loaded a **face-recognition-pretrained** checkpoint (`resnet18_msceleb.pth`, MS-Celeb-1M pretraining) rather than plain ImageNet weights, and L2-normalized features before the classifier. `src/sit_fer/models/resnet.py::ResNet18` now replicates this: pass a checkpoint path via `model.pretrained_path` in `configs/base.yaml` (or the `pretrained_path=` constructor arg) and it loads and L2-normalizes features the same way. When `pretrained_path` is not set, it falls back to ImageNet-pretrained `torchvision` weights instead.

**This repo still does not include the `resnet18_msceleb.pth` file itself** — it was never distributed with the original SIT-FER repo either. You need to source it yourself (it's commonly shared across RAF-DB/FER papers that build on Self-Cure-Network-style backbones) and point `model.pretrained_path` at your local copy.
