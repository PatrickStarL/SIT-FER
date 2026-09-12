# SIT-FER Development TODOs

## High Priority

### Core Functionality
- [x] Implement RAF-DB dataset loader in `src/sit_fer/data/raf_dataset.py`
- [x] Add data augmentation module (RandAugment, migrated to `src/sit_fer/data/randaugment.py`)
- [ ] **Fix `ResNet18` feature normalization / pretrained weights mismatch** — the legacy backbone loaded MS-Celeb-pretrained weights and L2-normalized features before the classifier; `src/sit_fer/models/resnet.py` currently uses plain ImageNet weights and does not normalize. Since text/instance similarity in `Trainer` assumes normalized embeddings, this should be resolved before running real training. See `legacy/README.md` for details.
- [ ] Implement other dataset loaders (FERPlus, AffectNet)
- [ ] Implement test/inference script
- [ ] Add model export (ONNX, TorchScript)

### Training Infrastructure
- [ ] Add learning rate scheduling
- [ ] Implement gradient accumulation
- [ ] Add mixed precision training (AMP)
- [ ] Add distributed training support (DDP)
- [ ] Implement early stopping

### Monitoring & Logging
- [ ] Expand TensorBoard logging (gradients, weights)
- [ ] Add Weights & Biases integration
- [ ] Implement confusion matrix visualization
- [ ] Add per-class accuracy tracking

## Medium Priority

### Code Quality
- [x] Write unit tests for core modules (config, models, losses, instance bank, data splitting, public API) — see `tests/`
- [ ] Add integration tests (end-to-end training loop on a tiny synthetic dataset)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Add pre-commit hooks (black, flake8, mypy)
- [ ] Type checking with mypy

### Documentation
- [ ] API documentation with Sphinx
- [ ] Training tutorial notebook
- [ ] Inference tutorial notebook
- [ ] Architecture diagram
- [ ] Add docstrings to all functions

### Features
- [ ] Implement ensemble methods
- [ ] Add attention visualization
- [ ] Implement grad-CAM for interpretability
- [ ] Add model pruning/quantization
- [ ] Support for custom backbones (ViT, EfficientNet)

## Low Priority

### Deployment
- [ ] Create Gradio demo
- [ ] Build REST API with FastAPI
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Create model card
- [ ] Benchmark on different hardware

### Experiments
- [ ] Hyperparameter search (Optuna)
- [ ] Ablation studies
- [ ] Cross-dataset evaluation
- [ ] Few-shot learning experiments

### Community
- [ ] Create contributing guidelines
- [ ] Add code of conduct
- [ ] Set up issue templates
- [ ] Create pull request template
- [ ] Add changelog

## Completed ✅
- [x] Refactor code into modular structure
- [x] Create configuration system
- [x] Implement modular losses
- [x] Create utility functions
- [x] Add logging system
- [x] Write improved README
- [x] Add requirements.txt
- [x] Create setup.py
- [x] Add .gitignore
- [x] Create Dockerfile
- [x] Move `Trainer` into `src/sit_fer/core/trainer.py` (was referenced but missing, breaking `import sit_fer`)
- [x] Migrate RAF-DB dataset + RandAugment into `src/sit_fer/data/`
- [x] Wire `scripts/train.py` into a real CLI entrypoint
- [x] Fix `.gitignore` bug where an unscoped `data/` rule was silently excluding `src/sit_fer/data/` from git
- [x] Untrack committed `.pyc` files from the original repo
- [x] Archive superseded root-level files (`main.py`, `losses.py`, `test3.py`, `text.py`, `text2.py`, old `models/`, `dataset/`, `utils/`) into `legacy/`
- [x] Consolidate `README.md`/`README_NEW.md` into a single README; move summary docs into `docs/`
- [x] Add `LICENSE` file (MIT, matching the badge/claim already in README)
- [x] Fix `PartialLoss.confidence_update` hardcoding `.cuda()` — found via the new test suite; it crashed on any CPU-only machine. Now uses `.to(temp_un_conf.device)`.
