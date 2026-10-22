# Leafy AI

[![CI](https://github.com/Leafy-AI-90/AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Leafy-AI-90/AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Open-source model foundations for paired RGB images, 18-band AS7265x reflectance,
and 660/710/940 nm transmission measurements of Choy Sum.

PlantVillage and PlantDoc source manifests are pinned. Download their archives
locally before image pretraining; images are excluded from Git. See
[dataset setup and licensing](docs/datasets.md).

**Status:** model installation and research scaffolding. Image backbones have
ImageNet weights. No model has been trained or validated to diagnose plant stress.
Real paired measurements and independently established labels are still required.

### Installed model families

| Review role | Implementation | State |
|---|---|---|
| B0 | M660 threshold | Binary low-chlorophyll comparator; threshold supplied from calibration |
| B1 | EfficientNet-B0 | Fold-local image fine-tuning runner from verified ImageNet weights |
| B2 | DINOv2 ViT-S/14 | Official SHA-256-pinned frozen features plus grouped logistic |
| B3 | PLS-DA, random forest, XGBoost, logistic | Fold-local spectral runner |
| B4 | Thermal-only logistic | Enabled for root-disease preset |
| B5 | PlantVillage-trained CNN | Requires licensed pretraining run and reviewed Choy Sum label mapping |
| Calibration | PLS regression | Trainable chlorophyll regression |
| F1 | Grouped out-of-fold logistic stacking | Trainable fusion component |
| F2 / proposed | Frozen image embeddings + fold-local PCA + measured features + XGBoost | Trainable fusion pipeline |
| F3 | Gated cross-attention on frozen image vector and measured features | Fold-local comparison runner |
| Phone | MobileNetV3-Large | Frozen image embeddings with proposed measured modalities; export follows in Phase 9 |

The document does not identify a specific PlantVillage-trained checkpoint.
ImageNet weights are not a substitute for one. No unverified third-party
diagnostic checkpoint is installed.

### Install

Python 3.11 to 3.13 on Linux, macOS or Windows. Install CPU PyTorch first to
avoid multi GB CUDA wheels (skip on macOS, where default wheels are CPU/MPS).

With `uv`:

```bash
uv venv --python 3.12
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
uv pip install -e ".[vision,image,stats,dev]"
```

With `pip`:

```bash
python -m venv .venv
# activate: source .venv/bin/activate (Linux/macOS) or .venv\Scripts\activate (Windows)
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
python -m pip install -e ".[vision,image,stats,dev]"
```

| Extra | Contents |
|---|---|
| core | numpy, scipy, scikit-learn, XGBoost, Optuna, pandas, pydantic, PyYAML, pandera, pyarrow |
| `[vision]` | PyTorch, torchvision, timm, Pillow, Grad-CAM |
| `[image]` | colour checker detection (colour-science, OpenCV), Pillow |
| `[stats]` | statsmodels, SHAP, matplotlib |
| `[export]` | phone export tools (Phase 9) |
| `[dev]` | pytest, pytest-cov, ruff, pre-commit |
| `[all]` | everything above |

Use editable install (`-e`): study presets load from `configs/` in repository.
`requirements-lock.txt` records original Windows Python 3.12 CPU environment;
`vendor/wheels` holds optional Grad-CAM wheel (`--find-links vendor/wheels`).

### Run checks

```bash
python scripts/download_models.py          # verified ImageNet and DINOv2 weights, about 250 MB
python -m pytest -q
python scripts/smoke_test.py               # or --skip-checkpoints without weights
```

### Study presets

Study design lives in YAML, not code. See [configs/README.md](configs/README.md).

```python
from leafy_ai.config import load_config

cfg = load_config("root_disease")   # or "xcc" (default), "salt_planb", or a YAML path
cfg.classes, cfg.leaf_positions, cfg.design.endpoint_day, cfg.sensors.thermal
```

### Ingest real data

Raw exports follow [data schema](docs/data-schema.md): one CSV or Parquet file per table.

```bash
python scripts/ingest.py --preset xcc --raw data/raw/cycle1 --out data/interim/cycle1
python scripts/process_images.py --interim data/interim/cycle1 --images-root data/raw/cycle1 --out data/interim/cycle1/images
```

Then build features (one row per plant/day; see [features](docs/features.md)):

```bash
python scripts/build_features.py --preset xcc --interim data/interim/cycle1
```

Ingest validates every table, checks plant and session links, writes read only
Parquet with SHA-256 manifest, and writes QC report to `reports/qc/cycle1/qc.html`.
QC flags rows and never drops them; exclusions follow `qc.exclude_flags` in preset.
Image step fits 3×3 colour correction per image from 24 patch checker and writes
leaf mask plus background removed variant.

For measured calibration plants, run the [H1 chlorophyll analysis](docs/calibration.md)
after ingestion and feature building. It selects PLS components in inner
plant-grouped folds and compares held-out predictions with M660 and red-edge
baselines. No H1 biological result is reported until real measurements exist.

Run plant-level classification with the [experiment guide](docs/experiments.md)
and a YAML in `configs/experiments/`. The runner writes held-out plant
probabilities, metrics, tuned parameters, model artifacts and input hashes to
ignored `runs/` directories. No real-data training run has been made.

`leafy_ai.CLASSES` keeps returning default `xcc` class list for backward compatibility.

### Use the models

```python
from leafy_ai.vision import load_backbone, image_transform
from leafy_ai.models import spectral_model, intermediate_fusion

# Five-class output layer is newly initialized and MUST be trained.
image_model = load_backbone("efficientnet_b0", "models/weights", num_classes=5)
preprocess = image_transform(training=True, size=384)
spectral = spectral_model("random_forest")
# X columns: 1280 frozen ImageNet embedding + 45 spectral + 3 positional.
fusion = intermediate_fusion(embedding_dim=1280, total_dim=1328)
```

For F2, call `embeddings(model, images)` with the model in `eval()` and
`torch.inference_mode()` when using frozen ImageNet features. Fit PCA and every
estimator only on training folds. If the encoder is fine-tuned, it must be trained
inside the same outer/inner split boundaries. SVM intentionally has no internal
probability calibration because its built-in CV is not grouped by plant.

Weights live under `models/weights/` and are excluded from Git. The versioned
`models/manifest.json` records exact upstream weight enums, URLs, sizes, and full
SHA-256 hashes. `download_models.py` verifies publisher hash prefixes and DINOv2's
full hash. Inference
loads only local verified checkpoints with `weights_only=True`.

### Research boundaries

- Default preset `xcc`: control, N deficiency, Fe deficiency, Xcc, N deficiency + Xcc.
- Primary endpoint: plant-level macro F1 on held-out cycles at preset `endpoint_day`
  (day 8 for `xcc`, day 10 for `root_disease`).
- Keep all days, leaves, repeats, and images of a plant together. Use globally
  unique IDs, including cycle where local plant numbers are reused.
- `grouped_folds`, `cycle_folds`, and the Phase 5 runner enforce outer and inner
  split boundaries; Phase 6 adds plant-clustered statistical comparisons.
- No dataset images, private source PDF, laboratory protocol, or fabricated
  biological measurement is published by this setup.
- Synthetic smoke tests demonstrate software operation only.

See [design mapping](docs/design.md), [model sources and licenses](docs/models.md),
[verification](docs/verification.json), [data checks](docs/data-verification.json),
and [setup status](docs/setup-status.md).

### Licence

Code: MIT ([LICENSE](LICENSE)). Datasets (PlantVillage, PlantDoc and others) and
pretrained weights (ImageNet checkpoints) keep upstream licences and terms; MIT
does not relicense them. See [datasets](docs/datasets.md) and [models](docs/models.md).
Cite this software with [CITATION.cff](CITATION.cff).
