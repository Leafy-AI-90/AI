# AI

## Leafy AI

Open-source model foundations for paired RGB images, 18-band AS7265x reflectance,
and 660/710/940 nm transmission measurements of Choy Sum.

Local project: `C:\Users\Lenovo\Desktop\Leafy AI`.

PlantVillage and PlantDoc are downloaded locally and ready for image-model
experiments. See [dataset setup and training usage](docs/datasets.md). Their
images are excluded from Git; pinned sources and audit summaries are tracked.

**Status:** model installation and research scaffolding. Image backbones have
ImageNet weights. No model has been trained or validated to diagnose plant stress.
Real paired measurements and independently established labels are still required.

### Installed model families

| Review role | Implementation | State |
|---|---|---|
| B0 | M660 threshold | Binary low-chlorophyll comparator; threshold supplied from calibration |
| B1 | Logistic regression | Trainable, with fold-local scaling |
| B2 | PLS-DA, random forest, SVM, small MLP | Trainable spectral baselines |
| B3 | EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Large | Official ImageNet checkpoints downloaded locally |
| Calibration | PLS regression | Trainable chlorophyll regression |
| F1 | Grouped out-of-fold logistic stacking | Trainable fusion component |
| F2 / proposed | PCA image embeddings + spectral/positional features + XGBoost | Trainable fusion pipeline |
| F3 | Gated cross-attention head | Untrained comparison implementation |
| B4 | PlantVillage-trained CNN | Deferred: requires a documented PlantVillage training run and label mapping |

The document does not identify a specific B4 checkpoint. ImageNet weights are
not a substitute for PlantVillage-trained weights. No unverified third-party
diagnostic checkpoint is installed.

### Run locally

Python 3.11-3.13 is supported; this checkout was installed with Python 3.12 on
Windows using CPU PyTorch. From this directory in PowerShell:

```powershell
# The local .venv is already installed in this checkout.
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

To recreate on a new Windows machine with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
.\.venv\Scripts\python.exe -m pip install --find-links vendor/wheels -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts/download_models.py
```

On Linux/macOS, use `.venv/bin/python`. The lock records this Windows CPU
environment; use `pip install --find-links vendor/wheels -e '.[dev]'` after installing the appropriate
PyTorch build for other platforms. GPU training is optional and not installed here.

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
SHA-256 hashes. `download_models.py` verifies publisher hash prefixes. Inference
loads only local verified checkpoints with `weights_only=True`.

### Research boundaries

- Default classes: control, N deficiency, Fe deficiency, Xcc, N deficiency + Xcc.
- The proposal's primary endpoint is plant-level day-8 macro F1 on held-out cycles.
- Keep all days, leaves, repeats, and images of a plant together. Use globally
  unique IDs, including cycle where local plant numbers are reused.
- `grouped_folds` and `cycle_folds` enforce the core split boundaries. They do not
  constitute a complete nested model-selection or statistical analysis runner.
- No dataset images, private source PDF, laboratory protocol, or fabricated
  biological measurement is published by this setup.
- Synthetic smoke tests demonstrate software operation only.

See [design mapping](docs/design.md), [model sources and licenses](docs/models.md),
[verification](docs/verification.json), [data checks](docs/data-verification.json),
and [setup status](docs/setup-status.md).

Remote: https://github.com/Leafy-AI-90/AI.git
