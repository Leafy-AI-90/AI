# Training and evaluation (Phase 5)

`scripts/run_experiment.py` takes a validated YAML in `configs/experiments/`.
The input is a Phase 3 plant/day feature table or a labelled image index made
with `scripts/build_image_index.py`. Nothing in this repository has yet been
trained on Leafy AI plants, so no research performance claim is available.

## Prepare local inputs

```bash
python scripts/build_features.py --help
python scripts/build_image_index.py data/interim/plants.parquet data/interim/images.parquet data/interim/image_index.parquet
python scripts/download_models.py
python scripts/extract_dinov2.py data/interim/features.parquet data/interim/images.parquet data/interim/features_with_image_embeddings.parquet
python scripts/extract_dinov2.py data/interim/features.parquet data/interim/images.parquet data/interim/features_with_mobile_embeddings.parquet --backbone mobilenet_v3_large
python scripts/run_experiment.py configs/experiments/xcc_spectral.yaml
```

The feature table has treatment `class` from `plants`, globally unique
`plant_id`, `cycle`, `day`, and requested numeric columns. Image index labels
are joined from `plants`, never generated from images or sensor proxies. The
image paths must point to local files. Raw and interim data, weights and runs
are ignored by Git. Relative paths in YAML are relative to the working
directory (normally the repository root).

`--quick` runs the first seed with one image epoch or two Optuna trials. It is
for CPU software checks only. Full runs use the listed seeds, 384 px image
transforms and chosen search budget. CUDA can be selected in `image_training`
when available. Set a new run name or remove an ignored, failed run directory
before rerunning a name in the same UTC second.

## Split and model contracts

- Primary endpoint uses the preset day (day 8 for `xcc`, day 10 for
  `root_disease`) unless `endpoint_day` is explicitly overridden. One feature
  row is allowed per plant at that endpoint. Missing and nonfinite features
  raise an error; complete cohort selection must be reported separately.
- `nested_grouped` has stratified outer plant folds; `leave_one_cycle_out`
  holds out whole cycles. `external_test` fits on its source table once and
  requires disjoint plant IDs and cycles in its separate test file. Every
  trained scaler, PCA, selector, Optuna trial and image epoch selection sees
  only training plants inside the current outer fold.
- B3 supports logistic, PLS-DA, random forest and XGBoost. Optuna uses a fixed
  small budget in inner plant folds for logistic, random forest and XGBoost.
  PLS-DA has fixed components. B4 is `thermal_logistic` with only thermal
  features. B2 is frozen DINOv2 ViT-S/14 embeddings and fold-local logistic.
- B1 is EfficientNet-B0 fine-tuned afresh in each outer fold. The `phone_image`
  comparison uses MobileNetV3-Large with the same fold-local procedure. A
  grouped inner fold selects the image epoch by plant-level macro F1; the
  final model is initialized again from verified ImageNet weights and trained
  on all outer training plants for the selected epoch count. Image losses use
  plant-count class weights, AdamW and a cosine schedule. Multiple test images
  are averaged within plant before scoring.
- F1 trains grouped out of fold modality probabilities for logistic stacking.
  F2 applies PCA to frozen image embeddings inside the outer fold and combines
  them with measured features for XGBoost. F3 trains the gated attention head
  on one frozen global image token and spectral features; it is a comparison
  implementation and does not claim patch-level attention. The `proposed`
  config fixes F2 plus positional, water, thermal and temporal features.
  The `phone` fusion config uses the same measured modalities with frozen,
  verified MobileNetV3-Large embeddings and fold-local PCA and XGBoost.
- M660 is a binary chlorophyll proxy, not a defensible five-class treatment
  classifier. Its threshold comparison is handled by H1 rather than being
  given a misleading five-class macro F1. PlantVillage lacks N/Fe/water/salt
  Choy Sum labels. A five-class B5 domain-gap result cannot run until a
  biologically reviewed mapping is provided; the code does not invent one.

All real image embedding tables require a sidecar manifest matching the exact
table SHA-256, the pinned DINOv2 or MobileNet checkpoint SHA-256 and `frozen=true`.
The extractor uses deterministic 518 px DINOv2 or 384 px MobileNet inputs and averages image embeddings
per plant/day. Missing images leave missing embeddings and cause an explicit
error at the endpoint. A production F2 with a fine-tuned image encoder would
need that encoder trained inside each outer fold; this path uses frozen B2
embeddings as its image modality.

## Outputs

Each ignored `runs/<UTC>_<name>/` folder contains `run.json` (YAML values,
commit hash, input hashes, package versions, seeds, selected hyperparameters
or image epochs), `oof.parquet` (one held-out probability vector per plant and
seed), `metrics.json`, and model artifacts. A real-data run refuses to start
with uncommitted code. AUROC is one-vs-rest macro across study classes when
all classes occur in the held-out set. These files are inputs for the Phase 6
analysis scripts, which will perform plant-clustered inference.

## Optional archive pretraining

`scripts/pretrain_archive.py` uses the audited train and validation splits
from `data/processed/` and never the source test split. It supports
PlantVillage and PlantDoc archives locally. PlantDoc's downloaded license
text must identify CC-BY-4.0. PlantVillage's data terms need a local
`data/raw/plantvillage/license-review.json` with `license`, `source_url` and
`reviewed_by`; the author card and repository disagree on whether the data
license is explicit. Cauliflower Black Rot is skipped until its data license
is clear. Pretraining checkpoints remain ignored and their SHA-256 plus
dataset split hash are recorded. No archive-only score measures Choy Sum
performance.
