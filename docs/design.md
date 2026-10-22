# Scope derived from the supplied review

Source: Plant_Stress_Sensor_Review_v2.docx.pdf, 32 pages. Model requirements are
on page 18, sections 6.9.2 and 6.10; validation and endpoints are on page 19;
calibration formulae are on page 15. The document was treated as reference material,
not as authorization to perform its laboratory, publication, contact, scheduling,
or account-management instructions.

## Implemented foundation

1. Official TorchVision image backbones and ImageNet checkpoint installation.
2. Scikit-learn PLSR, PLS-DA, logistic regression, random forest, SVM and MLP.
3. XGBoost intermediate fusion, grouped probability stacking, and a small
   PyTorch gated cross-attention comparison head.
4. Calibrated reflectance and absorbance; adjacent-band derivatives per nm;
   NDVI, NDRE, red-edge chlorophyll index, GNDVI, PRI proxy; M660 and M710;
   positional age/lesion contrasts and variance.
5. Plant-grouped and cycle-held-out splits, regression tests against leakage.
6. Installed timm, statsmodels, SHAP, and Grad-CAM for subsequent experiments.

## Decisions

- MobileNetV3-Large is the selected V3 variant; the proposal does not choose one.
- CPU wheels make setup usable without assuming an NVIDIA GPU.
- ImageNet versions are explicit, not floating `DEFAULT` references.
- Research transforms use 384 px and at most 5% brightness/contrast/saturation
  jitter. Native 224 px ImageNet evaluation preprocessing is available from each
  weight enum; the 384 px transforms are for the proposed study.
- PLS-DA is a one-hot PLS implementation with softmax scores. These scores require
  grouped calibration before calibration-sensitive analysis or threshold claims.
- B0 is a binary low-chlorophyll comparator, not a five-class diagnosis model.
- B4 is left pending a PlantVillage-specific training run. Its class taxonomy
  must be explicitly mapped for a domain-gap analysis; it does not cover the
  proposed five Choy Sum treatment labels out of the box.
- F3 is a generic trainable comparison head, not an exact reproduction of an
  unspecified CCAFM paper or a pretrained plant-stress model.

## Data contract for future training

Full table contract: [data-schema.md](data-schema.md). Feature columns: [features.md](features.md). Summary:

Use one analysis row per plant/day after explicitly aggregating measurement
repeats. Raw readings need plant_id, cycle, day, leaf/position, repeat ID,
calibration references, sensor settings and paired image IDs. Labels must come
from the independently established treatment/ground-truth records, never the
same optical proxy being tested. Keep raw files immutable and out of Git.

Reflectance column order is exported as `leafy_ai.BANDS`; transmission absorbance
order is 660, 710, 940 nm. Positional features take repeat-averaged M710 in the order
oldest leaf, youngest leaf, lesion site, off-lesion site, within the same plant/day.

Future work: end-to-end fine-tuning, nested hyperparameter
search, class weighting for all estimators, grouped probability calibration,
cluster bootstrap/statistical comparisons, explainability reports, band-budget
ablation and external validation. These are not completed or claimed by installation.

## Source review limits

The source's literature assertions and future performance targets were not
independently validated. This setup uses its software requirements, not its
claimed novelty, journal ranking, performance guarantees or biology procedures.
