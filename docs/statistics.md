# Statistics and hypotheses (Phase 6)

No biological results are available yet. The scripts below consume measured,
held-out predictions and produce a CSV table, Markdown summary and 300 dpi PNG
plus SVG figure. Synthetic input is refused unless `--allow-synthetic` is used;
all such figures and summaries are visibly labelled as software tests.

| Hypothesis | One primary script | Primary quantity |
|---|---|---|
| H1 | `analysis/h1_calibration.py` | Held-out chlorophyll R² and RPD |
| H2 | `analysis/h2_fusion.py` | Fused macro F1 and paired difference |
| H3 | `analysis/h3_confusion.py` | Relative reduction in prespecified confusion pair |
| H4 | `analysis/h4_lead_time.py` | Median days from sustained alert to blind symptom |
| H5 | `analysis/h5_root_vs_leaf.py` | Leaf-only / leaf-plus-root macro F1 ratio |

## Input and inference rules

Use `runs/<run>/oof.parquet` from the experiment runner. A run must have
`run.json` beside its OOF table for real-data reports. H2, H3, H5 and ablations
align identical plant IDs, treatment labels, cycles, days and a single seed.
Repeated seeds are separate analyses; they cannot be counted as extra plants.
Select the best single modality for H2 independently of the held-out test
cycle, and preregister the selection rule. McNemar tests paired **correctness**
and therefore is not a direct significance test of macro F1. DeLong tests each
one-versus-rest AUROC; those p-values use Holm adjustment. Bootstrap intervals
draw complete plants, stratified by true class, with 2,000 replicates by default.

H3 takes each key pair as `--pair class_a:class_b`. A zero baseline confusion
rate makes relative reduction undefined and is reported as such. H4 needs a
long daily held-out prediction table and independent blinded score table with
`plant_id`, `day`, and `blind_score`. Its sustained alert requires stress
probability > 0.5 on two consecutive calendar days. Positive lead time means
the alert happened before the first blind score ≥1. Missing alert or symptom
events remain missing and are counted; bootstrap lead-time intervals condition
on plants with both events. The step plot displays time to model alert, with
plants censored at their final measurement. Daily OOF predictions must come
from models trained without the corresponding plant in every day-specific run.
The Phase 5 endpoint runner alone does not create these daily runs.

H5 is available only when `root_disease` is among the class labels. The
leaf-only and leaf-plus-root models must share the same held-out plants.

## Supporting analyses

- `analysis/ablations.py` accepts `--kind bands|modality|background|positions|pretraining`,
  `--run label=path` for each run and `--reference label`. Band curves require
  3, 6, 12, 18 and 22 band runs. The corresponding YAML configurations use
  fold-local logistic RFE on 18 reflectance and four transmission absorbance
  bands; selected names are recorded for every outer fold in `run.json`.
- `analysis/repeated_cv.py` requires paired fold scores with identical
  `test_plant_hash` and applies the Nadeau–Bengio variance correction. It does
  not use repeated-measures ANOVA on fold scores.
- `analysis/mixed_models.py` fits `feature ~ class × day + (1 | plant)` to one
  measured value per plant/day, reports missingness, and Holm-adjusts nominated
  class and interaction tests.
- `analysis/agreement.py` calculates Cohen κ from independent rater pairs at
  a fixed day, with plant bootstrap interval.
- `analysis/power.py` computes the [Connor (1987)](https://pubmed.ncbi.nlm.nih.gov/3567305/)
  normal approximation for McNemar paired sample size from prespecified or
  pilot discordant proportions. Pilot estimates are planning assumptions.

The study has no real-data run, statistical result or paper figure as of this
phase. These scripts do not infer treatment labels from sensor proxies or fill
missing biological measurements.
