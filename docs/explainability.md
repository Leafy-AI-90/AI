# Explanations and paper report (Phase 7)

No trained biological model or measured result exists yet. These scripts are
ready to run after the experiment data, treatment records and held-out runs are
available. Synthetic outputs are software tests and require
`--allow-synthetic`; they are watermarked.

## Tabular SHAP

Run `python analysis/explain_tabular.py --run runs/<run> --seed 17 --fold 0`.
The script verifies the source-table hash in `run.json`, loads that fold's
saved model, chooses up to 20 held-out plants per class, and samples background
plants only from the outer training set. It writes global and per-class SHAP
bars, per-class beeswarms, and importance against wavelength for raw
reflectance and absorbance bands. Permutation SHAP operates on the **saved
probability predictor** and can be slow for wide fusion models. These values
describe this predictor's output under the selected background; they do not
identify causal plant physiology.

## Image Grad-CAM

Run `python analysis/gradcam.py --run runs/<image-run> --images
data/interim/images_explain.parquet --seed 17`. The image index needs one or
more leaf images per plant, `file_path`, `mask_path`, `class`, and `plant_id`.
Processed image masks from `scripts/process_images.py` can be joined to the
original image index by `image_id`. The script selects one leaf image per
randomly sampled held-out plant (up to 20 per class), loads that plant's outer
fold image checkpoint, and measures the fraction of Grad-CAM intensity inside
the leaf mask. It saves individual overlays and a per-class localization
table. This is a spatial audit, not an explanation of treatment causality.

## Paper report and Quarto site

Each H1–H5 and supporting analysis emits a `run.json` with the code revision
and hashes of its inputs. To assemble selected reports:

```sh
python reports/build_report.py --analysis h1=reports/h1 --analysis h2=reports/h2
quarto render reports/paper
```

The builder creates a new `reports/paper/` folder with every CSV rendered as
an HTML table, paired 300 dpi PNG and SVG figures, captions, a provenance
manifest, and a Quarto `index.qmd`. It refuses stale input hashes, uncommitted
real-data reports, mixed synthetic/real reports, and overwrite of an existing
paper folder. Quarto is a separate executable; install it from the official
project for local rendering. The generated `_site/` is a static site that can
be published to GitHub Pages after review. Generated reports and data remain
outside Git.

Use colorblind-safe blue, vermillion, green, purple and orange for category
plots; sequential intensity plots use Viridis. Labels include units where the
quantity has them.
