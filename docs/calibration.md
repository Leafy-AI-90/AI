# H1 chlorophyll calibration

`analysis/h1_calibration.py` compares PLSR on 18 reflectance bands and the
configured transmission absorbances against two univariate linear baselines:
M660 and the red-edge chlorophyll index. Each calibration plant contributes one
row at the configured endpoint day (or an explicitly supplied `--day`). Outer
folds hold out whole plants. PLS components are selected by inner plant-grouped
cross-validation, and every fitted model is trained only on its outer training
fold. The script saves plant-level out-of-fold predictions, a CSV metric table,
VIP values, 300 dpi PNG and SVG figures, and a Markdown summary. It needs
measured `ground_truth` values; it does not create biology data.

```shell
python analysis/h1_calibration.py --preset xcc \
  --features data/interim/features.parquet \
  --ground-truth data/interim/ground_truth.parquet \
  --out reports/h1
```

The extraction solvent is declared in each preset's `calibration.solvent`.
`leafy_ai.calibration.lichtenthaler` converts measured, blank-corrected extract
absorbances into µg/cm² using measured extract volume, dilution, cuvette path
length and sampled leaf area. Use 663.2/646.8 nm for 80% acetone or
664.1/648.6 nm for 95% ethanol. Record the resulting measured a, b and total
values in `ground_truth.csv`, with its solvent. Do not use AS7265x or clip LED
readings as extraction absorbances. The coefficients and wavelengths follow
[Lichtenthaler and Buschmann (2001)](https://www.jkip.kit.edu/molbio/download/425%20Lichtenthaler%20und%20Buschmann%202001%20Chl%20determ%20CurrProtFoodAnalChem%204-3-1.pdf).

RPD is the sample standard deviation of extracted chlorophyll divided by the
held-out RMSE. Bland–Altman differences are prediction minus measurement, with
limits at the mean difference ± 1.96 sample standard deviations. Targets
(R² ≥ 0.85 and RPD ≥ 2.0) are reported as targets, never assumed results.
Missing measurements and solvent mismatches are counted and left out of the
paired analysis. Synthetic input requires `--allow-synthetic` and visibly
watermarks every figure.
