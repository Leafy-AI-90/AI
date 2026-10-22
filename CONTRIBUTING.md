# Contributing

## Setup

See [README install](README.md#install). Then:

```bash
pre-commit install
pytest -q
python scripts/smoke_test.py --skip-checkpoints
```

## Workflow

1. One branch and pull request per phase or focused change
2. Run `ruff check .`, `ruff format --check .`, `pytest -q` and smoke test before pushing
3. Add short entry under `Unreleased` in `CHANGELOG.md`
4. Update docs touched by change

## Research rules

1. Unit of analysis = plant. Every split, bootstrap and test groups by `plant_id`
2. Fit scalers, PCA, feature and band selectors, calibration and hyperparameter search inside training folds only
3. Labels come from treatment records and ground truth, never from optical proxy under test
4. Never fabricate, impute or fill in biological measurements; missing stays missing and is reported
5. No metric or figure enters README, docs or paper folder unless committed code produced it from real data, with run directory referenced
6. Raw data stays immutable and out of Git; track only schemas, manifests, hashes and small summaries
7. Pin versions and record seeds; runs reproduce from config + commit hash
8. Respect dataset and weight licences; record attribution in `docs/models.md` and `docs/datasets.md`
9. Small typed functions with docstrings and tests
10. Ask maintainers before deleting files or changing public signatures; keep deprecation shims

## Study design changes

Edit YAML in `configs/presets/`, not Python. Loader validation lives in
`src/leafy_ai/config.py`; add test for every new rule.
