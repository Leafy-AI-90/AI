## Summary

## Checks

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest -q` passes
- [ ] `python scripts/smoke_test.py --skip-checkpoints` passes
- [ ] Splits and resampling stay grouped by plant_id; fitting stays inside training folds
- [ ] No raw data, weights or synthetic results presented as real
- [ ] CHANGELOG and docs updated
