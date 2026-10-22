# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added
- Phase 1: `configs/presets/` YAML for `root_disease`, `xcc` and `salt_planb` with pydantic loader `leafy_ai.config.load_config`
- GitHub Actions CI: ruff lint and format check, pytest on Ubuntu and Windows, Python 3.11 to 3.13, CPU only, synthetic smoke test, cached checkpoint smoke test
- `.gitignore`, pre-commit, `CONTRIBUTING.md`, `CITATION.cff`, issue and PR templates, `CODEOWNERS`
- `pyproject.toml` extras `[vision]`, `[stats]`, `[export]`, `[dev]`, `[all]`
- `scripts/smoke_test.py --skip-checkpoints` for runs without downloaded weights

### Changed
- `leafy_ai.CLASSES` now reads default `xcc` preset; value unchanged
- Core install no longer pulls PyTorch; use `[vision]` extra
- Platform neutral install docs; personal paths removed
- Code formatted with ruff; no behaviour change
