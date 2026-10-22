import numpy as np
import pytest
from conftest import make_tables

from leafy_ai.config import load_config
from leafy_ai.feature_table import build_features
from leafy_ai.qc import run_qc
from leafy_ai.registry import feature_names

META = ["plant_id", "cycle", "class", "day", "n_positions", "n_readings", "n_excluded_readings"]


@pytest.mark.parametrize("preset", ["xcc", "root_disease", "salt_planb"])
def test_columns_match_registry(preset):
    cfg = load_config(preset)
    table = build_features(make_tables(cfg, days=(-1, 0, 1)), cfg)
    assert [c for c in table.columns if c not in META] == feature_names(cfg)
    assert not table.duplicated(["plant_id", "day"]).any()


def test_calibration_by_hand(cfg):
    tables = make_tables(cfg)
    clip = tables["clip_readings"]
    clip.loc[:, [c for c in clip if c.startswith("as_")]] = 20100.0
    table = build_features(tables, cfg)
    # (20100 - 100) / (40000 - 100) x 0.99 and log10((30000 - 50) / (8000 - 50))
    assert table["r_680"].iloc[0] == pytest.approx(20000 / 39900 * 0.99)
    assert table["a_660"].iloc[0] == pytest.approx(np.log10(29950 / 7950))
    assert table["m710"].iloc[0] == pytest.approx(0.0)
    if cfg.sensors.thermal:
        assert table["thermal_cwsi"].iloc[0] == pytest.approx(2 / 8)
        assert table["w970"].iloc[0] == pytest.approx(0.0)


def test_excluded_readings_left_out_and_counted(cfg):
    tables = make_tables(cfg)
    clip = tables["clip_readings"]
    clip.loc[0, "as_680"] = cfg.qc.saturation_count
    readings = run_qc(tables, cfg).readings
    table = build_features(tables, cfg, readings)
    first = table.set_index(["plant_id", "day"]).loc[(clip.loc[0, "plant_id"], clip.loc[0, "day"])]
    assert first["n_excluded_readings"] == 1
    assert first["n_readings"] == len(cfg.leaf_positions) * cfg.repeats - 1
    assert first["r_680"] < 0.6  # saturated reading not averaged in


def test_missing_position_gives_nan_contrast_not_fill(cfg):
    tables = make_tables(cfg)
    clip = tables["clip_readings"]
    plant = clip.loc[0, "plant_id"]
    drop = (
        (clip["plant_id"] == plant)
        & (clip["day"] == 0)
        & (clip["leaf_position"] == cfg.leaf_positions[0])
    )
    tables["clip_readings"] = clip[~drop]
    table = build_features(tables, cfg).set_index(["plant_id", "day"])
    assert np.isnan(table.loc[(plant, 0), "pos_m710_variance"])
    assert table.loc[(plant, 0), "n_positions"] == len(cfg.leaf_positions) - 1
    assert not np.isnan(table.loc[(plant, -1), "pos_m710_variance"])


def test_feature_docs_are_current():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("fdocs", root / "scripts/build_feature_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (root / "docs/features.md").read_text(encoding="utf-8") == module.render()


def test_build_features_script_after_ingest(tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / f"scripts/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    cfg = load_config("root_disease")
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, df in make_tables(cfg, days=(-1, 0, 1)).items():
        df.to_csv(raw / f"{name}.csv", index=False)
    out = tmp_path / "interim"
    load("ingest").main(
        [
            "--preset",
            "root_disease",
            "--raw",
            str(raw),
            "--out",
            str(out),
            "--report",
            str(tmp_path / "qc"),
        ]
    )
    load("build_features").main(["--preset", "root_disease", "--interim", str(out)])
    import pandas as pd

    assert len(pd.read_parquet(out / "features.parquet")) == len(cfg.classes) * 2 * 3
