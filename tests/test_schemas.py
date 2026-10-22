import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from leafy_ai.config import load_config
from leafy_ai.schemas import band_columns, led_columns, required_tables, schema, validate


def test_fixture_tables_validate(cfg, tables):
    for name, df in tables.items():
        assert len(validate(name, df, cfg)) == len(df)


def test_columns_follow_preset():
    xcc, root = load_config("xcc"), load_config("root_disease")
    assert "led970_on" in led_columns(root) and "led970_on" not in led_columns(xcc)
    assert len(band_columns(xcc)) == 18
    assert "thermal" in required_tables(root) and "thermal" not in required_tables(xcc)
    assert "root" in schema("images", root).columns["view"].checks[0].statistics["allowed_values"]


def broken(tables, table, **changes):
    df = tables[table].copy()
    for column, value in changes.items():
        df.loc[df.index[0], column] = value
    return df


@pytest.mark.parametrize(
    "table,changes",
    [
        ("plants", {"class": "unknown_class"}),
        ("plants", {"plant_id": "C2_00"}),  # cycle 1 plant must start with C1_
        ("plants", {"exclusion_flag": True}),  # excluded without reason
        ("clip_readings", {"leaf_position": "top"}),
        ("clip_readings", {"repeat": 9}),
        ("clip_readings", {"day": 30}),
        ("references", {"tile_r_410": 1.5}),
        ("images", {"view": "side"}),
        ("ground_truth", {"extraction_solvent": "methanol"}),
        ("ground_truth", {"rater1_score": 7}),
    ],
)
def test_invalid_rows_rejected(cfg, tables, table, changes):
    with pytest.raises(SchemaErrors):
        validate(table, broken(tables, table, **changes), cfg)


def test_duplicate_reading_and_extra_column_rejected(cfg, tables):
    clip = tables["clip_readings"]
    with pytest.raises(SchemaErrors):
        validate("clip_readings", pd.concat([clip, clip.head(1)]), cfg)
    with pytest.raises(SchemaErrors):
        validate("plants", tables["plants"].assign(extra=1), cfg)


def test_missing_biology_stays_missing(cfg, tables):
    gt = tables["ground_truth"].copy()
    gt.loc[:, ["chl_a_ug_cm2", "chl_b_ug_cm2", "chl_total_ug_cm2", "rwc_pct"]] = None
    out = validate("ground_truth", gt, cfg)
    assert out["chl_total_ug_cm2"].isna().all() and out["rwc_pct"].isna().all()


def test_root_view_rejected_without_root_images(tables):
    cfg = load_config("xcc")
    with pytest.raises(SchemaErrors):
        validate("images", broken(tables, "images", view="root"), cfg)


def test_schema_docs_are_current():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("docs", root / "scripts/build_schema_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = (root / "docs/data-schema.md").read_text(encoding="utf-8")
    assert current == module.render(), "Run python scripts/build_schema_docs.py"
