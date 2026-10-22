import json

import pytest

from leafy_ai.io import check_integrity, join_plant_day, load_raw, write_interim


def export(tables, folder, parquet=("references",)):
    folder.mkdir()
    for name, df in tables.items():
        if name in parquet:
            df.to_parquet(folder / f"{name}.parquet", index=False)
        else:
            df.to_csv(folder / f"{name}.csv", index=False)
    return folder


def test_load_validate_and_write_immutable(cfg, tables, tmp_path):
    data = load_raw(export(tables, tmp_path / "raw"), cfg)
    assert set(data.tables) == set(tables)
    manifest = write_interim(data, tmp_path / "interim", cfg)
    assert manifest["outputs"]["plant_day"]["rows"] == len(tables["plants"]) * 2
    assert all(len(v["sha256"]) == 64 for v in manifest["sources"].values())
    again = write_interim(data, tmp_path / "interim", cfg)  # identical rerun is a no-op
    assert again == json.loads((tmp_path / "interim/manifest.json").read_text())
    data.tables["plants"].loc[0, "rack"] = "R9"
    with pytest.raises(FileExistsError):
        write_interim(data, tmp_path / "interim", cfg)


def test_missing_ambiguous_and_disabled_tables(cfg, tables, tmp_path):
    raw = export({k: v for k, v in tables.items() if k != "images"}, tmp_path / "a")
    with pytest.raises(FileNotFoundError):
        load_raw(raw, cfg)
    raw = export(tables, tmp_path / "b")
    tables["plants"].to_parquet(raw / "plants.parquet")
    with pytest.raises(ValueError, match="Ambiguous"):
        load_raw(raw, cfg)
    if not cfg.sensors.thermal:
        raw = export(tables, tmp_path / "c")
        tables["ground_truth"].to_csv(raw / "thermal.csv")
        with pytest.raises(ValueError, match="disables"):
            load_raw(raw, cfg)


def test_orphan_plants_and_sessions_rejected(tables):
    tables["ground_truth"].loc[0, "plant_id"] = "C1_ghost"
    with pytest.raises(ValueError, match="unknown plant_id"):
        check_integrity(tables)
    tables["ground_truth"].loc[0, "plant_id"] = tables["plants"].loc[0, "plant_id"]
    tables["references"] = tables["references"].iloc[2:]
    with pytest.raises(ValueError, match="sessions without references"):
        check_integrity(tables)


def test_join_keeps_one_row_per_plant_day(cfg, tables):
    joined = join_plant_day(tables)
    assert not joined.duplicated(["plant_id", "day"]).any()
    assert set(joined["class"]) == set(cfg.classes)
    if cfg.sensors.thermal:
        assert joined["leaf_mean_c"].notna().all()
        tables["thermal"] = tables["thermal"].assign(timestamp=0)
        doubled = tables["thermal"].copy()
        doubled["timestamp"] = 1
        import pandas as pd

        tables["thermal"] = pd.concat([tables["thermal"], doubled])
        with pytest.raises(ValueError, match="Multiple thermal"):
            join_plant_day(tables)


def test_ingest_script_end_to_end(cfg, tables, tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/ingest.py"
    spec = importlib.util.spec_from_file_location("ingest", script)
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)
    raw = export(tables, tmp_path / "raw")
    out, report = tmp_path / "interim", tmp_path / "qc"
    argv = ["--preset", cfg.preset, "--raw", str(raw), "--out", str(out), "--report", str(report)]
    ingest.main(argv)
    assert (out / "clip_readings_qc.parquet").exists() and (report / "qc.html").exists()
    ingest.main(argv)  # rerun on identical input is a no-op, not an error
