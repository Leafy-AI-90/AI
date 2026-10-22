import pytest

from leafy_ai.qc import run_qc, write_report


def flags(result, column):
    return result.readings[column]


def test_clean_fixture_has_no_flags(cfg, tables):
    result = run_qc(tables, cfg)
    assert len(result.readings) == len(tables["clip_readings"])
    qc = [c for c in result.readings if c.startswith("qc_")]
    assert not result.readings[qc].any().any()
    assert not result.readings["excluded"].any()


def test_saturation_and_negative_excluded_but_kept(cfg, tables):
    clip = tables["clip_readings"]
    clip.loc[0, "as_680"] = cfg.qc.saturation_count
    clip.loc[1, "as_410"] = 50.0  # below dark count of 100
    result = run_qc(tables, cfg)
    assert len(result.readings) == len(clip)
    assert result.readings.loc[0, "qc_saturated"] and result.readings.loc[0, "excluded"]
    assert result.readings.loc[1, "qc_negative_counts"]
    assert "negative_counts" in result.readings.loc[1, "exclusion_reasons"]


def test_white_and_temperature_drift_flag_whole_session(cfg, tables):
    refs = tables["references"]
    refs.loc[refs["session_id"] == "S-1", "white_500"] = 0  # no such column: guard
    refs = refs.drop(columns="white_500")
    first = refs.index[refs["session_id"] == "S-1"][0]
    refs.loc[first, "white_560"] = 40000 * 1.05  # 5% drift > 2%
    tables["references"] = refs
    clip = tables["clip_readings"]
    clip.loc[clip.index[clip["session_id"] == "S0"][0], "sensor_temp_c"] = 28.5
    result = run_qc(tables, cfg)
    r = result.readings
    assert r.loc[r["session_id"] == "S-1", "qc_white_drift"].all()
    assert not r.loc[r["session_id"] == "S0", "qc_white_drift"].any()
    assert r.loc[r["session_id"] == "S0", "qc_temperature_drift"].all()
    assert not r["excluded"].any()  # drift is flagged, not excluded, by default


def test_repeat_cv_hand_calculated(cfg, tables):
    clip = tables["clip_readings"]
    plant, pos = clip.loc[0, "plant_id"], cfg.leaf_positions[0]
    rows = clip.index[
        (clip["plant_id"] == plant) & (clip["day"] == -1) & (clip["leaf_position"] == pos)
    ]
    for i, value in zip(rows, (10100.0, 12100.0, 14100.0), strict=True):
        for band in cfg.spectral.bands_nm:
            clip.loc[i, f"as_{band}"] = value
    result = run_qc(tables, cfg)
    # dark-corrected 10000, 12000, 14000: CV = 2000 / 12000
    cv = result.readings.loc[rows, "repeat_cv"].iloc[0]
    assert cv == pytest.approx(2000 / 12000)
    assert result.readings.loc[rows, "qc_repeat_cv"].all()


def test_missing_position_and_short_repeats(cfg, tables):
    clip = tables["clip_readings"]
    plant = clip.loc[0, "plant_id"]
    drop = (
        (clip["plant_id"] == plant)
        & (clip["day"] == 0)
        & (clip["leaf_position"] == cfg.leaf_positions[-1])
    )
    short = (
        (clip["plant_id"] == plant)
        & (clip["day"] == -1)
        & (clip["leaf_position"] == cfg.leaf_positions[0])
        & (clip["repeat"] == 3)
    )
    tables["clip_readings"] = clip[~drop & ~short]
    result = run_qc(tables, cfg)
    c = result.completeness.set_index(["plant_id", "day"])
    assert c.loc[(plant, 0), "qc_missing_position"]
    assert c.loc[(plant, 0), "missing_positions"] == cfg.leaf_positions[-1]
    assert c.loc[(plant, -1), "short_repeats"] == cfg.leaf_positions[0]


def test_plant_exclusion_and_colour_checker(cfg, tables, tmp_path):
    tables["plants"].loc[0, ["exclusion_flag", "exclusion_reason"]] = [True, "pest damage"]
    tables["images"].loc[0, "colour_checker_detected"] = False
    result = run_qc(tables, cfg)
    plant = tables["plants"].loc[0, "plant_id"]
    mine = result.readings["plant_id"] == plant
    assert (
        result.readings.loc[mine, "excluded"].all()
        and not result.readings.loc[~mine, "excluded"].any()
    )
    assert result.images.loc[0, "qc_no_colour_checker"]
    page = write_report(result, tmp_path, cfg).read_text()
    assert "pest damage" in page and "no_colour_checker" in page
    assert (tmp_path / "flagged_readings.csv").exists()
