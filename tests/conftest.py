"""Small hand-built tables for tests. Fixtures only: not study data, never written to data/."""

import numpy as np
import pandas as pd
import pytest

from leafy_ai.config import load_config
from leafy_ai.schemas import band_columns, led_columns, views


def make_tables(cfg, plants_per_class=2, days=(-1, 0)):
    rng = np.random.default_rng(0)
    plant_rows, clip, gt, images, thermal = [], [], [], [], []
    start = pd.Timestamp("2026-01-05 09:00")
    for k, cls in enumerate(cfg.classes):
        for j in range(plants_per_class):
            plant_rows.append(
                dict(
                    plant_id=f"C1_{k}{j}",
                    cycle=1,
                    **{"class": cls},
                    rack="R1",
                    shelf="S1",
                    position=f"P{k}{j}",
                    sow_date="2025-12-01",
                    transplant_date="2025-12-15",
                    cultivar="choy_sum",
                    seed_lot="L1",
                    exclusion_flag=False,
                    exclusion_reason=None,
                )
            )
    bands, darks, leds = band_columns(cfg), band_columns(cfg, "dark"), led_columns(cfg)
    for p in plant_rows:
        for day in days:
            session = f"S{day}"
            t = start + pd.Timedelta(days=day + 1)
            for pos in cfg.leaf_positions:
                for rep in range(1, cfg.repeats + 1):
                    row = dict(
                        plant_id=p["plant_id"],
                        day=day,
                        timestamp=t,
                        leaf_position=pos,
                        repeat=rep,
                        integration_time_ms=50.0,
                        gain=16.0,
                        sensor_temp_c=25.0,
                        session_id=session,
                    )
                    row |= {c: 20000.0 * (1 + rng.normal(0, 0.005)) for c in bands}
                    row |= {c: 100.0 for c in darks}
                    row |= {c: (8000.0 if c.endswith("_on") else 50.0) for c in leds}
                    clip.append(row)
            gt.append(
                dict(
                    plant_id=p["plant_id"],
                    day=day,
                    chl_a_ug_cm2=30.0,
                    chl_b_ug_cm2=10.0,
                    chl_total_ug_cm2=40.0,
                    extraction_solvent="acetone_80",
                    tissue_n_pct=None,
                    sap_nitrate_ppm=None,
                    rwc_pct=85.0,
                    root_browning=None,
                    pathogen_load=None,
                    pathogen_confirmed=None,
                    rater1_score=0,
                    rater2_score=0,
                )
            )
            for view in views(cfg):
                images.append(
                    dict(
                        image_id=f"{p['plant_id']}_{day}_{view}",
                        plant_id=p["plant_id"],
                        day=day,
                        view=view,
                        file_path=f"img/{p['plant_id']}_{day}_{view}.jpg",
                        phone_model="test",
                        exposure_s=0.01,
                        iso=100,
                        white_balance_k=5000,
                        colour_checker_detected=True,
                    )
                )
            thermal.append(
                dict(
                    plant_id=p["plant_id"],
                    day=day,
                    timestamp=t,
                    leaf_mean_c=24.0,
                    leaf_median_c=24.0,
                    leaf_p10_c=23.0,
                    leaf_p90_c=25.0,
                    leaf_std_c=0.5,
                    air_temp_c=26.0,
                    rh_pct=70.0,
                    wet_ref_c=22.0,
                    dry_ref_c=30.0,
                    frame_file="f.npy",
                )
            )
    refs = []
    for day in days:
        for minute in (0, 60):
            row = dict(
                session_id=f"S{day}", timestamp=start + pd.Timedelta(days=day + 1, minutes=minute)
            )
            row |= {c: 40000.0 for c in band_columns(cfg, "white")}
            row |= {c: 100.0 for c in band_columns(cfg, "dark")}
            row |= {c: (30000.0 if c.endswith("_on") else 50.0) for c in led_columns(cfg, "empty")}
            row |= {c: 0.99 for c in band_columns(cfg, "tile_r")}
            refs.append(row)
    tables = dict(
        plants=pd.DataFrame(plant_rows),
        clip_readings=pd.DataFrame(clip),
        references=pd.DataFrame(refs),
        images=pd.DataFrame(images),
        ground_truth=pd.DataFrame(gt),
    )
    if cfg.sensors.thermal:
        tables["thermal"] = pd.DataFrame(thermal)
    return tables


@pytest.fixture(params=["xcc", "root_disease"])
def cfg(request):
    return load_config(request.param)


@pytest.fixture
def tables(cfg):
    return make_tables(cfg)
