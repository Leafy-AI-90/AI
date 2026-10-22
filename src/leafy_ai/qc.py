"""Measurement quality control. Flags rows, never drops them.

Each rule adds a boolean ``qc_<flag>`` column. Rows are marked ``excluded``
only for flags listed in the preset's pre-registered ``qc.exclude_flags``, or
for plants whose ``exclusion_flag`` is set in ``plants``. Reports list every
flag so reviewers see what was kept and what was excluded.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig
from .schemas import band_columns, led_columns


@dataclass
class QCResult:
    readings: pd.DataFrame
    completeness: pd.DataFrame
    images: pd.DataFrame
    summary: pd.DataFrame


def _signal_columns(cfg: StudyConfig) -> list[str]:
    return band_columns(cfg) + led_columns(cfg)


def flag_saturated(clip: pd.DataFrame, cfg: StudyConfig) -> pd.Series:
    return (clip[_signal_columns(cfg)] >= cfg.qc.saturation_count).any(axis=1)


def flag_negative(clip: pd.DataFrame, cfg: StudyConfig) -> pd.Series:
    """Negative raw counts, or band signal at or below its dark count."""
    raw = clip[_signal_columns(cfg) + band_columns(cfg, "dark")] < 0
    corrected = clip[band_columns(cfg)].to_numpy() - clip[band_columns(cfg, "dark")].to_numpy()
    return raw.any(axis=1) | pd.Series((corrected <= 0).any(axis=1), index=clip.index)


def white_drift(references: pd.DataFrame, cfg: StudyConfig) -> pd.Series:
    """Per session: max over bands of (max - min) / mean of dark-corrected white counts."""
    white = (
        references[band_columns(cfg, "white")].to_numpy()
        - references[band_columns(cfg, "dark")].to_numpy()
    )
    frame = pd.DataFrame(white, index=references["session_id"])
    grouped = frame.groupby(level=0)
    return ((grouped.max() - grouped.min()) / grouped.mean()).max(axis=1)


def temperature_drift(clip: pd.DataFrame) -> pd.Series:
    """Per session: sensor temperature range in degrees C."""
    temp = clip.groupby("session_id")["sensor_temp_c"]
    return temp.max() - temp.min()


def repeat_cv(clip: pd.DataFrame, cfg: StudyConfig) -> pd.Series:
    """Per plant/day/position: median over bands of dark-corrected CV across repeats."""
    keys = ["plant_id", "day", "leaf_position"]
    corrected = clip[band_columns(cfg)].to_numpy() - clip[band_columns(cfg, "dark")].to_numpy()
    frame = pd.DataFrame(corrected, index=pd.MultiIndex.from_frame(clip[keys]))
    grouped = frame.groupby(level=keys)
    return (grouped.std(ddof=1) / grouped.mean()).median(axis=1)


def completeness(clip: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    """Per plant/day: configured positions and repeats actually measured."""
    rows = []
    for (plant, day), group in clip.groupby(["plant_id", "day"]):
        counts = group.groupby("leaf_position")["repeat"].nunique()
        missing = [p for p in cfg.leaf_positions if counts.get(p, 0) == 0]
        short = [p for p in cfg.leaf_positions if 0 < counts.get(p, 0) < cfg.repeats]
        rows.append(
            dict(
                plant_id=plant,
                day=day,
                missing_positions=",".join(missing),
                short_repeats=",".join(short),
                qc_missing_position=bool(missing),
            )
        )
    return pd.DataFrame(rows)


def run_qc(tables: dict[str, pd.DataFrame], cfg: StudyConfig) -> QCResult:
    clip = tables["clip_readings"].copy()
    rules = cfg.qc
    clip["qc_saturated"] = flag_saturated(clip, cfg)
    clip["qc_negative_counts"] = flag_negative(clip, cfg)
    drift = white_drift(tables["references"], cfg)
    clip["white_drift"] = clip["session_id"].map(drift)
    clip["qc_white_drift"] = clip["white_drift"] > rules.white_drift_max
    temp = temperature_drift(clip)
    clip["temperature_drift_c"] = clip["session_id"].map(temp)
    clip["qc_temperature_drift"] = clip["temperature_drift_c"] > rules.temperature_drift_max_c
    cv = repeat_cv(clip, cfg).rename("repeat_cv").reset_index()
    clip = clip.merge(cv, on=["plant_id", "day", "leaf_position"], how="left")
    clip["qc_repeat_cv"] = clip["repeat_cv"] > rules.repeat_cv_max
    complete = completeness(clip, cfg)
    clip = clip.merge(
        complete[["plant_id", "day", "qc_missing_position"]], on=["plant_id", "day"], how="left"
    )
    images = tables["images"].copy()
    images["qc_no_colour_checker"] = ~images["colour_checker_detected"].astype(bool)
    clip = apply_exclusions(clip, tables["plants"], cfg)
    images = apply_exclusions(images, tables["plants"], cfg)
    return QCResult(clip, complete, images, summarise(clip, images, complete))


def apply_exclusions(df: pd.DataFrame, plants: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    """Add ``excluded`` and ``exclusion_reasons``; row count never changes."""
    out = df.copy()
    flags = [f for f in cfg.qc.exclude_flags if f"qc_{f}" in out]
    hits = out[[f"qc_{f}" for f in flags]].fillna(False).astype(bool).to_numpy()
    excluded = plants[plants["exclusion_flag"].astype(bool)]
    plant_reason = dict(zip(excluded["plant_id"], excluded["exclusion_reason"], strict=True))
    reasons = []
    for row_hits, plant in zip(hits, out["plant_id"], strict=True):
        found = [f for f, hit in zip(flags, row_hits, strict=True) if hit]
        if plant in plant_reason:
            found.append(f"plant: {plant_reason[plant]}")
        reasons.append("; ".join(found))
    out["exclusion_reasons"] = reasons
    out["excluded"] = out["exclusion_reasons"] != ""
    return out


def summarise(clip: pd.DataFrame, images: pd.DataFrame, complete: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in (("clip_readings", clip), ("images", images), ("plant_day", complete)):
        for column in [c for c in df.columns if c.startswith("qc_")]:
            rows.append(
                dict(
                    table=name,
                    flag=column[3:],
                    flagged=int(df[column].fillna(False).sum()),
                    total=len(df),
                )
            )
        if "excluded" in df:
            rows.append(
                dict(table=name, flag="excluded", flagged=int(df["excluded"].sum()), total=len(df))
            )
    summary = pd.DataFrame(rows)
    summary["pct"] = np.round(100 * summary["flagged"] / summary["total"].clip(lower=1), 2)
    return summary


def write_report(result: QCResult, out_dir: str | Path, cfg: StudyConfig) -> Path:
    """Write ``qc.html`` plus CSV of every flagged row to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flag_cols = [c for c in result.readings.columns if c.startswith("qc_")]
    flagged = result.readings[result.readings[flag_cols].any(axis=1) | result.readings["excluded"]]
    flagged.to_csv(out / "flagged_readings.csv", index=False)
    result.summary.to_csv(out / "qc_summary.csv", index=False)
    incomplete = result.completeness[
        result.completeness["qc_missing_position"] | (result.completeness["short_repeats"] != "")
    ]
    no_checker = result.images[result.images["qc_no_colour_checker"]]
    rules = html.escape(cfg.qc.model_dump_json(indent=2))
    sections = [
        ("Summary", result.summary),
        ("Incomplete plant/days", incomplete),
        ("Images without colour checker", no_checker),
        ("Flagged readings (first 200; full list in flagged_readings.csv)", flagged.head(200)),
    ]
    body = "".join(
        f"<h2>{html.escape(t)}</h2>{df.to_html(index=False, na_rep='')}" for t, df in sections
    )
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>QC {cfg.preset}</title>"
        "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse;font-size:12px}"
        "td,th{border:1px solid #ccc;padding:2px 6px}</style></head><body>"
        f"<h1>QC report: {html.escape(cfg.preset)}</h1>"
        "<p>Rows are flagged, never dropped. Excluded rows follow pre-registered rules below.</p>"
        f"<pre>{rules}</pre>{body}</body></html>"
    )
    path = out / "qc.html"
    path.write_text(page, encoding="utf-8")
    return path
