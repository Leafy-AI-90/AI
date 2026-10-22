"""H4: first sustained model alert versus first blinded visible symptom."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, write_summary


def _first_alert(group: pd.DataFrame, control: str) -> float:
    ordered = group.sort_values("day")
    if ordered.day.duplicated().any():
        raise ValueError("One probability row per plant and day required")
    probability = 1 - ordered[f"p_{control}"].to_numpy(dtype=float)
    days = ordered.day.to_numpy(dtype=int)
    pair = (probability[:-1] > 0.5) & (probability[1:] > 0.5) & (np.diff(days) == 1)
    return float(days[np.flatnonzero(pair)[0]]) if pair.any() else np.nan


def evaluate(
    daily: pd.DataFrame,
    scores: pd.DataFrame,
    control: str = "control",
    seed: int | None = None,
    reps: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return class summary, per-plant lead time, and detection survival curve."""
    if reps < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    required = {"plant_id", "day", "class", "seed", f"p_{control}"}
    if required - set(daily) or {"plant_id", "day", "blind_score"} - set(scores):
        raise ValueError("Daily OOF or blinded score table lacks required columns")
    seeds = daily.seed.unique()
    if seed is None:
        if len(seeds) != 1:
            raise ValueError("Select one seed; repeated seeds are not independent plants")
        seed = seeds[0]
    data = daily.loc[daily.seed == seed].copy()
    if data.duplicated(["plant_id", "day"]).any() or scores.duplicated(["plant_id", "day"]).any():
        raise ValueError("Duplicate plant-day observations")
    if not np.isfinite(data[f"p_{control}"]).all():
        raise ValueError("Missing model probability; no imputation allowed")
    if ((data[f"p_{control}"] < 0) | (data[f"p_{control}"] > 1)).any():
        raise ValueError("Control probabilities must lie in [0, 1]")
    rows = []
    for plant, group in data.groupby("plant_id"):
        if group["class"].nunique() != 1:
            raise ValueError("Plant class changes across days")
        plant_scores = scores.loc[scores.plant_id == plant].sort_values("day")
        scored_days = plant_scores.loc[plant_scores.blind_score.notna(), "day"]
        missing_score_days = int((~group.day.isin(scored_days)).sum())
        symptoms = plant_scores.loc[plant_scores.blind_score >= 1, "day"]
        symptom_day = float(symptoms.iloc[0]) if not symptoms.empty else np.nan
        alert_day = _first_alert(group, control)
        rows.append(
            {
                "plant_id": plant,
                "class": group["class"].iloc[0],
                "alert_day": alert_day,
                "symptom_day": symptom_day,
                "lead_days": symptom_day - alert_day,
                "last_day": int(group.day.max()),
                "missing_score_days": missing_score_days,
            }
        )
    per_plant = pd.DataFrame(rows)
    per_plant = per_plant.loc[per_plant["class"] != control].reset_index(drop=True)
    rng = np.random.default_rng(23)
    summaries, curves = [], []
    for name, group in per_plant.groupby("class"):
        observed = group.lead_days.dropna().to_numpy()
        if len(observed):
            draws = [
                np.median(rng.choice(observed, len(observed), replace=True)) for _ in range(reps)
            ]
            low, high = np.quantile(draws, [0.025, 0.975])
            median = float(np.median(observed))
        else:
            low = high = median = np.nan
        summaries.append(
            {
                "class": name,
                "plants": len(group),
                "paired_events": len(observed),
                "without_alert": int(group.alert_day.isna().sum()),
                "without_symptom": int(group.symptom_day.isna().sum()),
                "missing_score_days": int(group.missing_score_days.sum()),
                "median_lead_days": median,
                "ci_low": low,
                "ci_high": high,
            }
        )
        survival = 1.0
        for day in sorted(set(group.last_day) | set(group.alert_day.dropna().astype(int))):
            risk = int(
                (
                    ((group.alert_day.isna()) | (group.alert_day >= day)) & (group.last_day >= day)
                ).sum()
            )
            events = int((group.alert_day == day).sum())
            if risk:
                survival *= 1 - events / risk
            curves.append(
                {
                    "class": name,
                    "day": day,
                    "not_yet_alerted": survival,
                    "at_risk": risk,
                    "events": events,
                }
            )
    return pd.DataFrame(summaries), per_plant, pd.DataFrame(curves)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-oof", type=Path, required=True)
    parser.add_argument("--blind-scores", type=Path, required=True)
    parser.add_argument("--control", default="control")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("reports/h4"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    synthetic = prepare_report(
        [args.daily_oof, args.blind_scores], args.out, args.allow_synthetic, require_run=False
    )
    if not synthetic and not (args.daily_oof.parent / "run.json").is_file():
        raise ValueError("Real daily OOF predictions require adjacent run.json provenance")
    table, per_plant, curve = evaluate(
        read_table(args.daily_oof),
        read_table(args.blind_scores),
        args.control,
        args.seed,
        args.reps,
    )
    table.to_csv(args.out / "h4_lead_time.csv", index=False)
    per_plant.to_csv(args.out / "h4_per_plant.csv", index=False)
    curve.to_csv(args.out / "h4_survival.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, group) in enumerate(curve.groupby("class")):
        ax.step(
            group.day,
            group.not_yet_alerted,
            where="post",
            label=name,
            color=PALETTE[i % len(PALETTE)],
        )
    ax.set(
        xlabel="Day from treatment",
        ylabel="Fraction not yet alerted",
        ylim=(0, 1.05),
        title="Sustained model alerts (plant level)",
    )
    ax.legend()
    save_figure(fig, args.out / "h4_lead_time", synthetic)
    passed = table.median_lead_days.ge(2).all()
    write_summary(
        args.out / "h4_summary.md",
        "H4 lead time",
        f"Median lead time is symptom day minus first sustained alert day; positive means earlier model alert. "
        f"All class medians meet ≥2 days: {passed}. Missing alerts and symptoms remain censored and are counted. "
        "Bootstrap intervals condition on plants with both observed events; see per-plant table.",
        synthetic,
    )


if __name__ == "__main__":
    main()
