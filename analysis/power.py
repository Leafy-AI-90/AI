"""Connor (1987) normal-approximation planning for paired McNemar discordance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import PALETTE, prepare_report, read_table, save_figure, write_summary
from scipy.stats import norm


def connor_pairs(p12: float, p21: float, alpha: float = 0.05, power: float = 0.8) -> int | float:
    """Approximate required paired plants from two discordant proportions.

    n = [z_(1-alpha/2)*sqrt(p_d) + z_power*sqrt(p_d-delta²)]²/delta²,
    with p_d=p12+p21 and delta=p21-p12. This is a planning estimate;
    exact-binomial operating characteristics should be checked for small n.
    """
    if min(p12, p21) < 0 or p12 + p21 > 1 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("Invalid discordance, alpha, or power")
    delta = abs(p21 - p12)
    if delta == 0:
        return math.inf
    total = p12 + p21
    numerator = norm.ppf(1 - alpha / 2) * np.sqrt(total)
    numerator += norm.ppf(power) * np.sqrt(total - delta**2)
    return int(np.ceil((numerator / delta) ** 2))


def evaluate(
    p12: float,
    p21: float,
    alpha: float,
    power: float,
    planned_plants: int,
    pilot_n: int | None = None,
) -> pd.DataFrame:
    """One planning row, optionally recalculated from observed pilot discordance."""
    required = connor_pairs(p12, p21, alpha, power)
    return pd.DataFrame(
        [
            {
                "p12": p12,
                "p21": p21,
                "discordance": p12 + p21,
                "difference": abs(p21 - p12),
                "alpha": alpha,
                "power": power,
                "required_paired_plants": required,
                "planned_plants": planned_plants,
                "pilot_n": pilot_n,
                "planned_sufficient": planned_plants >= required,
            }
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p12", type=float, help="Proportion only first model correct")
    parser.add_argument("--p21", type=float, help="Proportion only second model correct")
    parser.add_argument(
        "--pilot", type=Path, help="Paired pilot CSV with plant_id, first_correct, second_correct"
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--planned-plants", type=int, default=360)
    parser.add_argument("--out", type=Path, default=Path("reports/power"))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    if args.pilot:
        synthetic = prepare_report([args.pilot], args.out, args.allow_synthetic)
        pilot = read_table(args.pilot)
        if {"plant_id", "first_correct", "second_correct"} - set(
            pilot
        ) or pilot.plant_id.duplicated().any():
            raise ValueError("Pilot needs one paired correctness observation per plant")
        if pilot[["plant_id", "first_correct", "second_correct"]].isna().any().any():
            raise ValueError("Pilot correctness and plant IDs cannot be missing")
        if (
            not pilot.first_correct.isin([True, False, 0, 1]).all()
            or not pilot.second_correct.isin([True, False, 0, 1]).all()
        ):
            raise ValueError("Pilot correctness must be boolean or 0/1")
        first, second = pilot.first_correct.astype(bool), pilot.second_correct.astype(bool)
        p12 = float((first & ~second).mean())
        p21 = float((~first & second).mean())
        pilot_n = len(pilot)
    else:
        if args.p12 is None or args.p21 is None:
            parser.error("Supply --pilot or both --p12 and --p21")
        synthetic, p12, p21, pilot_n = False, args.p12, args.p21, None
        args.out.mkdir(parents=True, exist_ok=True)
    table = evaluate(p12, p21, args.alpha, args.power, args.planned_plants, pilot_n)
    table.to_csv(args.out / "power.csv", index=False)
    difference = abs(p21 - p12)
    scales = np.linspace(0.5, 1.5, 50)
    values = [
        connor_pairs(p12 * scale, p21 * scale, args.alpha, args.power)
        if (p12 + p21) * scale <= 1
        else np.nan
        for scale in scales
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scales * (p12 + p21), values, color=PALETTE[0])
    ax.axhline(args.planned_plants, color=PALETTE[1], linestyle="--", label="Planned plants")
    ax.set(
        xlabel="Assumed discordant proportion (fixed discordant odds ratio)",
        ylabel="Required paired plants",
        title="McNemar sample-size sensitivity",
    )
    ax.legend()
    save_figure(fig, args.out / "power", synthetic)
    row = table.iloc[0]
    write_summary(
        args.out / "power_summary.md",
        "Paired comparison power",
        f"Connor (1987) approximation: {row.required_paired_plants} paired plants "
        f"for α={args.alpha:.2f}, power={args.power:.2f}, discordant proportions "
        f"{p12:.3f} and {p21:.3f} (absolute difference {difference:.3f}). "
        "Pilot rates are planning assumptions, not a model result. "
        "Source: https://pubmed.ncbi.nlm.nih.gov/3567305/",
        synthetic,
    )


if __name__ == "__main__":
    main()
