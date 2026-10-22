"""Run a pre-specified plant-level experiment from YAML."""

from __future__ import annotations

import argparse

from leafy_ai.experiment import load_experiment, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Experiment YAML")
    parser.add_argument("--out", default="runs", help="Ignored run root")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="First seed and two search trials; software checks only",
    )
    args = parser.parse_args()
    print(run_experiment(load_experiment(args.config), args.out, quick=args.quick))


if __name__ == "__main__":
    main()
