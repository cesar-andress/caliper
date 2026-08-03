#!/usr/bin/env python3
"""Run subset-vs-full task sampling analysis for Paper 1."""

from __future__ import annotations

import argparse
from pathlib import Path

from caliper.statistics.task_sampling import run_task_sampling_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 task-sampling comparison analysis")
    parser.add_argument(
        "--full-experiment",
        type=Path,
        required=True,
        help="Completed 164-task HumanEval+ experiment directory",
    )
    parser.add_argument(
        "--subset-experiment",
        type=Path,
        required=True,
        help="Completed 40-task HumanEval+ experiment directory",
    )
    parser.add_argument("--metric", default="pass_at_1")
    parser.add_argument("--n-subsets", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260404)
    args = parser.parse_args()
    paths = run_task_sampling_analysis(
        args.full_experiment,
        args.subset_experiment,
        metric=args.metric,
        n_subsets=args.n_subsets,
        seed=args.seed,
    )
    print(f"Task-sampling analysis written to {paths.root}")


if __name__ == "__main__":
    main()
