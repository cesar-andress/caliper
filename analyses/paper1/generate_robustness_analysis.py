#!/usr/bin/env python3
"""Generate Paper 1 robustness analysis outputs from a completed pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from caliper.statistics.robustness_report import run_robustness_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 robustness analysis")
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/paper1_ollama_pilot"))
    parser.add_argument("--metric", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    out = run_robustness_analysis(args.experiment_dir, metric=args.metric, n_bootstrap=args.n_bootstrap)
    print(f"Robustness analysis written to {out}")


if __name__ == "__main__":
    main()
