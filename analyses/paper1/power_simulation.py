#!/usr/bin/env python3
"""Paper 1 power simulation analysis script."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from caliper.statistics.gtheory import estimate_g_variance_components
from caliper.statistics.power import compute_power
from caliper.statistics.power_sim import simulate_power_grid
from caliper.statistics.prepare import prepare_results_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 power simulation")
    parser.add_argument("--results", required=True, type=Path, help="Results parquet/csv/jsonl")
    parser.add_argument("--metric", default=None, help="Metric name (defaults to config primary_metric)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment YAML (defaults to config.yaml beside results)",
    )
    parser.add_argument("--effect-size", type=float, default=0.05, help="Model mean difference")
    parser.add_argument("--simulations", type=int, default=500, help="Monte Carlo replications")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write power grid CSV here")
    args = parser.parse_args()

    suffix = args.results.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(args.results)
    elif suffix == ".jsonl":
        raw = pd.read_json(args.results, orient="records", lines=True)
    else:
        raw = pd.read_csv(args.results)

    from caliper.config.metrics import resolve_analysis_metric

    resolved_metric, _ = resolve_analysis_metric(
        metric=args.metric,
        results_path=args.results,
        config_path=args.config,
    )
    df = prepare_results_table(raw, metric_name=resolved_metric)
    gstudy = estimate_g_variance_components(df)
    components = gstudy.components

    print("=== Analytic Power (two-sample t-test reference) ===")
    for n in (10, 20, 50, 100):
        result = compute_power(args.effect_size, n)
        print(f"  n={n:3d}  power={result.power:.3f}")

    print("\n=== Simulated Power Grid ===")
    grid = simulate_power_grid(
        components,
        effect_size=args.effect_size,
        task_counts=[3, 5, 10],
        prompt_counts=[1, 2, 3],
        run_counts=[1, 3, 5],
        n_simulations=args.simulations,
    )
    print(grid.to_string(index=False))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.output_dir / "power_grid.csv", index=False)
        print(f"\nWrote {args.output_dir / 'power_grid.csv'}")


if __name__ == "__main__":
    main()
