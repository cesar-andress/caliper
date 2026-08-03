#!/usr/bin/env python3
"""Paper 1 variance decomposition analysis script."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from caliper.statistics.bootstrap import bootstrap_ci_by_factor
from caliper.statistics.descriptive import descriptive_all_factors
from caliper.statistics.gtheory import estimate_g_variance_components, simulate_d_study_grid
from caliper.statistics.mixed_effects import fit_mixed_model
from caliper.statistics.prepare import completed_rows_only, prepare_results_table
from caliper.statistics.variance import decompose_variance


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 variance decomposition")
    parser.add_argument("--results", required=True, type=Path, help="Results parquet/csv/jsonl")
    parser.add_argument("--metric", default=None, help="Metric name (defaults to config primary_metric)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment YAML (defaults to config.yaml beside results)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Write summary tables here")
    args = parser.parse_args()

    suffix = args.results.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(args.results)
    elif suffix == ".jsonl":
        raw = pd.read_json(args.results, orient="records", lines=True)
    else:
        raw = pd.read_csv(args.results)

    # Guard against append-only historical failed rows in raw results dumps.
    raw = completed_rows_only(raw)

    from caliper.config.metrics import resolve_analysis_metric

    resolved_metric, _ = resolve_analysis_metric(
        metric=args.metric,
        results_path=args.results,
        config_path=args.config,
    )
    df = prepare_results_table(raw, metric_name=resolved_metric)
    facets = [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in df.columns]

    print("=== Descriptive Statistics ===")
    for factor, table in descriptive_all_factors(df, facets).items():
        print(f"\n-- {factor} --")
        print(table.to_string(index=False))

    print("\n=== Variance Decomposition (sequential ANOVA) ===")
    vc = decompose_variance(df)
    for key, value in vc.as_dict().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n=== Bootstrap 95% CI by Model ===")
    if "model" in df.columns:
        print(bootstrap_ci_by_factor(df, "model").to_string(index=False))

    print("\n=== Mixed Model (task random intercept) ===")
    mm = fit_mixed_model(df, group_col="task_id")
    print(f"  method: {mm.method}")
    print(f"  converged: {mm.converged}")
    for note in mm.notes:
        print(f"  note: {note}")

    print("\n=== G-Study ===")
    gstudy = estimate_g_variance_components(df, facets)
    print(f"  components: {gstudy.components}")

    print("\n=== D-Study Grid (tasks × prompts × runs) ===")
    dgrid = simulate_d_study_grid(
        gstudy.components,
        task_counts=[1, 3, 5, 10],
        prompt_counts=[1, 2, 3],
        run_counts=[1, 3, 5],
        universe_facets=["task_id", "prompt_id"],
    )
    print(dgrid.to_string(index=False))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dgrid.to_csv(args.output_dir / "d_study_grid.csv", index=False)
        vc_df = pd.DataFrame([vc.as_dict()])
        vc_df.to_csv(args.output_dir / "variance_components.csv", index=False)
        print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
