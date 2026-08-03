"""Task-sampling comparison between subset and full HumanEval+ confirmatory studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from caliper.runners.experiment_paths import resolve_experiment_dir
from caliper.statistics.gtheory import (
    estimate_g_variance_components,
    simulate_d_study,
)
from caliper.statistics.glmm_analysis import run_pass_fail_glmm_analysis
from caliper.statistics.prepare import load_analysis_frame

DEFAULT_TASK_COUNTS = [10, 20, 40, 60, 80, 100, 120, 140, 164]
DEFAULT_N_SUBSETS = 1000
DEFAULT_SEED = 20260404
KENDALL_TARGETS = (0.90, 0.95)
G_TARGETS = (0.80, 0.90)


@dataclass(frozen=True)
class TaskSamplingPaths:
    root: Path
    csv: Path
    parquet: Path
    figures: Path
    latex: Path
    summary: Path

    @classmethod
    def from_experiment(cls, full_experiment_dir: Path) -> TaskSamplingPaths:
        root = full_experiment_dir / "paper1_analysis" / "task_sampling"
        return cls(
            root=root,
            csv=root / "csv",
            parquet=root / "parquet",
            figures=root / "figures",
            latex=root / "latex",
            summary=root / "summary",
        )

    def ensure(self) -> None:
        for path in (self.root, self.csv, self.parquet, self.figures, self.latex, self.summary):
            path.mkdir(parents=True, exist_ok=True)


def _load_frame(experiment_dir: Path, metric: str = "pass_at_1") -> pd.DataFrame:
    experiment_dir = resolve_experiment_dir(experiment_dir)
    return load_analysis_frame(
        experiment_dir,
        metric_name=metric,
        require_statistical_dataset=True,
    )


def _model_means(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("model", observed=True)["metric_value"].mean().sort_values(ascending=False)


def _kendall_tau(reference: pd.Series, sample: pd.Series) -> float:
    from scipy.stats import kendalltau

    aligned = reference.to_frame("ref").join(sample.to_frame("sample"), how="inner")
    if len(aligned) < 2:
        return float("nan")
    tau, _ = kendalltau(aligned["ref"].rank(ascending=False), aligned["sample"].rank(ascending=False))
    return float(tau)


def _pairwise_reversal_rate(reference: pd.Series, sample: pd.Series) -> float:
    models = list(reference.index)
    reversals = 0
    pairs = 0
    for i, left in enumerate(models):
        for right in models[i + 1 :]:
            pairs += 1
            ref_order = reference[left] >= reference[right]
            sample_order = sample[left] >= sample[right]
            if ref_order != sample_order:
                reversals += 1
    return reversals / pairs if pairs else float("nan")


def _prompt_effect(frame: pd.DataFrame) -> float:
    if "prompt_id" not in frame.columns:
        return float("nan")
    return float(frame.groupby("prompt_id", observed=True)["metric_value"].mean().std(ddof=0) or 0.0)


def _g_coefficient_for_tasks(frame: pd.DataFrame, n_tasks: int) -> float:
    """Deprecated: current D-study G estimator ignores design sample sizes.

    Returns NaN so task-sampling reliability recommendations never use defective G.
    See paper1_analysis/final/summary/gtheory_validity_assessment.md.
    """
    del frame, n_tasks
    return float("nan")


def compare_point_estimates(
    subset_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compare aggregate estimates between subset and full benchmark runs."""
    subset_means = _model_means(subset_frame)
    full_means = _model_means(full_frame)
    rows: list[dict[str, Any]] = []
    for model in full_means.index:
        rows.append(
            {
                "model": model,
                "subset_pass_at_1": float(subset_means.get(model, np.nan)),
                "full_pass_at_1": float(full_means.get(model, np.nan)),
                "absolute_difference": float(full_means.get(model, np.nan) - subset_means.get(model, np.nan)),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["subset_rank"] = comparison["subset_pass_at_1"].rank(ascending=False, method="min")
    comparison["full_rank"] = comparison["full_pass_at_1"].rank(ascending=False, method="min")
    comparison["rank_difference"] = comparison["full_rank"] - comparison["subset_rank"]
    comparison.loc[len(comparison)] = {
        "model": "__overall__",
        "subset_pass_at_1": float(subset_frame["metric_value"].mean()),
        "full_pass_at_1": float(full_frame["metric_value"].mean()),
        "absolute_difference": float(full_frame["metric_value"].mean() - subset_frame["metric_value"].mean()),
        "subset_rank": np.nan,
        "full_rank": np.nan,
        "rank_difference": np.nan,
    }
    return comparison


def _glmm_model_coefficients(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if "pass_fail" not in frame.columns:
        working = frame.copy()
        working["pass_fail"] = (working["metric_value"] >= 0.5).astype(int)
    else:
        working = frame
    result = run_pass_fail_glmm_analysis(working, metric="pass_at_1")
    table = result.coefficients.copy()
    table["study"] = "provided"
    return table


def simulate_task_subsets(
    full_frame: pd.DataFrame,
    *,
    task_counts: list[int] | None = None,
    n_subsets: int = DEFAULT_N_SUBSETS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Repeated random task subsets to quantify sampling stability.

    Uses pre-aggregated model×task means (average over prompt, temperature, run)
    so B×|grid| resampling remains tractable on the full confirmatory frame.
    G coefficients are intentionally NaN (defective D-study estimator).
    """
    task_counts = task_counts or DEFAULT_TASK_COUNTS
    rng = np.random.default_rng(seed)
    mt = (
        full_frame.groupby(["model", "task_id"], observed=True)["metric_value"]
        .mean()
        .unstack("task_id")
    )
    all_tasks = list(mt.columns)
    full_means = mt.mean(axis=1).sort_values(ascending=False)
    # Prompt effect on the full frame only (constant across subsets in this export).
    prompt_std = _prompt_effect(full_frame)
    rows: list[dict[str, Any]] = []

    for n_tasks in task_counts:
        effective_n = min(n_tasks, len(all_tasks))
        for subset_index in range(n_subsets):
            sampled_tasks = rng.choice(all_tasks, size=effective_n, replace=False)
            sample_means = mt[sampled_tasks].mean(axis=1)
            rows.append(
                {
                    "n_tasks": effective_n,
                    "subset_index": subset_index,
                    "kendall_tau": _kendall_tau(full_means, sample_means),
                    "pairwise_reversal_rate": _pairwise_reversal_rate(full_means, sample_means),
                    "prompt_effect_std": prompt_std,
                    "overall_pass_at_1": float(sample_means.mean()),
                    "g_coefficient_task": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def summarize_reliability_thresholds(simulation: pd.DataFrame) -> pd.DataFrame:
    """Recommend minimum task counts for reliability targets."""
    rows: list[dict[str, Any]] = []
    for target in KENDALL_TARGETS:
        for n_tasks, group in simulation.groupby("n_tasks"):
            fraction = float((group["kendall_tau"] >= target).mean())
            rows.append(
                {
                    "metric": "kendall_tau",
                    "target": target,
                    "n_tasks": int(n_tasks),
                    "fraction_meeting_target": fraction,
                }
            )
    for target in G_TARGETS:
        for n_tasks, group in simulation.groupby("n_tasks"):
            fraction = float((group["g_coefficient_task"] >= target).mean())
            rows.append(
                {
                    "metric": "g_coefficient_task",
                    "target": target,
                    "n_tasks": int(n_tasks),
                    "fraction_meeting_target": fraction,
                }
            )

    recommendations: list[dict[str, Any]] = []
    for metric in ("kendall_tau", "g_coefficient_task"):
        for target in (KENDALL_TARGETS if metric == "kendall_tau" else G_TARGETS):
            metric_rows = [row for row in rows if row["metric"] == metric and row["target"] == target]
            minimum = None
            for row in sorted(metric_rows, key=lambda item: item["n_tasks"]):
                if row["fraction_meeting_target"] >= 0.80:
                    minimum = row["n_tasks"]
                    break
            recommendations.append(
                {
                    "metric": metric,
                    "target": target,
                    "recommended_min_tasks_80pct_subsets": minimum,
                }
            )
    return pd.DataFrame(recommendations)


def _rank_uncertainty_table(simulation: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset_40 = simulation[simulation["n_tasks"] == 40]
    if subset_40.empty:
        return pd.DataFrame()
    for n_tasks, group in simulation.groupby("n_tasks"):
        row: dict[str, Any] = {"n_tasks": int(n_tasks)}
        for model in models:
            # rank variability approximated via pass@1 std in subsets - placeholder uses overall metric only
            row[f"{model}_pass_at_1_std"] = float(group["overall_pass_at_1"].std(ddof=1))
        row["kendall_tau_median"] = float(group["kendall_tau"].median())
        row["kendall_tau_p05"] = float(group["kendall_tau"].quantile(0.05))
        row["kendall_tau_p95"] = float(group["kendall_tau"].quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def _save_figures(
    paths: TaskSamplingPaths,
    simulation: pd.DataFrame,
    comparison: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    if not simulation.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = simulation.groupby("n_tasks")["kendall_tau"].median()
        ax.plot(grouped.index, grouped.values, marker="o")
        ax.axhline(0.90, linestyle="--", color="gray", label="tau=0.90")
        ax.axhline(0.95, linestyle="--", color="black", label="tau=0.95")
        ax.set_xlabel("Number of tasks")
        ax.set_ylabel("Median Kendall tau vs full benchmark")
        ax.set_title("Ranking stability versus number of tasks")
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figures / "fig_ranking_stability_vs_tasks.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        grouped_g = simulation.groupby("n_tasks")["g_coefficient_task"].median()
        ax.plot(grouped_g.index, grouped_g.values, marker="o", color="#C44E52")
        ax.axhline(0.80, linestyle="--", color="gray", label="G=0.80")
        ax.axhline(0.90, linestyle="--", color="black", label="G=0.90")
        ax.set_xlabel("Number of tasks")
        ax.set_ylabel("Median G coefficient (task facet)")
        ax.set_title("G coefficient versus number of tasks")
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figures / "fig_g_coefficient_vs_tasks.pdf")
        plt.close(fig)

        subset_40 = simulation[simulation["n_tasks"] == 40]
        if not subset_40.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(subset_40["overall_pass_at_1"], bins=30, color="#4C72B0", alpha=0.85)
            ax.set_xlabel("Overall pass@1")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of pass@1 across random 40-task subsets")
            fig.tight_layout()
            fig.savefig(paths.figures / "fig_pass_at_1_random_40_subsets.pdf")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        grouped_rev = simulation.groupby("n_tasks")["pairwise_reversal_rate"].median()
        ax.plot(grouped_rev.index, grouped_rev.values, marker="o", color="#55A868")
        ax.set_xlabel("Number of tasks")
        ax.set_ylabel("Median pairwise reversal rate")
        ax.set_title("Pairwise rank reversal probability versus number of tasks")
        fig.tight_layout()
        fig.savefig(paths.figures / "fig_pairwise_reversal_vs_tasks.pdf")
        plt.close(fig)

    if not comparison.empty and "__overall__" not in comparison["model"].values:
        pass
    if not comparison.empty:
        model_rows = comparison[comparison["model"] != "__overall__"]
        if not model_rows.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(model_rows["subset_pass_at_1"], model_rows["full_pass_at_1"])
            for _, row in model_rows.iterrows():
                ax.annotate(str(row["model"]), (row["subset_pass_at_1"], row["full_pass_at_1"]), fontsize=8)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("40-task subset pass@1")
            ax.set_ylabel("164-task full benchmark pass@1")
            ax.set_title("Coefficient drift: subset versus full benchmark")
            fig.tight_layout()
            fig.savefig(paths.figures / "fig_coefficient_drift_subset_vs_full.pdf")
            plt.close(fig)


def run_task_sampling_analysis(
    full_experiment_dir: Path | str,
    subset_experiment_dir: Path | str,
    *,
    metric: str = "pass_at_1",
    task_counts: list[int] | None = None,
    n_subsets: int = DEFAULT_N_SUBSETS,
    seed: int = DEFAULT_SEED,
) -> TaskSamplingPaths:
    """Run subset-vs-full task sampling analysis and export publication artifacts."""
    full_dir = resolve_experiment_dir(full_experiment_dir)
    subset_dir = resolve_experiment_dir(subset_experiment_dir)
    paths = TaskSamplingPaths.from_experiment(full_dir)
    paths.ensure()

    full_frame = _load_frame(full_dir, metric=metric)
    subset_frame = _load_frame(subset_dir, metric=metric)

    comparison = compare_point_estimates(subset_frame, full_frame)
    simulation = simulate_task_subsets(
        full_frame,
        task_counts=task_counts,
        n_subsets=n_subsets,
        seed=seed,
    )
    recommendations = summarize_reliability_thresholds(simulation)
    rank_uncertainty = _rank_uncertainty_table(simulation, models=sorted(full_frame["model"].unique()))

    subset_glmm = _glmm_model_coefficients(subset_frame)
    full_glmm = _glmm_model_coefficients(full_frame)
    if not subset_glmm.empty:
        subset_glmm["study"] = "subset_40"
    if not full_glmm.empty:
        full_glmm["study"] = "full_164"
    glmm_compare = pd.concat([subset_glmm, full_glmm], ignore_index=True)

    comparison.to_csv(paths.csv / "table_subset_vs_full_estimates.csv", index=False)
    recommendations.to_csv(paths.csv / "table_recommended_task_counts.csv", index=False)
    rank_uncertainty.to_csv(paths.csv / "table_model_rank_uncertainty_by_task_count.csv", index=False)
    simulation.to_csv(paths.csv / "task_subset_simulation_summary.csv", index=False)
    glmm_compare.to_csv(paths.csv / "table_glmm_subset_vs_full.csv", index=False)
    simulation.to_parquet(paths.parquet / "task_subset_simulation.parquet", index=False)

    _save_figures(paths, simulation, comparison, recommendations)

    def _md_table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except ImportError:
            return frame.to_string(index=False)

    summary_lines = [
        "# Task sampling analysis",
        "",
        f"- Full experiment: `{full_dir}`",
        f"- Subset experiment: `{subset_dir}`",
        f"- Subset simulations: {n_subsets} random draws per task count",
        f"- Seed: {seed}",
        f"- G coefficients: omitted (invalid D-study estimator)",
        "",
        "## 40-task versus 164-task estimates",
        "",
        _md_table(comparison),
        "",
        "## Recommended task counts",
        "",
        _md_table(recommendations),
        "",
    ]
    (paths.summary / "task_sampling_report.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (paths.root / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "full_experiment_dir": str(full_dir),
                "subset_experiment_dir": str(subset_dir),
                "n_subsets": n_subsets,
                "task_counts": task_counts or DEFAULT_TASK_COUNTS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
