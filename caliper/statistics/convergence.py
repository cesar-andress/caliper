"""Convergence analysis: stability of conclusions vs sample size."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from caliper.ranking.aggregate import aggregate_scores_by_model, rank_models
from caliper.ranking.metrics import kendall_tau_between_rankings
from caliper.statistics.bootstrap import bootstrap_ci
from caliper.statistics.gtheory import estimate_g_variance_components
from caliper.statistics.robust_analysis import run_anova

DEFAULT_SUBSET_SIZES = (100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000)


def _cohens_d_top_bottom(df: pd.DataFrame, factor: str, value_col: str = "metric_value") -> float:
    if factor not in df.columns or df[factor].nunique() < 2:
        return float("nan")
    means = df.groupby(factor, observed=True)[value_col].mean().sort_values()
    low = df.loc[df[factor] == means.index[0], value_col].to_numpy(dtype=float)
    high = df.loc[df[factor] == means.index[-1], value_col].to_numpy(dtype=float)
    if len(low) < 2 or len(high) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(low) - 1) * np.var(low, ddof=1) + (len(high) - 1) * np.var(high, ddof=1))
        / (len(low) + len(high) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((np.mean(high) - np.mean(low)) / pooled)


def analyze_convergence(
    df: pd.DataFrame,
    *,
    subset_sizes: tuple[int, ...] = DEFAULT_SUBSET_SIZES,
    seed: int = 42,
    value_col: str = "metric_value",
) -> pd.DataFrame:
    """Evaluate statistical summaries on increasing random subsets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    full_scores = aggregate_scores_by_model(df, value_col=value_col)
    full_ranks = rank_models(full_scores)
    full_var = estimate_g_variance_components(
        df,
        [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in df.columns],
        value_col=value_col,
    ).components

    rows: list[dict[str, Any]] = []
    for n in subset_sizes:
        n_eff = min(n, len(df))
        subset = df.iloc[indices[:n_eff]].copy()
        scores = aggregate_scores_by_model(subset, value_col=value_col)
        ranks = rank_models(scores)
        common = full_ranks.index.intersection(ranks.index)
        if len(common) < 2:
            tau = float("nan")
        else:
            tau = kendall_tau_between_rankings(full_ranks.loc[common], ranks.loc[common])
        var = estimate_g_variance_components(
            subset,
            [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in subset.columns],
            value_col=value_col,
        ).components
        ci = bootstrap_ci(subset[value_col], n_bootstrap=500, seed=seed + n).as_dict()
        try:
            anova = run_anova(subset, anova_type=2)
            top_p = float(anova.loc[anova["partial_eta_squared"].idxmax(), "partial_eta_squared"])
        except Exception:
            top_p = float("nan")

        rows.append(
            {
                "n_observations": n_eff,
                "kendall_tau_vs_full_ranking": tau,
                "model_variance": var.get("model", 0.0),
                "prompt_variance": var.get("prompt_id", 0.0),
                "task_variance": var.get("task_id", 0.0),
                "residual_variance": var.get("residual", 0.0),
                "explained_variance_model_pct": 100.0 * var.get("model", 0.0) / max(var.get("total", 1e-12), 1e-12),
                "explained_variance_prompt_pct": 100.0 * var.get("prompt_id", 0.0) / max(var.get("total", 1e-12), 1e-12),
                "mean_metric_ci_lower": ci["lower"],
                "mean_metric_ci_upper": ci["upper"],
                "cohens_d_model": _cohens_d_top_bottom(subset, "model", value_col=value_col),
                "cohens_d_prompt": _cohens_d_top_bottom(subset, "prompt_id", value_col=value_col),
                "max_partial_eta_squared_type2": top_p,
            }
        )
    return pd.DataFrame(rows)
