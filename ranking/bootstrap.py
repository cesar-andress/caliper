"""Bootstrap resampling of model rankings over experimental facets."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from caliper.ranking.aggregate import aggregate_scores_by_model, rank_models
from caliper.ranking.metrics import kendall_tau_between_rankings, rank_change_rate

BootstrapFacet = Literal["task_id", "prompt_id", "run_id"]


def bootstrap_rankings(
    df: pd.DataFrame,
    facet_col: BootstrapFacet,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    value_col: str = "metric_value",
    model_col: str = "model",
) -> tuple[pd.DataFrame, pd.Series, list[float]]:
    """Bootstrap model rankings by resampling facet levels with replacement.

    Returns:
        Long-format bootstrap samples, baseline mean scores, list of Kendall taus.
    """
    if facet_col not in df.columns:
        msg = f"facet column '{facet_col}' not in results table"
        raise ValueError(msg)

    baseline_scores = aggregate_scores_by_model(df, value_col=value_col, model_col=model_col)
    baseline_ranks = rank_models(baseline_scores)

    facet_levels = df[facet_col].dropna().unique()
    if len(facet_levels) == 0:
        msg = f"no levels found for facet '{facet_col}'"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    taus: list[float] = []

    for iteration in range(n_bootstrap):
        sampled = rng.choice(facet_levels, size=len(facet_levels), replace=True)
        resampled = pd.concat(
            [df[df[facet_col] == level] for level in sampled],
            ignore_index=True,
        )
        scores = aggregate_scores_by_model(resampled, value_col=value_col, model_col=model_col)
        ranks = rank_models(scores)
        tau = kendall_tau_between_rankings(baseline_ranks, ranks)
        taus.append(tau)

        for model in scores.index:
            rows.append(
                {
                    "bootstrap_type": facet_col,
                    "iteration": iteration,
                    "model": model,
                    "score": float(scores[model]),
                    "rank": float(ranks[model]),
                    "kendall_tau": tau,
                    "rank_changed": float(
                        rank_change_rate(baseline_ranks, ranks) > 0,
                    ),
                }
            )

    samples = pd.DataFrame(rows)
    return samples, baseline_scores, taus


def bootstrap_all_facets(
    df: pd.DataFrame,
    *,
    facets: list[BootstrapFacet] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    value_col: str = "metric_value",
    model_col: str = "model",
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[float]]]:
    """Run bootstrap over tasks, runs, and prompts (when columns exist)."""
    if facets is None:
        facets = [f for f in ("task_id", "prompt_id", "run_id") if f in df.columns]

    all_samples: list[pd.DataFrame] = []
    taus_by_facet: dict[str, list[float]] = {}
    baseline_scores: pd.Series | None = None

    for offset, facet in enumerate(facets):
        samples, baseline, taus = bootstrap_rankings(
            df,
            facet,
            n_bootstrap=n_bootstrap,
            seed=seed + offset * 1000,
            value_col=value_col,
            model_col=model_col,
        )
        all_samples.append(samples)
        taus_by_facet[facet] = taus
        if baseline_scores is None:
            baseline_scores = baseline

    if baseline_scores is None:
        baseline_scores = aggregate_scores_by_model(df, value_col=value_col, model_col=model_col)

    combined = pd.concat(all_samples, ignore_index=True)
    return combined, baseline_scores, taus_by_facet
