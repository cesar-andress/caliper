"""Ranking comparison metrics for Paper 2."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def kendall_tau_between_rankings(
    baseline_ranks: pd.Series,
    resampled_ranks: pd.Series,
) -> float:
    """Kendall tau between two model rank vectors (aligned by model index)."""
    aligned = baseline_ranks.align(resampled_ranks, join="inner")
    if len(aligned[0]) < 2:
        return 1.0
    tau, _ = kendalltau(aligned[0].values, aligned[1].values)
    return float(tau) if not np.isnan(tau) else 1.0


def rank_change_rate(baseline_ranks: pd.Series, resampled_ranks: pd.Series) -> float:
    """Fraction of models whose integer rank changed."""
    aligned = baseline_ranks.align(resampled_ranks, join="inner")
    base_int = aligned[0].rank(ascending=False, method="first")
    res_int = aligned[1].rank(ascending=False, method="first")
    if len(base_int) == 0:
        return 0.0
    return float((base_int != res_int).mean())


def ranking_fragility_index(kendall_taus: list[float]) -> float:
    """Convert mean Kendall tau to a fragility index in [0, 1].

    0 = perfectly stable rankings, 1 = maximally fragile (tau → -1).
    """
    if not kendall_taus:
        return 0.0
    mean_tau = float(np.mean(kendall_taus))
    return float((1 - mean_tau) / 2)


def rank_probability_matrix(
    bootstrap_ranks: pd.DataFrame,
    *,
    model_col: str = "model",
    rank_col: str = "rank",
    n_ranks: int | None = None,
) -> pd.DataFrame:
    """Estimate P(model occupies rank k) from bootstrap samples."""
    models = sorted(bootstrap_ranks[model_col].unique())
    if n_ranks is None:
        n_ranks = len(models)

    counts = (
        bootstrap_ranks.groupby([model_col, rank_col], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    probs = counts.div(counts.sum(axis=1), axis=0)

    for rank in range(1, n_ranks + 1):
        if rank not in probs.columns:
            probs[rank] = 0.0
    probs = probs[sorted(probs.columns)]
    probs = probs.reindex(models, fill_value=0.0)
    probs.index.name = model_col
    probs.columns.name = "rank"
    return probs


def pairwise_reversal_probability(
    bootstrap_ranks: pd.DataFrame,
    baseline_scores: pd.Series,
    *,
    model_col: str = "model",
    rank_col: str = "rank",
    iteration_col: str = "iteration",
) -> pd.DataFrame:
    """Estimate P(model A beats B in baseline but not in bootstrap)."""
    models = list(baseline_scores.index)
    baseline_order = baseline_scores.sort_values(ascending=False).index.tolist()

    group_cols = [iteration_col]
    if "bootstrap_type" in bootstrap_ranks.columns:
        group_cols.append("bootstrap_type")

    rows: list[dict[str, float | str]] = []
    for model_a, model_b in itertools.combinations(models, 2):
        baseline_a_beats_b = baseline_order.index(model_a) < baseline_order.index(model_b)
        reversals = 0
        total = 0

        for _, group in bootstrap_ranks.groupby(group_cols, observed=True):
            ranks = group.set_index(model_col)[rank_col]
            if model_a not in ranks.index or model_b not in ranks.index:
                continue
            rank_a = float(ranks.loc[model_a])
            rank_b = float(ranks.loc[model_b])
            total += 1
            bootstrap_a_beats_b = rank_a < rank_b
            if baseline_a_beats_b and not bootstrap_a_beats_b:
                reversals += 1
            elif not baseline_a_beats_b and bootstrap_a_beats_b:
                reversals += 1

        prob = reversals / total if total > 0 else 0.0
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "reversal_probability": prob,
                "n_iterations": total,
            }
        )

    return pd.DataFrame(rows)
