"""Aggregate benchmark scores into model-level summaries and matrices."""

from __future__ import annotations

import pandas as pd


def aggregate_scores_by_model(
    df: pd.DataFrame,
    *,
    value_col: str = "metric_value",
    model_col: str = "model",
) -> pd.Series:
    """Return mean score per model."""
    return df.groupby(model_col, observed=True)[value_col].mean().sort_values(ascending=False)


def rank_models(scores: pd.Series) -> pd.Series:
    """Rank models by score (rank 1 = best/highest score)."""
    return scores.rank(ascending=False, method="average")


def build_score_matrix(
    df: pd.DataFrame,
    *,
    row_col: str,
    model_col: str = "model",
    value_col: str = "metric_value",
) -> pd.DataFrame:
    """Build a row × model score matrix (e.g. tasks × models)."""
    matrix = df.pivot_table(
        index=row_col,
        columns=model_col,
        values=value_col,
        aggfunc="mean",
    )
    return matrix.sort_index()
