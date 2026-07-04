"""Descriptive statistics by experimental factor."""

from __future__ import annotations

import pandas as pd


def descriptive_by_factor(
    df: pd.DataFrame,
    factor_col: str,
    *,
    value_col: str = "metric_value",
) -> pd.DataFrame:
    """Compute descriptive statistics grouped by one factor."""
    grouped = df.groupby(factor_col, observed=True)[value_col]
    summary = grouped.agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        q25=lambda s: s.quantile(0.25),
        median="median",
        q75=lambda s: s.quantile(0.75),
        max="max",
    )
    summary["sem"] = summary["std"] / summary["count"].pow(0.5)
    return summary.reset_index()


def descriptive_all_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    value_col: str = "metric_value",
) -> dict[str, pd.DataFrame]:
    """Return descriptive tables for each factor."""
    return {
        factor: descriptive_by_factor(df, factor, value_col=value_col)
        for factor in factor_cols
        if factor in df.columns
    }
