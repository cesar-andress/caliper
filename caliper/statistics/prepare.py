"""Normalize results tables for Paper 1 statistical analysis."""

from __future__ import annotations

import pandas as pd

STANDARD_COLUMNS = (
    "model",
    "task_id",
    "prompt_id",
    "run_id",
    "temperature",
    "metric_name",
    "metric_value",
)

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "model": ("model", "model_id"),
    "task_id": ("task_id", "task"),
    "prompt_id": ("prompt_id", "prompt_variant_id"),
    "run_id": ("run_id", "run_index"),
    "temperature": ("temperature",),
    "metric_name": ("metric_name", "metric"),
    "metric_value": ("metric_value", "score"),
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    for alias in COLUMN_ALIASES[canonical]:
        if alias in df.columns:
            return alias
    return None


def _score_from_dict(scores: object, metric_name: str) -> float | None:
    if not isinstance(scores, dict):
        return None
    value = scores.get(metric_name)
    if value is None:
        return None
    return float(value)


def prepare_results_table(
    df: pd.DataFrame,
    *,
    metric_name: str | None = None,
) -> pd.DataFrame:
    """Map experiment/evaluation results to the Paper 1 standard schema.

    Args:
        df: Raw results or evaluations DataFrame.
        metric_name: If provided, select this metric (from ``scores`` dict when present).

    Returns:
        Copy with canonical column names.

    Raises:
        ValueError: If required columns cannot be resolved.
    """
    out = df.copy()

    rename: dict[str, str] = {}
    for canonical in STANDARD_COLUMNS:
        if canonical in out.columns:
            continue
        source = _resolve_column(out, canonical)
        if source is not None:
            rename[source] = canonical

    out = out.rename(columns=rename)

    if metric_name is not None and "scores" in out.columns:
        extracted = out["scores"].apply(
            lambda scores: _score_from_dict(scores, metric_name),
        )
        if extracted.notna().any():
            out["metric_value"] = extracted
            out["metric_name"] = metric_name
            out = out[extracted.notna()].copy()

    missing = [col for col in ("model", "task_id", "metric_value") if col not in out.columns]
    if missing:
        msg = f"results table missing required columns: {', '.join(missing)}"
        raise ValueError(msg)

    if "metric_name" not in out.columns:
        out["metric_name"] = "score"
    if "prompt_id" not in out.columns:
        out["prompt_id"] = "default"
    if "run_id" not in out.columns:
        out["run_id"] = 0
    if "temperature" not in out.columns:
        out["temperature"] = 0.0

    if metric_name is not None:
        if "metric_name" in out.columns:
            out = out[out["metric_name"] == metric_name].copy()
        elif "scores" not in out.columns:
            metric_col = _resolve_column(out, "metric_name")
            if metric_col is not None:
                out = out[out[metric_col] == metric_name].copy()

    out["metric_value"] = out["metric_value"].astype(float)
    return out.reset_index(drop=True)
