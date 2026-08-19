"""Normalize results tables for Paper 1 statistical analysis."""

from __future__ import annotations

from pathlib import Path

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


def completed_rows_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep terminal completed rows; drop append-only historical failures.

    When ``cell_id`` is present and duplicates exist, retain the **last** completed
    row per cell (append-only recovery / retry history).
    """
    if df.empty:
        return df.copy()
    out = df
    if "status" in out.columns:
        # Include budget_exhausted: visible response empty after shared
        # thinking+response budget; still a terminal scored outcome (pass@1=0).
        out = out[out["status"].isin(["completed", "budget_exhausted"])].copy()
    if "cell_id" in out.columns and out["cell_id"].duplicated().any():
        out = out.drop_duplicates(subset=["cell_id"], keep="last").copy()
    return out.reset_index(drop=True)


def load_analysis_frame(
    experiment_dir: Path | str,
    *,
    metric_name: str | None = None,
    require_statistical_dataset: bool = True,
) -> pd.DataFrame:
    """Load the Paper 1 analysis frame from the frozen statistical dataset.

    Prefers ``statistical_dataset.parquet`` (completed-only, analysis schema).
    Falls back to ``results.parquet`` only when ``require_statistical_dataset`` is
    False, and then filters to completed rows with latest-per-``cell_id`` semantics
    so historical ``failed`` append-only rows cannot enter confirmatory analyses.
    """
    experiment_dir = Path(experiment_dir)
    stats_path = experiment_dir / "statistical_dataset.parquet"
    results_path = experiment_dir / "results.parquet"

    if stats_path.exists():
        raw = pd.read_parquet(stats_path)
        source = "statistical_dataset.parquet"
    elif require_statistical_dataset:
        msg = (
            f"Frozen analysis requires {stats_path}; refusing to read append-only "
            "results.jsonl/results.parquet for confirmatory statistics."
        )
        raise FileNotFoundError(msg)
    elif results_path.exists():
        raw = completed_rows_only(pd.read_parquet(results_path))
        source = "results.parquet(completed_latest)"
    else:
        msg = f"No statistical_dataset.parquet or results.parquet under {experiment_dir}"
        raise FileNotFoundError(msg)

    if source.startswith("statistical_dataset") and "status" in raw.columns:
        raw = completed_rows_only(raw)

    frame = prepare_results_table(raw, metric_name=metric_name)
    if "run_index" in frame.columns and frame.get("run_id", pd.Series(dtype=object)).nunique(
        dropna=False
    ) <= 1:
        frame = frame.copy()
        frame["run_id"] = frame["run_index"]
    frame.attrs["analysis_source"] = source
    return frame
