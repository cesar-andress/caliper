"""Variance decomposition for LLM evaluation scores (Paper 1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VarianceComponents:
    """Decomposed variance attributable to each experimental factor."""

    total_variance: float
    model_variance: float
    prompt_variance: float
    task_variance: float
    run_variance: float
    temperature_variance: float = 0.0
    residual_variance: float = 0.0
    n_observations: int = 0
    method: str = "sequential_anova"
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str | int]:
        base = {
            "total_variance": self.total_variance,
            "model_variance": self.model_variance,
            "prompt_variance": self.prompt_variance,
            "task_variance": self.task_variance,
            "run_variance": self.run_variance,
            "temperature_variance": self.temperature_variance,
            "residual_variance": self.residual_variance,
            "n_observations": float(self.n_observations),
            "method": self.method,
        }
        for key, value in self.components.items():
            base[f"component_{key}"] = value
        return base


def _factor_variance_contribution(
    df: pd.DataFrame,
    residuals: pd.Series,
    col: str,
    value_col: str,
) -> tuple[float, pd.Series]:
    """Remove one factor's between-group variance from residuals (Type I sequential)."""
    if col not in df.columns or df[col].nunique() <= 1:
        return 0.0, residuals

    group_means = df.groupby(col, observed=True)[value_col].transform("mean")
    grand_mean = float(df[value_col].mean())
    factor_effect = group_means - grand_mean
    factor_var = float(np.var(factor_effect, ddof=1)) if len(df) > 1 else 0.0
    new_residuals = residuals - factor_effect
    return max(factor_var, 0.0), new_residuals


def decompose_variance(
    df: pd.DataFrame,
    *,
    score_col: str = "metric_value",
    model_col: str = "model",
    prompt_col: str = "prompt_id",
    task_col: str = "task_id",
    run_col: str = "run_id",
    temperature_col: str = "temperature",
    facet_order: list[str] | None = None,
) -> VarianceComponents:
    """Estimate variance components via sequential (Type I) ANOVA approximation.

    Factors are removed in order, assigning each factor's between-group variance
    from the remaining residual. This is an approximation; see ``fit_mixed_model``
    for a mixed-effects alternative when statsmodels is available.

    Legacy alias columns (``score``, ``task``, ``run_index``) are still accepted
    via column renaming in ``prepare_results_table``.
    """
    if score_col not in df.columns:
        legacy_map = {"score": score_col, "task": task_col, "run_index": run_col}
        for old, new in legacy_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

    scores = df[score_col].to_numpy(dtype=float)
    total_var = float(np.var(scores, ddof=1)) if len(scores) > 1 else 0.0
    residuals = pd.Series(scores - scores.mean())

    if facet_order is None:
        facet_order = [model_col, task_col, prompt_col, run_col, temperature_col]

    component_vars: dict[str, float] = {}
    for col in facet_order:
        if col not in df.columns:
            continue
        var_contrib, residuals = _factor_variance_contribution(df, residuals, col, score_col)
        component_vars[col] = var_contrib

    residual_var = float(np.var(residuals.to_numpy(), ddof=1)) if len(residuals) > 1 else 0.0
    residual_var = max(residual_var, 0.0)

    return VarianceComponents(
        total_variance=total_var,
        model_variance=component_vars.get(model_col, 0.0),
        task_variance=component_vars.get(task_col, 0.0),
        prompt_variance=component_vars.get(prompt_col, 0.0),
        run_variance=component_vars.get(run_col, 0.0),
        temperature_variance=component_vars.get(temperature_col, 0.0),
        residual_variance=residual_var,
        n_observations=len(df),
        method="sequential_anova",
        components=component_vars,
    )


def estimate_variance_components(
    df: pd.DataFrame,
    facets: list[str],
    *,
    value_col: str = "metric_value",
) -> dict[str, float]:
    """Return a flat mapping of facet name to estimated variance component."""
    result = decompose_variance(
        df,
        score_col=value_col,
        facet_order=facets,
    )
    components = {facet: result.components.get(facet, 0.0) for facet in facets}
    components["residual"] = result.residual_variance
    components["total"] = result.total_variance
    return components
