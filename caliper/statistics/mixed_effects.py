"""Mixed-effects model interface for Paper 1 analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from caliper.statistics.variance import VarianceComponents, decompose_variance


@dataclass
class MixedModelResult:
    """Result of a mixed-effects fit or fallback approximation."""

    formula: str
    method: str
    converged: bool
    fixed_effects: dict[str, float] = field(default_factory=dict)
    variance_components: dict[str, float] = field(default_factory=dict)
    aic: float | None = None
    bic: float | None = None
    notes: list[str] = field(default_factory=list)
    raw_summary: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "method": self.method,
            "converged": self.converged,
            "fixed_effects": self.fixed_effects,
            "variance_components": self.variance_components,
            "aic": self.aic,
            "bic": self.bic,
            "notes": self.notes,
        }


def fit_mixed_model(
    df: pd.DataFrame,
    *,
    value_col: str = "metric_value",
    fixed_formula: str = "C(model)",
    group_col: str = "task_id",
    re_formula: str = "1",
) -> MixedModelResult:
    """Fit a linear mixed-effects model with statsmodels, or fall back to ANOVA.

    statsmodels ``MixedLM`` supports a single grouping factor. Crossed random
    effects (prompt × task × run) require specialized software; when MixedLM
    fails or is unavailable we document limitations and return a sequential
    ANOVA approximation instead.

    Args:
        df: Prepared results table.
        value_col: Dependent variable column.
        fixed_formula: Right-hand side of fixed-effects formula (without DV).
        group_col: Grouping column for random intercepts.
        re_formula: Random-effects formula (default random intercept).

    Returns:
        MixedModelResult with variance components and diagnostics.
    """
    formula = f"{value_col} ~ {fixed_formula}"
    notes: list[str] = []

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        notes.append("statsmodels not installed; using sequential ANOVA fallback")
        return _fallback_result(df, value_col=value_col, formula=formula, notes=notes)

    if group_col not in df.columns or df[group_col].nunique() < 2:
        notes.append(f"group column '{group_col}' unusable; using sequential ANOVA fallback")
        return _fallback_result(df, value_col=value_col, formula=formula, notes=notes)

    try:
        model = smf.mixedlm(formula, data=df, groups=df[group_col], re_formula=re_formula)
        fit = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
    except Exception as exc:
        notes.append(f"MixedLM failed ({exc}); using sequential ANOVA fallback")
        return _fallback_result(df, value_col=value_col, formula=formula, notes=notes)

    variance_components = {
        "group_intercept": float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else 0.0,
        "residual": float(fit.scale),
    }
    fixed_effects = {str(k): float(v) for k, v in fit.fe_params.items()}

    notes.append(
        "MixedLM estimates one random intercept grouping factor. "
        "Crossed facets (prompt, run, temperature) are not jointly modeled."
    )

    return MixedModelResult(
        formula=formula,
        method="statsmodels_mixedlm",
        converged=bool(fit.converged),
        fixed_effects=fixed_effects,
        variance_components=variance_components,
        aic=float(fit.aic) if fit.aic is not None else None,
        bic=float(fit.bic) if fit.bic is not None else None,
        notes=notes,
        raw_summary=str(fit.summary()),
    )


def _fallback_result(
    df: pd.DataFrame,
    *,
    value_col: str,
    formula: str,
    notes: list[str],
) -> MixedModelResult:
    facets = [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in df.columns]
    vc = decompose_variance(df, score_col=value_col, facet_order=facets)
    components = dict(vc.components)
    components["residual"] = vc.residual_variance
    notes.append(
        "Fallback uses sequential Type-I ANOVA which confounds facet order. "
        "Treat components as approximate."
    )
    return MixedModelResult(
        formula=formula,
        method="sequential_anova_fallback",
        converged=True,
        variance_components=components,
        notes=notes,
    )
