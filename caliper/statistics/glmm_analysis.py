"""Generalized linear mixed models for pass/fail confirmatory analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GLMMResult:
    """Container for a binomial GLMM fit and coefficient table."""

    formula: str
    coefficients: pd.DataFrame
    converged: bool
    notes: list[str]
    n_observations: int
    method: str


def _prepare_pass_fail_frame(df: pd.DataFrame, *, metric: str = "pass_at_1") -> pd.DataFrame:
    frame = df.copy()
    if metric in frame.columns:
        frame["pass_fail"] = (frame[metric] >= 0.5).astype(int)
    elif "score" in frame.columns and frame.get("metric", pd.Series([metric])).eq(metric).all():
        frame["pass_fail"] = (frame["score"] >= 0.5).astype(int)
    else:
        frame["pass_fail"] = (frame["score"] >= 0.5).astype(int)

    required = {"model_id", "prompt_variant_id", "temperature", "task_id", "run_index"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"GLMM requires columns {sorted(required)}; missing {sorted(missing)}"
        raise ValueError(msg)

    frame["model_id"] = frame["model_id"].astype(str)
    frame["prompt_variant_id"] = frame["prompt_variant_id"].astype(str)
    frame["task_id"] = frame["task_id"].astype(str)
    frame["run_index"] = frame["run_index"].astype(int)
    frame["temperature"] = frame["temperature"].astype(float)
    frame["pass_fail"] = frame["pass_fail"].astype(int)
    return frame


def fit_pass_fail_glmm(
    df: pd.DataFrame,
    *,
    metric: str = "pass_at_1",
) -> GLMMResult:
    """Fit a binomial GLMM with model, prompt, temperature fixed effects.

    Random effects: task and run when supported; otherwise task-only or
    cluster-robust logistic regression as fallback.
    """
    frame = _prepare_pass_fail_frame(df, metric=metric)
    formula = "pass_fail ~ C(model_id) + C(prompt_variant_id) + temperature"
    notes: list[str] = []

    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        model = BinomialBayesMixedGLM.from_formula(
            formula,
            {"task": "0 + C(task_id)", "run": "0 + C(run_index)"},
            frame,
        )
        fit = model.fit_vb()
        coef = pd.DataFrame(
            {
                "term": fit.params.index,
                "estimate": fit.params.values,
                "std_error": fit.std_params.values,
            }
        )
        coef["z_value"] = coef["estimate"] / coef["std_error"].replace(0, np.nan)
        coef["p_value"] = np.nan
        return GLMMResult(
            formula=formula + " + (1|task_id) + (1|run_index)",
            coefficients=coef,
            converged=True,
            notes=notes,
            n_observations=len(frame),
            method="BinomialBayesMixedGLM_VB",
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"BinomialBayesMixedGLM failed ({exc}).")

    try:
        import statsmodels.formula.api as smf

        model = smf.mixedlm(
            formula,
            data=frame,
            groups=frame["task_id"],
            re_formula="1",
        )
        result = model.fit(method="lbfgs", maxiter=200, disp=False)
        coef = pd.DataFrame(
            {
                "term": result.fe_params.index,
                "estimate": result.fe_params.values,
                "std_error": result.bse_fe.values,
                "z_value": result.tvalues.values,
                "p_value": result.pvalues.values,
            }
        )
        notes.append("Approximate linear mixed model on binary outcome; interpret cautiously.")
        return GLMMResult(
            formula=formula + " + (1|task_id)",
            coefficients=coef,
            converged=bool(result.converged),
            notes=notes,
            n_observations=len(frame),
            method="MixedLM_approximate_binomial",
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"MixedLM failed ({exc}); using cluster-robust logistic regression.")

    import statsmodels.api as sm

    design = pd.get_dummies(
        frame[["model_id", "prompt_variant_id", "temperature"]],
        columns=["model_id", "prompt_variant_id"],
        drop_first=True,
    )
    design = sm.add_constant(design, has_constant="add").astype(float)
    y = frame["pass_fail"].astype(float)

    logit = sm.GLM(y, design, family=sm.families.Binomial())
    cluster_groups = frame["task_id"].astype("category").cat.codes
    result = logit.fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})

    coef = pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    return GLMMResult(
        formula=formula + " (cluster-robust SE by task_id)",
        coefficients=coef,
        converged=bool(result.converged),
        notes=notes,
        n_observations=len(frame),
        method="BinomialGLM_cluster_robust",
    )


def glmm_coefficients_table(result: GLMMResult) -> pd.DataFrame:
    """Return publication-ready GLMM coefficient table."""
    table = result.coefficients.copy()
    table["significant_005"] = table["p_value"] < 0.05
    return table
