"""Binomial GLMM analysis for pass/fail confirmatory outcomes (Paper 1)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

PRIMARY_FIXED_FORMULA = "pass_fail ~ C(model_id) + C(prompt_variant_id) + temperature"
FULL_RANDOM_SPEC = {"task": "0 + C(task_id)", "run": "0 + C(run_index)"}
REDUCED_RANDOM_SPEC = {"task": "0 + C(task_id)"}
RUN_VARIANCE_THRESHOLD = 1e-6
SENSITIVITY_LPM_LABEL = "Linear probability mixed model — sensitivity analysis only"
REPRESENTATIVE_LPM_WARNINGS = (
    "Random effects covariance is singular.",
    "Hessian matrix not positive definite.",
)


@dataclass
class GLMMFitDiagnostics:
    """Diagnostics for one fitted model."""

    model_id: str
    role: str
    method: str
    formula: str
    converged: bool
    valid_for_inference: bool
    optimizer: str
    warnings: list[str] = field(default_factory=list)
    singularity: bool = False
    n_observations: int = 0
    n_tasks: int = 0
    n_runs: int = 0
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    random_effect_variances: dict[str, float] = field(default_factory=dict)
    hessian_positive_definite: bool | None = None
    notes: list[str] = field(default_factory=list)
    fit_exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "method": self.method,
            "formula": self.formula,
            "converged": self.converged,
            "valid_for_inference": self.valid_for_inference,
            "optimizer": self.optimizer,
            "warnings": "; ".join(_summarize_warnings(self.warnings)),
            "singularity": self.singularity,
            "n_observations": self.n_observations,
            "n_tasks": self.n_tasks,
            "n_runs": self.n_runs,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "random_effect_variances": json_dumps_variances(self.random_effect_variances),
            "hessian_positive_definite": self.hessian_positive_definite,
            "notes": "; ".join(self.notes),
            "fit_exception": self.fit_exception or "",
        }


@dataclass
class GLMMAnalysisResult:
    """Primary binomial inferential analysis with reduced-model comparison."""

    primary: GLMMFitDiagnostics
    coefficients: pd.DataFrame
    random_effects: pd.DataFrame
    diagnostics: pd.DataFrame
    model_comparison: pd.DataFrame
    reduced_model: GLMMFitDiagnostics | None = None
    reduced_coefficients: pd.DataFrame | None = None
    sensitivity_lpm: GLMMFitDiagnostics | None = None
    sensitivity_lpm_coefficients: pd.DataFrame | None = None
    reduced_model_needed: bool = False
    conclusions_changed: bool | None = None
    reference_categories: dict[str, str] = field(default_factory=dict)

    @property
    def primary_method(self) -> str:
        return self.primary.method


def json_dumps_variances(values: dict[str, float]) -> str:
    if not values:
        return ""
    return "; ".join(f"{key}={value:.6g}" for key, value in values.items())


def _summarize_warnings(warnings: list[str]) -> list[str]:
    """Return deduplicated, human-readable warning messages."""
    seen: set[str] = set()
    summarized: list[str] = []
    for warning in warnings:
        message = warning.split(":", 1)[-1].strip() if ":" in warning else warning.strip()
        if not message or message in seen:
            continue
        seen.add(message)
        summarized.append(message)
    return summarized


def _representative_lpm_warnings(warnings: list[str]) -> list[str]:
    """Map captured statsmodels warnings to publication-facing bullet points."""
    normalized = " ".join(_summarize_warnings(warnings)).lower()
    bullets: list[str] = []
    if "singular" in normalized:
        bullets.append(REPRESENTATIVE_LPM_WARNINGS[0])
    if "hessian" in normalized and "positive definite" in normalized:
        bullets.append(REPRESENTATIVE_LPM_WARNINGS[1])
    if not bullets:
        bullets.extend(_summarize_warnings(warnings)[:2])
    return bullets


def validate_binary_outcome(series: pd.Series, *, metric_name: str = "pass_at_1") -> None:
    """Ensure the outcome is binary on {0, 1}."""
    values = pd.Series(series).dropna().unique()
    if len(values) == 0:
        msg = f"No non-missing values for outcome {metric_name}"
        raise ValueError(msg)
    if not set(values).issubset({0, 0.0, 1, 1.0, False, True}):
        msg = f"Outcome {metric_name} must be binary; observed values {sorted(set(values))[:10]}"
        raise ValueError(msg)


def _prepare_pass_fail_frame(df: pd.DataFrame, *, metric: str = "pass_at_1") -> pd.DataFrame:
    frame = df.copy()
    alias_map = {
        "model_id": ("model_id", "model"),
        "prompt_variant_id": ("prompt_variant_id", "prompt_id"),
        "task_id": ("task_id", "task"),
        "run_index": ("run_index", "run_id"),
    }
    for canonical, options in alias_map.items():
        if canonical in frame.columns:
            continue
        for option in options:
            if option in frame.columns:
                frame[canonical] = frame[option]
                break

    if metric in frame.columns:
        frame["pass_fail"] = (frame[metric] >= 0.5).astype(int)
    elif "metric_value" in frame.columns and (
        "metric_name" not in frame.columns or frame["metric_name"].eq(metric).all()
    ):
        frame["pass_fail"] = (frame["metric_value"] >= 0.5).astype(int)
    elif "score" in frame.columns:
        frame["pass_fail"] = (frame["score"] >= 0.5).astype(int)
    else:
        frame["pass_fail"] = (frame["metric_value"] >= 0.5).astype(int)

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
    validate_binary_outcome(frame["pass_fail"], metric_name=metric)
    return frame


def _capture_warnings(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, list[str]]:
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    for record in records:
        captured.append(f"{record.category.__name__}: {record.message}")
    return result, captured


def _reference_categories(frame: pd.DataFrame) -> dict[str, str]:
    refs: dict[str, str] = {}
    for col in ("model_id", "prompt_variant_id"):
        refs[col] = str(sorted(frame[col].unique())[0])
    return refs


def _coefficients_from_log_odds(
    terms: pd.Index,
    estimates: np.ndarray,
    std_errors: np.ndarray,
    *,
    method: str,
    p_values: np.ndarray | None = None,
    valid_for_inference: bool,
) -> pd.DataFrame:
    z = 1.96
    table = pd.DataFrame(
        {
            "term": terms.astype(str),
            "log_odds_estimate": estimates,
            "std_error": std_errors,
            "odds_ratio": np.exp(estimates),
            "or_ci_lower": np.exp(estimates - z * std_errors),
            "or_ci_upper": np.exp(estimates + z * std_errors),
            "method": method,
            "valid_for_inference": valid_for_inference,
        }
    )
    if p_values is not None:
        table["p_value"] = p_values
    else:
        table["p_value"] = np.nan
    table["include_in_publication"] = valid_for_inference & np.isfinite(table["log_odds_estimate"]).all()
    finite_mask = np.isfinite(table["log_odds_estimate"]) & np.isfinite(table["std_error"])
    table.loc[~finite_mask, "include_in_publication"] = False
    return table


def _assess_coefficient_validity(coef: pd.DataFrame) -> bool:
    if coef.empty:
        return False
    estimates = coef["log_odds_estimate"].to_numpy(dtype=float)
    if not np.all(np.isfinite(estimates)):
        return False
    return True


def _mark_validity(
    diag: GLMMFitDiagnostics,
    coef: pd.DataFrame,
) -> GLMMFitDiagnostics:
    valid = diag.converged and _assess_coefficient_validity(coef)
    if diag.hessian_positive_definite is False:
        valid = False
    if diag.fit_exception:
        valid = False
    diag.valid_for_inference = valid
    coef["valid_for_inference"] = valid
    coef["include_in_publication"] = valid & np.isfinite(coef["log_odds_estimate"])
    return diag


def _extract_bayes_random_variances(fit: Any, spec: dict[str, str]) -> dict[str, float]:
    variances: dict[str, float] = {}
    if hasattr(fit, "vcp_mean"):
        values = np.asarray(fit.vcp_mean, dtype=float)
        keys = list(spec.keys())
        for index, key in enumerate(keys):
            if index < len(values):
                variances[key] = float(np.exp(values[index]))
    return variances


def _fit_binomial_bayes_mixed(
    frame: pd.DataFrame,
    *,
    model_id: str,
    role: str,
    random_spec: dict[str, str],
) -> tuple[GLMMFitDiagnostics, pd.DataFrame, pd.DataFrame]:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    random_terms = " + ".join(f"(1|{key})" for key in random_spec)
    formula = f"{PRIMARY_FIXED_FORMULA} + {random_terms}" if random_terms else PRIMARY_FIXED_FORMULA
    diag = GLMMFitDiagnostics(
        model_id=model_id,
        role=role,
        method="BinomialBayesMixedGLM_VB",
        formula=formula,
        converged=False,
        valid_for_inference=False,
        optimizer="variational_bayes",
        n_observations=len(frame),
        n_tasks=int(frame["task_id"].nunique()),
        n_runs=int(frame["run_index"].nunique()),
    )

    try:
        model = BinomialBayesMixedGLM.from_formula(
            PRIMARY_FIXED_FORMULA,
            random_spec,
            frame,
        )
        fit, captured = _capture_warnings(model.fit_vb)
        diag.warnings.extend(captured)
        diag.converged = True
        diag.log_likelihood = float(getattr(fit, "llf", np.nan)) if hasattr(fit, "llf") else None

        names = list(model.exog_names)
        estimates = np.asarray(fit.params[: len(names)], dtype=float)
        cov = fit.cov_params()
        if hasattr(cov, "iloc"):
            variances = np.asarray(cov.iloc[: len(names)], dtype=float)
        else:
            variances = np.asarray(cov)[: len(names)]
        std_errors = np.sqrt(np.maximum(variances, 0.0))

        diag.random_effect_variances = _extract_bayes_random_variances(fit, random_spec)
        diag.singularity = any("singular" in warning.lower() for warning in captured)

        coef = _coefficients_from_log_odds(
            pd.Index(names),
            estimates,
            std_errors,
            method=diag.method,
            valid_for_inference=False,
        )
        re_rows = [
            {"effect": key, "variance": value, "method": diag.method, "model_id": model_id}
            for key, value in diag.random_effect_variances.items()
        ]
        random_effects = pd.DataFrame(re_rows)
        diag = _mark_validity(diag, coef)
        return diag, coef, random_effects
    except Exception as exc:  # noqa: BLE001
        diag.fit_exception = str(exc)
        diag.notes.append(f"BinomialBayesMixedGLM failed: {exc}")
        return diag, pd.DataFrame(), pd.DataFrame()


def _fit_cluster_robust_binomial_glm(frame: pd.DataFrame) -> tuple[GLMMFitDiagnostics, pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm

    formula = PRIMARY_FIXED_FORMULA + " (cluster-robust SE by task_id)"
    diag = GLMMFitDiagnostics(
        model_id="cluster_robust_glm",
        role="fallback",
        method="BinomialGLM_cluster_robust_by_task",
        formula=formula,
        converged=False,
        valid_for_inference=False,
        optimizer="IRLS",
        notes=["Fallback: cluster-robust binomial GLM with task clusters."],
        n_observations=len(frame),
        n_tasks=int(frame["task_id"].nunique()),
        n_runs=int(frame["run_index"].nunique()),
    )

    design = pd.get_dummies(
        frame[["model_id", "prompt_variant_id", "temperature"]],
        columns=["model_id", "prompt_variant_id"],
        drop_first=True,
    )
    design = sm.add_constant(design, has_constant="add").astype(float)
    y = frame["pass_fail"].astype(float)
    cluster_groups = frame["task_id"].astype("category").cat.codes

    try:
        logit = sm.GLM(y, design, family=sm.families.Binomial())
        result, captured = _capture_warnings(
            logit.fit,
            cov_type="cluster",
            cov_kwds={"groups": cluster_groups},
        )
        diag.warnings.extend(captured)
        diag.converged = bool(result.converged)
        diag.log_likelihood = float(result.llf)
        diag.aic = float(result.aic)
        diag.bic = float(result.bic)

        coef = _coefficients_from_log_odds(
            result.params.index,
            np.asarray(result.params.values, dtype=float),
            np.asarray(result.bse.values, dtype=float),
            method=diag.method,
            p_values=np.asarray(result.pvalues.values, dtype=float),
            valid_for_inference=False,
        )
        diag = _mark_validity(diag, coef)
        return diag, coef, pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        diag.fit_exception = str(exc)
        diag.notes.append(f"Cluster-robust GLM failed: {exc}")
        return diag, pd.DataFrame(), pd.DataFrame()


def fit_linear_probability_sensitivity(
    frame: pd.DataFrame,
) -> tuple[GLMMFitDiagnostics, pd.DataFrame]:
    """Gaussian MixedLM on binary outcomes — sensitivity analysis only."""
    import statsmodels.formula.api as smf

    work = frame.copy()
    if "pass_fail" not in work.columns:
        if "metric_value" in work.columns:
            work["pass_fail"] = (work["metric_value"] >= 0.5).astype(int)
        else:
            msg = "LPM sensitivity requires pass_fail or metric_value column"
            raise ValueError(msg)
    if "run_id" not in work.columns and "run_index" in work.columns:
        work["run_id"] = work["run_index"]
    if "model_id" not in work.columns and "model" in work.columns:
        work["model_id"] = work["model"]
    if "prompt_variant_id" not in work.columns and "prompt_id" in work.columns:
        work["prompt_variant_id"] = work["prompt_id"]
    work["metric_value"] = work["pass_fail"].astype(float)

    fixed_formula = "metric_value ~ C(model_id) + C(prompt_variant_id) + temperature"
    diag = GLMMFitDiagnostics(
        model_id="lpm_sensitivity",
        role="sensitivity_lpm",
        method=SENSITIVITY_LPM_LABEL,
        formula=fixed_formula + " + (1|task_id) + (1|run_id)",
        converged=False,
        valid_for_inference=False,
        optimizer="lbfgs",
        notes=["Not for primary inference on binary pass/fail outcomes."],
        n_observations=len(work),
        n_tasks=int(work["task_id"].nunique()),
        n_runs=int(work["run_index"].nunique()) if "run_index" in work.columns else int(work["run_id"].nunique()),
    )

    try:
        model = smf.mixedlm(
            fixed_formula,
            work,
            groups=work["task_id"],
            vc_formula={"run": "0 + C(run_id)"},
        )
        fit, captured = _capture_warnings(
            model.fit,
            reml=True,
            method="lbfgs",
            maxiter=300,
            disp=False,
        )
        diag.warnings.extend(_summarize_warnings(captured))
        diag.converged = bool(getattr(fit, "converged", False))
        diag.singularity = any("singular" in warning.lower() for warning in captured)
        diag.log_likelihood = float(fit.llf) if hasattr(fit, "llf") else None
        diag.aic = float(fit.aic) if hasattr(fit, "aic") else None
        diag.bic = float(fit.bic) if hasattr(fit, "bic") else None
        if hasattr(fit, "cov_re") and getattr(fit.cov_re, "size", 0):
            diag.random_effect_variances = {"task_intercept": float(fit.cov_re.iloc[0, 0])}

        hessian = getattr(fit, "hessian", None)
        if hessian is not None:
            eigvals = np.linalg.eigvalsh(np.asarray(hessian, dtype=float))
            diag.hessian_positive_definite = bool(np.all(eigvals > 0))

        conf = fit.conf_int()
        coef = pd.DataFrame(
            {
                "term": fit.fe_params.index.astype(str),
                "log_odds_estimate": fit.fe_params.values,
                "std_error": fit.bse_fe.values,
                "odds_ratio": np.nan,
                "or_ci_lower": conf.iloc[:, 0].values,
                "or_ci_upper": conf.iloc[:, 1].values,
                "p_value": fit.pvalues.values,
                "method": diag.method,
                "valid_for_inference": False,
                "include_in_publication": False,
            }
        )
        if not diag.converged or diag.singularity or diag.hessian_positive_definite is False:
            diag.notes.append("Excluded from publication tables due to convergence or singularity issues.")
            coef["include_in_publication"] = False
        diag.valid_for_inference = False
        return diag, coef
    except Exception as exc:  # noqa: BLE001
        diag.fit_exception = str(exc)
        diag.notes.append(f"LPM sensitivity fit failed: {exc}")
        return diag, pd.DataFrame()


def _compare_fixed_effects(
    full_coef: pd.DataFrame,
    reduced_coef: pd.DataFrame,
) -> tuple[bool, pd.DataFrame]:
    merged = full_coef.merge(
        reduced_coef,
        on="term",
        suffixes=("_full", "_reduced"),
        how="inner",
    )
    if merged.empty:
        return False, pd.DataFrame()

    rows: list[dict[str, Any]] = []
    changed = False
    for _, row in merged.iterrows():
        full_or = float(row.get("odds_ratio_full", np.nan))
        reduced_or = float(row.get("odds_ratio_reduced", np.nan))
        same_direction = (
            np.isfinite(full_or)
            and np.isfinite(reduced_or)
            and np.sign(np.log(full_or)) == np.sign(np.log(reduced_or))
        )
        ci_overlap = not (
            row.get("or_ci_upper_full", np.nan) < row.get("or_ci_lower_reduced", np.nan)
            or row.get("or_ci_upper_reduced", np.nan) < row.get("or_ci_lower_full", np.nan)
        )
        conclusion_changed = not (same_direction and ci_overlap)
        changed = changed or conclusion_changed
        rows.append(
            {
                "term": row["term"],
                "odds_ratio_full": full_or,
                "odds_ratio_reduced": reduced_or,
                "same_direction": same_direction,
                "ci_overlap": ci_overlap,
                "conclusion_changed": conclusion_changed,
            }
        )
    return changed, pd.DataFrame(rows)


def run_pass_fail_glmm_analysis(
    df: pd.DataFrame,
    *,
    metric: str = "pass_at_1",
) -> GLMMAnalysisResult:
    """Fit the preregistered binomial hierarchy for pass/fail confirmatory inference."""
    frame = _prepare_pass_fail_frame(df, metric=metric)
    refs = _reference_categories(frame)

    diagnostics: list[GLMMFitDiagnostics] = []
    comparisons: list[dict[str, Any]] = []

    full_diag, full_coef, full_re = _fit_binomial_bayes_mixed(
        frame,
        model_id="binomial_mixed_full",
        role="primary_candidate",
        random_spec=FULL_RANDOM_SPEC,
    )
    diagnostics.append(full_diag)

    reduced_diag: GLMMFitDiagnostics | None = None
    reduced_coef = pd.DataFrame()
    reduced_re = pd.DataFrame()
    reduced_needed = False
    conclusions_changed: bool | None = None

    run_variance = full_diag.random_effect_variances.get("run")
    full_invalid = not full_diag.valid_for_inference
    run_near_zero = run_variance is not None and run_variance <= RUN_VARIANCE_THRESHOLD

    if run_near_zero or full_invalid:
        reduced_needed = True
        if run_near_zero:
            full_diag.notes.append(
                "Run random-effect variance near zero; comparing with task-only reduced model."
            )
        reduced_diag, reduced_coef, reduced_re = _fit_binomial_bayes_mixed(
            frame,
            model_id="binomial_mixed_task_only",
            role="reduced",
            random_spec=REDUCED_RANDOM_SPEC,
        )
        diagnostics.append(reduced_diag)
        if not full_coef.empty and not reduced_coef.empty:
            conclusions_changed, comparison = _compare_fixed_effects(full_coef, reduced_coef)
            comparisons.extend(comparison.to_dict(orient="records"))

    primary_diag = full_diag
    primary_coef = full_coef
    primary_re = full_re
    if full_invalid and reduced_diag is not None and reduced_diag.valid_for_inference:
        primary_diag = reduced_diag
        primary_diag.role = "primary"
        primary_coef = reduced_coef
        primary_re = reduced_re
    elif full_diag.valid_for_inference:
        primary_diag = full_diag
        primary_diag.role = "primary"
    else:
        fallback_diag, fallback_coef, fallback_re = _fit_cluster_robust_binomial_glm(frame)
        diagnostics.append(fallback_diag)
        if fallback_diag.valid_for_inference:
            primary_diag = fallback_diag
            primary_diag.role = "primary"
            primary_coef = fallback_coef
            primary_re = fallback_re
        elif reduced_diag is not None and reduced_diag.valid_for_inference:
            primary_diag = reduced_diag
            primary_diag.role = "primary"
            primary_coef = reduced_coef
            primary_re = reduced_re

    if primary_diag.role != "primary":
        primary_diag.role = "primary"

    lpm_diag, lpm_coef = fit_linear_probability_sensitivity(frame)
    diagnostics.append(lpm_diag)

    diagnostics_df = pd.DataFrame([diag.to_dict() for diag in diagnostics])
    model_comparison = pd.DataFrame(comparisons)

    publication_coef = primary_coef[primary_coef.get("include_in_publication", True)].copy()
    if publication_coef.empty and not primary_coef.empty:
        publication_coef = primary_coef.copy()

    return GLMMAnalysisResult(
        primary=primary_diag,
        coefficients=publication_coef,
        random_effects=primary_re,
        diagnostics=diagnostics_df,
        model_comparison=model_comparison,
        reduced_model=reduced_diag,
        reduced_coefficients=reduced_coef if not reduced_coef.empty else None,
        sensitivity_lpm=lpm_diag,
        sensitivity_lpm_coefficients=lpm_coef if not lpm_coef.empty else None,
        reduced_model_needed=reduced_needed,
        conclusions_changed=conclusions_changed,
        reference_categories=refs,
    )


def glmm_coefficients_table(result: GLMMAnalysisResult) -> pd.DataFrame:
    """Return publication-ready coefficient table with odds ratios."""
    table = result.coefficients.copy()
    if "include_in_publication" in table.columns:
        table = table[table["include_in_publication"]].copy()
    return table


def render_glmm_interpretation(result: GLMMAnalysisResult) -> str:
    """Render markdown interpretation for the primary binomial model."""
    primary = result.primary
    lines = [
        "# GLMM interpretation — pass/fail confirmatory analysis",
        "",
        "## Primary inferential model",
        "",
        f"- **Method:** {primary.method}",
        f"- **Formula:** `{primary.formula}`",
        f"- **Converged:** {primary.converged}",
        f"- **Valid for inference:** {primary.valid_for_inference}",
        f"- **Observations:** {primary.n_observations}",
        f"- **Tasks:** {primary.n_tasks}",
        f"- **Runs:** {primary.n_runs}",
        "",
        "## Random-effect variances",
        "",
    ]
    if primary.random_effect_variances:
        for key, value in primary.random_effect_variances.items():
            lines.append(f"- `{key}`: {value:.6g}")
    else:
        lines.append("- Not available for the selected primary model.")

    lines.extend(["", "## Reference categories", ""])
    for factor, level in result.reference_categories.items():
        lines.append(f"- `{factor}` → `{level}`")

        lines.extend(["", "## Fixed effects (odds ratios, approximate 95% variational intervals)", ""])
    coef = glmm_coefficients_table(result)
    if coef.empty:
        lines.append("_No coefficients qualified for publication export._")
    else:
        for _, row in coef.iterrows():
            lines.append(
                f"- `{row['term']}`: OR={row['odds_ratio']:.3f} "
                f"[{row['or_ci_lower']:.3f}, {row['or_ci_upper']:.3f}] "
                "(approx. variational posterior interval on the OR scale)"
            )

    lines.extend(["", "## Reduced-model comparison", ""])
    if result.reduced_model_needed:
        lines.append(
            f"- Reduced task-only model fitted: **{result.reduced_model.method if result.reduced_model else 'n/a'}**"
        )
        if result.conclusions_changed is None:
            lines.append("- Comparison unavailable.")
        elif result.conclusions_changed:
            lines.append(
                "- Fixed-effect odds ratios or uncertainty intervals differ materially between full and reduced models. "
                "Interpret substantive claims using the reduced model and report both estimates."
            )
        else:
            lines.append(
                "- Substantive fixed-effect directions and overlapping uncertainty intervals are consistent between "
                "full and reduced models."
            )
        lines.append(
            "- A near-zero run variance does **not** prove runs have no effect; it indicates the full crossed "
            "random-effects structure was poorly identified for run."
        )
    else:
        lines.append("- Reduced model was not required.")

    lines.extend(["", "## Sensitivity analysis", ""])
    if result.sensitivity_lpm is not None:
        lpm = result.sensitivity_lpm
        excluded = (
            lpm.singularity
            or lpm.hessian_positive_definite is False
            or not lpm.valid_for_inference
            or bool(lpm.fit_exception)
        )
        lines.extend(
            [
                "The linear probability mixed model was fitted only as a sensitivity analysis.",
                "",
            ]
        )
        if excluded:
            lines.extend(
                [
                    "The fit exhibited singular covariance and a non-positive definite Hessian.",
                    "",
                    "Consequently, its estimates are not used for inference and are omitted from all publication tables.",
                    "",
                    "Representative warnings:",
                    "",
                ]
            )
            for bullet in _representative_lpm_warnings(lpm.warnings):
                lines.append(f"- {bullet}")
        else:
            lines.append(
                "The linear probability mixed model converged without singularity flags, "
                "but it remains a sensitivity check only and is not used for primary inference."
            )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Primary inference uses a binomial model on pass/fail; ANOVA on continuous scores remains descriptive.",
            "- Odds ratios describe association on the logit scale; uncertainty should be read from "
            "approximate variational posterior intervals (normal approximation on log-odds), "
            "not classical frequentist confidence intervals.",
            "- p-values are reported when available but are not treated as the sole evidence of importance.",
            "- Crossed random effects may be weakly identified even when the reduced model converges.",
            "",
        ]
    )
    return "\n".join(lines)


# Backward-compatible entry point used by older imports.
def fit_pass_fail_glmm(df: pd.DataFrame, *, metric: str = "pass_at_1") -> GLMMAnalysisResult:
    return run_pass_fail_glmm_analysis(df, metric=metric)
