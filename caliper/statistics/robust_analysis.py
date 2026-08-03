"""Robust statistical analysis: ANOVA Type II/III and mixed-effects models."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

AnovaType = Literal[2, 3]

SENSITIVITY_LPM_METHOD = "Linear probability mixed model — sensitivity analysis only"

FIXED_FACTORS = ("model", "prompt_id", "temperature")
RANDOM_FACTORS = ("task_id", "run_id")
DEFAULT_FORMULA = (
    "metric_value ~ C(model) + C(prompt_id) + C(temperature) + C(task_id) + C(run_id)"
)


@dataclass
class RobustAnalysisResult:
    """Combined output from ANOVA and mixed-effects fits."""

    anova_type2: pd.DataFrame
    anova_type3: pd.DataFrame
    mixed_effects: pd.DataFrame
    method_comparison: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "anova_type2_rows": len(self.anova_type2),
            "anova_type3_rows": len(self.anova_type3),
            "mixed_effects_rows": len(self.mixed_effects),
            "notes": self.notes,
        }


def _partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    denom = ss_effect + ss_error
    return float(ss_effect / denom) if denom > 0 else 0.0


def _omega_squared(ss_effect: float, df_effect: float, ms_error: float, ss_total: float) -> float:
    numerator = ss_effect - df_effect * ms_error
    denom = ss_total + ms_error
    if denom <= 0:
        return 0.0
    return float(max(numerator / denom, 0.0))


def _anova_table_to_effects(table: pd.DataFrame, *, anova_type: AnovaType) -> pd.DataFrame:
    """Convert statsmodels ANOVA table to standardized effect rows."""
    residual_row = table.loc["Residual"] if "Residual" in table.index else None
    ss_error = float(residual_row["sum_sq"]) if residual_row is not None else 0.0
    df_error = float(residual_row["df"]) if residual_row is not None else 1.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0
    ss_total = float(table["sum_sq"].sum())

    rows: list[dict[str, Any]] = []
    for term, row in table.iterrows():
        if term == "Residual":
            continue
        ss_effect = float(row["sum_sq"])
        df_effect = float(row["df"])
        rows.append(
            {
                "term": str(term),
                "method": f"anova_type_{anova_type}",
                "sum_sq": ss_effect,
                "df": df_effect,
                "f_stat": float(row["F"]) if "F" in row and pd.notna(row["F"]) else float("nan"),
                "p_value": float(row["PR(>F)"]) if "PR(>F)" in row and pd.notna(row["PR(>F)"]) else float("nan"),
                "partial_eta_squared": _partial_eta_squared(ss_effect, ss_error),
                "omega_squared": _omega_squared(ss_effect, df_effect, ms_error, ss_total),
                "estimate": float("nan"),
                "std_err": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _build_anova_formula(df: pd.DataFrame) -> str:
    terms: list[str] = []
    for col in ("model", "prompt_id", "temperature", "task_id", "run_id"):
        if col in df.columns and df[col].nunique() > 1:
            terms.append(f"C({col})")
    if not terms:
        return "metric_value ~ 1"
    return "metric_value ~ " + " + ".join(terms)


def run_anova(
    df: pd.DataFrame,
    *,
    anova_type: AnovaType = 2,
    formula: str | None = None,
    value_col: str = "metric_value",
) -> pd.DataFrame:
    """Run Type II or Type III ANOVA on the prepared results table."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    if value_col != "metric_value" and value_col in df.columns:
        work = df.rename(columns={value_col: "metric_value"})
    else:
        work = df

    formula = formula or _build_anova_formula(work)
    if formula.endswith("~ 1"):
        return pd.DataFrame(
            columns=[
                "term",
                "method",
                "sum_sq",
                "df",
                "f_stat",
                "p_value",
                "partial_eta_squared",
                "omega_squared",
                "estimate",
                "std_err",
                "ci_lower",
                "ci_upper",
            ]
        )

    model = smf.ols(formula, data=work).fit()
    try:
        table = anova_lm(model, typ=anova_type)
    except (ValueError, np.linalg.LinAlgError):
        return pd.DataFrame(
            columns=[
                "term",
                "method",
                "sum_sq",
                "df",
                "f_stat",
                "p_value",
                "partial_eta_squared",
                "omega_squared",
                "estimate",
                "std_err",
                "ci_lower",
                "ci_upper",
            ]
        )
    return _anova_table_to_effects(table, anova_type=anova_type)


def fit_robust_mixed_model(
    df: pd.DataFrame,
    *,
    value_col: str = "metric_value",
) -> tuple[pd.DataFrame, list[str]]:
    """Fit MixedLM with task and run random effects and fixed model/prompt/temperature."""
    import statsmodels.formula.api as smf

    notes: list[str] = []
    if value_col != "metric_value" and value_col in df.columns:
        work = df.rename(columns={value_col: "metric_value"})
    else:
        work = df.copy()

    if "run_index" in work.columns and work.get("run_id", pd.Series()).nunique(dropna=False) <= 1:
        work["run_id"] = work["run_index"]

    fixed_formula = "metric_value ~ C(model) + C(prompt_id) + C(temperature)"
    rows: list[dict[str, Any]] = []

    fit = None
    method = "statsmodels_mixedlm"
    if work["task_id"].nunique() >= 2:
        try:
            if work["run_id"].nunique() >= 2:
                model = smf.mixedlm(
                    fixed_formula,
                    work,
                    groups=work["task_id"],
                    vc_formula={"run": "0 + C(run_id)"},
                )
                notes.append("MixedLM: random intercepts for task (groups) and run (vc_formula).")
            else:
                model = smf.mixedlm(
                    fixed_formula,
                    work,
                    groups=work["task_id"],
                    re_formula="1",
                )
                notes.append("MixedLM: random intercept for task only (single run level).")
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                fit = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
            for record in records:
                notes.append(f"{record.category.__name__}: {record.message}")
        except Exception as exc:
            notes.append(f"MixedLM with task+run failed ({exc}); trying task-only grouping.")
            fit = None

    if fit is None and work["run_id"].nunique() >= 2:
        try:
            model = smf.mixedlm(
                fixed_formula,
                work,
                groups=work["run_id"],
                re_formula="1",
            )
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                fit = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
            for record in records:
                notes.append(f"{record.category.__name__}: {record.message}")
            method = "statsmodels_mixedlm_run_group"
            notes.append("MixedLM fallback: random intercept for run only.")
        except Exception as exc:
            notes.append(f"MixedLM run-group fallback failed ({exc}).")

    if fit is None:
        notes.append("MixedLM unavailable; returning empty fixed-effects table.")
        return pd.DataFrame(
            columns=[
                "term",
                "method",
                "estimate",
                "std_err",
                "ci_lower",
                "ci_upper",
                "p_value",
                "partial_eta_squared",
                "omega_squared",
            ]
        ), notes

    conf = fit.conf_int()
    ss_resid = float(getattr(fit, "scale", 0.0) or 0.0) * max(len(work) - 1, 1)
    ss_total = float(((work["metric_value"] - work["metric_value"].mean()) ** 2).sum())

    for term in fit.fe_params.index:
        estimate = float(fit.fe_params[term])
        std_err = float(fit.bse[term])
        p_value = float(fit.pvalues[term])
        ci_lo = float(conf.loc[term, 0])
        ci_hi = float(conf.loc[term, 1])
        partial_eta = estimate**2 / (estimate**2 + ss_resid) if ss_resid > 0 else 0.0
        omega = partial_eta  # conservative placeholder for single coefficient
        rows.append(
            {
                "term": str(term),
                "method": method,
                "estimate": estimate,
                "std_err": std_err,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": p_value,
                "partial_eta_squared": partial_eta,
                "omega_squared": omega,
                "sum_sq": float("nan"),
                "df": float("nan"),
                "f_stat": float("nan"),
            }
        )

    if hasattr(fit, "cov_re") and fit.cov_re.size:
        re_var = float(fit.cov_re.iloc[0, 0])
        rows.append(
            {
                "term": "random_intercept_variance",
                "method": method,
                "estimate": re_var,
                "std_err": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "p_value": float("nan"),
                "partial_eta_squared": re_var / ss_total if ss_total > 0 else 0.0,
                "omega_squared": re_var / ss_total if ss_total > 0 else 0.0,
                "sum_sq": float("nan"),
                "df": float("nan"),
                "f_stat": float("nan"),
            }
        )

    notes.append(f"MixedLM converged={bool(getattr(fit, 'converged', False))}.")
    if not rows:
        return pd.DataFrame(
            columns=[
                "term",
                "method",
                "estimate",
                "std_err",
                "ci_lower",
                "ci_upper",
                "p_value",
                "partial_eta_squared",
                "omega_squared",
            ]
        ), notes

    out = pd.DataFrame(rows)
    out["method"] = SENSITIVITY_LPM_METHOD
    out["include_in_publication"] = bool(getattr(fit, "converged", False)) and not any(
        "singular" in note.lower() for note in notes
    )
    return out, notes


def compare_methods(df: pd.DataFrame) -> RobustAnalysisResult:
    """Run Type II, Type III ANOVA and mixed-effects model on the same dataset."""
    notes: list[str] = []
    anova2 = run_anova(df, anova_type=2)
    anova3 = run_anova(df, anova_type=3)
    mixed, mixed_notes = fit_robust_mixed_model(df)
    notes.extend(mixed_notes)

    comparison_rows: list[dict[str, Any]] = []
    for label, frame in (
        ("type_ii_anova", anova2),
        ("type_iii_anova", anova3),
        ("mixed_effects", mixed),
    ):
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            comparison_rows.append(
                {
                    "source": label,
                    "term": row.get("term"),
                    "p_value": row.get("p_value"),
                    "partial_eta_squared": row.get("partial_eta_squared"),
                    "omega_squared": row.get("omega_squared"),
                    "estimate": row.get("estimate"),
                    "ci_lower": row.get("ci_lower"),
                    "ci_upper": row.get("ci_upper"),
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    return RobustAnalysisResult(
        anova_type2=anova2,
        anova_type3=anova3,
        mixed_effects=mixed,
        method_comparison=comparison,
        notes=notes,
    )


def leave_one_out_sensitivity(
    df: pd.DataFrame,
    *,
    value_col: str = "metric_value",
) -> pd.DataFrame:
    """Measure ranking and variance changes when removing one level per factor."""
    from caliper.ranking.aggregate import aggregate_scores_by_model, rank_models
    from caliper.ranking.metrics import kendall_tau_between_rankings
    from caliper.statistics.gtheory import estimate_g_variance_components

    baseline_scores = aggregate_scores_by_model(df, value_col=value_col)
    baseline_ranks = rank_models(baseline_scores)
    baseline_var = estimate_g_variance_components(
        df,
        [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in df.columns],
        value_col=value_col,
    ).components

    rows: list[dict[str, Any]] = []
    drop_specs = [
        ("model", "model"),
        ("prompt_id", "prompt"),
        ("task_id", "task"),
        ("run_id", "run"),
    ]
    for col, label in drop_specs:
        if col not in df.columns:
            continue
        levels = sorted(df[col].unique())
        # Full leave-one-task-out is O(n_tasks) variance refits; subsample for large benchmarks.
        if col == "task_id" and len(levels) > 40:
            rng = np.random.default_rng(20260404)
            levels = sorted(rng.choice(levels, size=40, replace=False).tolist())
        for level in levels:
            subset = df[df[col] != level]
            if subset.empty or subset[col].nunique() < 1:
                continue
            scores = aggregate_scores_by_model(subset, value_col=value_col)
            ranks = rank_models(scores)
            common = baseline_ranks.index.intersection(ranks.index)
            if len(common) < 2:
                tau = float("nan")
            else:
                tau = kendall_tau_between_rankings(baseline_ranks.loc[common], ranks.loc[common])
            var = estimate_g_variance_components(
                subset,
                [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in subset.columns],
                value_col=value_col,
            ).components
            rows.append(
                {
                    "dropped_factor": label,
                    "dropped_level": str(level),
                    "n_observations": len(subset),
                    "kendall_tau_vs_full": tau,
                    "model_variance_change": var.get("model", 0.0) - baseline_var.get("model", 0.0),
                    "prompt_variance_change": var.get("prompt_id", 0.0) - baseline_var.get("prompt_id", 0.0),
                    "task_variance_change": var.get("task_id", 0.0) - baseline_var.get("task_id", 0.0),
                    "residual_variance_change": var.get("residual", 0.0) - baseline_var.get("residual", 0.0),
                }
            )
    return pd.DataFrame(rows)


def factor_effect_sizes(df: pd.DataFrame, *, value_col: str = "metric_value") -> pd.DataFrame:
    """Cohen's d, Cliff's delta, η² and ω² for each main factor (Type II ANOVA)."""
    anova = run_anova(df, anova_type=2)
    anova_by_term = {row["term"]: row for _, row in anova.iterrows()}
    rows: list[dict[str, Any]] = []
    factor_map = {
        "C(model)": "model",
        "C(prompt_id)": "prompt_id",
        "C(temperature)": "temperature",
        "C(task_id)": "task_id",
        "C(run_id)": "run_id",
    }
    for term, col in factor_map.items():
        if col not in df.columns or df[col].nunique() < 2:
            continue
        means = df.groupby(col, observed=True)[value_col].mean().sort_values()
        low = df.loc[df[col] == means.index[0], value_col].to_numpy(dtype=float)
        high = df.loc[df[col] == means.index[-1], value_col].to_numpy(dtype=float)
        cohen = _cohens_d(high, low)
        cliff = _cliffs_delta(high, low)
        anova_row = anova_by_term.get(term, {})
        rows.append(
            {
                "factor": col,
                "cohens_d": cohen,
                "cliffs_delta": cliff,
                "partial_eta_squared": anova_row.get("partial_eta_squared", float("nan")),
                "omega_squared": anova_row.get("omega_squared", float("nan")),
                "p_value_type2": anova_row.get("p_value", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    greater = sum(1 for a in x for b in y if b > a)
    less = sum(1 for a in x for b in y if b < a)
    return float((greater - less) / (len(x) * len(y)))


def bootstrap_ranking_robustness(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
    value_col: str = "metric_value",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap ranking stability with confidence intervals."""
    from caliper.ranking.bootstrap import bootstrap_all_facets
    from caliper.ranking.metrics import ranking_fragility_index

    samples, _baseline_scores, taus_by_facet = bootstrap_all_facets(
        df,
        n_bootstrap=n_bootstrap,
        seed=seed,
        value_col=value_col,
    )
    all_taus = [t for taus in taus_by_facet.values() for t in taus]
    fragility = ranking_fragility_index(all_taus)

    tau_ci = np.percentile(all_taus, [2.5, 50, 97.5]) if all_taus else [float("nan")] * 3

    iter_taus: list[float] = []
    for iteration in sorted(samples["iteration"].unique()):
        sub = samples[samples["iteration"] == iteration]["kendall_tau"].drop_duplicates()
        if not sub.empty:
            iter_taus.append(float(sub.mean()))
    fragility_samples = [(1 - t) / 2 for t in iter_taus]
    frag_ci = np.percentile(fragility_samples, [2.5, 50, 97.5]) if fragility_samples else [float("nan")] * 3

    summary = pd.DataFrame(
        [
            {
                "metric": "kendall_tau",
                "point_estimate": float(np.mean(all_taus)) if all_taus else float("nan"),
                "ci_lower": float(tau_ci[0]),
                "ci_median": float(tau_ci[1]),
                "ci_upper": float(tau_ci[2]),
                "n_bootstrap": n_bootstrap,
            },
            {
                "metric": "fragility_index",
                "point_estimate": fragility,
                "ci_lower": float(frag_ci[0]),
                "ci_median": float(frag_ci[1]),
                "ci_upper": float(frag_ci[2]),
                "n_bootstrap": n_bootstrap,
            },
        ]
    )

    rank_rows: list[dict[str, Any]] = []
    for model, group in samples.groupby("model", observed=True):
        ranks = group["rank"].to_numpy(dtype=float)
        q = np.percentile(ranks, [2.5, 50, 97.5]) if len(ranks) else [float("nan")] * 3
        rank_rows.append(
            {
                "model": model,
                "rank_median": float(q[1]),
                "rank_ci_lower": float(q[0]),
                "rank_ci_upper": float(q[2]),
                "n_bootstrap": n_bootstrap,
            }
        )
    rank_ci = pd.DataFrame(rank_rows).sort_values("rank_median")
    return summary, rank_ci, samples
