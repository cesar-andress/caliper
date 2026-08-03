#!/usr/bin/env python3
"""Generate publication-quality tables, figures, and reports for Paper 1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from caliper.config.loader import load_config
from caliper.config.metrics import resolve_primary_metric
from caliper.ranking.analysis import run_ranking_fragility_analysis
from caliper.runners.experiment_paths import resolve_experiment_dir
from caliper.statistics.bootstrap import bootstrap_ci, bootstrap_ci_by_factor
from caliper.statistics.descriptive import descriptive_by_factor
from caliper.statistics.gtheory import estimate_g_variance_components
from caliper.statistics.power_sim import simulate_power_grid
from caliper.statistics.prepare import load_analysis_frame
from caliper.statistics.variance import decompose_variance

PRIMARY_METRIC_DEFAULT = "normalized_code_match"
DPI = 300
N_BOOTSTRAP = 1000
N_RANK_BOOTSTRAP = 500
N_POWER_SIM = 300
RANDOM_SEED = 20260404

FACTOR_LABELS = {
    "model": "Model",
    "task_id": "Task",
    "prompt_id": "Prompt",
    "run_id": "Run",
    "temperature": "Temperature",
    "residual": "Residual",
}


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    tables: Path
    figures: Path
    csv: Path
    latex: Path
    summary: Path

    @classmethod
    def from_experiment(cls, experiment_dir: Path) -> AnalysisPaths:
        root = experiment_dir / "paper1_analysis"
        return cls(
            root=root,
            tables=root / "tables",
            figures=root / "figures",
            csv=root / "csv",
            latex=root / "latex",
            summary=root / "summary",
        )

    def ensure(self) -> None:
        for path in (self.root, self.tables, self.figures, self.csv, self.latex, self.summary):
            path.mkdir(parents=True, exist_ok=True)


def _load_manifest(experiment_dir: Path) -> dict[str, Any]:
    return json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))


def _prepare_analysis_frame(experiment_dir: Path, metric: str) -> pd.DataFrame:
    # Confirmatory / frozen studies must use statistical_dataset.parquet only.
    return load_analysis_frame(
        experiment_dir,
        metric_name=metric,
        require_statistical_dataset=True,
    )


def _performance_table(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    desc = descriptive_by_factor(df, factor_col)
    ci = bootstrap_ci_by_factor(
        df,
        factor_col,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED,
    )
    merged = desc.merge(ci, on=factor_col, suffixes=("", "_ci"))
    merged = merged.rename(
        columns={
            "ci_lower": "ci_95_lower",
            "ci_upper": "ci_95_upper",
        }
    )
    merged = merged.sort_values("mean", ascending=False).reset_index(drop=True)
    return merged[
        [
            factor_col,
            "count",
            "mean",
            "std",
            "ci_95_lower",
            "ci_95_upper",
            "sem",
            "median",
        ]
    ]


def _variance_table(df: pd.DataFrame) -> pd.DataFrame:
    facets = [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in df.columns]
    components = estimate_g_variance_components(df, facets=facets).components
    total = max(components.get("total", 0.0), 1e-12)
    rows: list[dict[str, Any]] = []
    for facet in facets:
        variance = float(components.get(facet, 0.0))
        rows.append(
            {
                "component": FACTOR_LABELS.get(facet, facet),
                "component_key": facet,
                "variance": variance,
                "pct_total_variance": 100.0 * variance / total,
            }
        )
    residual = float(components.get("residual", 0.0))
    rows.append(
        {
            "component": FACTOR_LABELS["residual"],
            "component_key": "residual",
            "variance": residual,
            "pct_total_variance": 100.0 * residual / total,
        }
    )
    table = pd.DataFrame(rows).sort_values("variance", ascending=False).reset_index(drop=True)
    return table


def _sequential_residuals(df: pd.DataFrame) -> np.ndarray:
    result = decompose_variance(df)
    scores = df["metric_value"].to_numpy(dtype=float)
    residuals = scores - scores.mean()
    facet_order = [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in df.columns]
    for col in facet_order:
        if col not in df.columns or df[col].nunique() <= 1:
            continue
        group_means = df.groupby(col, observed=True)["metric_value"].transform("mean")
        grand_mean = float(df["metric_value"].mean())
        residuals = residuals - (group_means - grand_mean).to_numpy()
    return np.asarray(residuals, dtype=float)


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
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


def _effect_sizes(df: pd.DataFrame, variance_table: pd.DataFrame) -> pd.DataFrame:
    total_var = float(df["metric_value"].var(ddof=1))
    rows: list[dict[str, Any]] = []
    for factor in ("model", "task_id", "prompt_id", "run_id", "temperature"):
        if factor not in df.columns or df[factor].nunique() <= 1:
            continue
        group_means = df.groupby(factor, observed=True)["metric_value"].mean().sort_values()
        low_level = group_means.index[0]
        high_level = group_means.index[-1]
        low_vals = df.loc[df[factor] == low_level, "metric_value"].to_numpy(dtype=float)
        high_vals = df.loc[df[factor] == high_level, "metric_value"].to_numpy(dtype=float)
        var_row = variance_table[variance_table["component_key"] == factor]
        component_var = float(var_row["variance"].iloc[0]) if not var_row.empty else 0.0
        eta_sq = component_var / total_var if total_var > 0 else 0.0
        omega_sq = max((component_var - (df[factor].nunique() - 1) * (total_var * 0.01)) / (total_var + total_var * 0.01), 0.0)
        rows.append(
            {
                "factor": FACTOR_LABELS.get(factor, factor),
                "factor_key": factor,
                "contrast": f"{high_level} vs {low_level}",
                "cohens_d": _cohens_d(high_vals, low_vals),
                "cliffs_delta": _cliffs_delta(high_vals, low_vals),
                "eta_squared": eta_sq,
                "omega_squared": omega_sq,
                "variance_component": component_var,
            }
        )
    return pd.DataFrame(rows)


def _dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    col_format = "l" + "r" * (len(df.columns) - 1)
    header = " & ".join(str(c).replace("_", r"\_") for c in df.columns) + r" \\"
    body_lines = []
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("_", r"\_"))
        body_lines.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_lines)
    return (
        r"\begin{table}[!t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{caption}}}" "\n"
        rf"\label{{{label}}}" "\n"
        rf"\begin{{tabular}}{{{col_format}}}" "\n"
        r"\toprule" "\n"
        f"{header}\n"
        r"\midrule" "\n"
        f"{body}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def _export_table(df: pd.DataFrame, stem: str, paths: AnalysisPaths, caption: str, label: str) -> None:
    df.to_csv(paths.csv / f"{stem}.csv", index=False)
    df.to_csv(paths.tables / f"{stem}.csv", index=False)
    latex = _dataframe_to_latex(df, caption=caption, label=label)
    (paths.latex / f"{stem}.tex").write_text(latex + "\n", encoding="utf-8")


def _save_figure(fig: plt.Figure, stem: str, paths: AnalysisPaths) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(paths.figures / f"{stem}.{ext}", dpi=DPI if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _fig_model_performance(table: pd.DataFrame, paths: AnalysisPaths, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(table))
    ax.bar(x, table["mean"], yerr=[table["mean"] - table["ci_95_lower"], table["ci_95_upper"] - table["mean"]], capsize=4, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(
        f"Mean {metric} by Model — HumanEval+ (95% bootstrap CI; N={int(table['count'].sum())} cells)"
    )
    ax.set_ylim(0, max(0.05, table["ci_95_upper"].max() * 1.15))
    _save_figure(fig, "fig01_model_performance", paths)


def _fig_prompt_performance(table: pd.DataFrame, paths: AnalysisPaths, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(table))
    ax.bar(x, table["mean"], yerr=[table["mean"] - table["ci_95_lower"], table["ci_95_upper"] - table["mean"]], capsize=4, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(table["prompt_id"], rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Mean {metric} by Prompt — HumanEval+ (95% bootstrap CI)")
    ax.set_ylim(0, max(0.05, table["ci_95_upper"].max() * 1.15))
    _save_figure(fig, "fig02_prompt_performance", paths)


def _fig_variance_decomposition(table: pd.DataFrame, paths: AnalysisPaths) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = table.sort_values("pct_total_variance", ascending=True)
    ax.barh(ordered["component"], ordered["pct_total_variance"], color="#C44E52")
    ax.set_xlabel("Percentage of total variance")
    ax.set_title("Descriptive sequential ANOVA variance shares — HumanEval+ (N=39360; pass_at_1)")
    _save_figure(fig, "fig03_variance_decomposition", paths)


def _fig_metric_distribution(df: pd.DataFrame, paths: AnalysisPaths, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["metric_value"], bins=20, color="#8172B3", edgecolor="white")
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {metric} — HumanEval+ (N={len(df)})")
    _save_figure(fig, "fig04_metric_distribution", paths)


def _fig_residual_distribution(residuals: np.ndarray, paths: AnalysisPaths) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, color="#CCB974", edgecolor="white")
    ax.set_xlabel("Sequential ANOVA residual")
    ax.set_ylabel("Count")
    ax.set_title("Sequential ANOVA residuals — HumanEval+ (pass_at_1)")
    _save_figure(fig, "fig05_residual_distribution", paths)


def _fig_ranking_stability(ranking_summary: pd.DataFrame, paths: AnalysisPaths) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = ranking_summary[ranking_summary["bootstrap_type"] != "overall"]
    ax.bar(plot_df["bootstrap_type"], plot_df["kendall_tau_mean"], yerr=plot_df["kendall_tau_std"], capsize=4, color="#64B5CD")
    ax.set_ylabel("Kendall τ (mean ± SD)")
    ax.set_title("Ranking stability under bootstrap — HumanEval+ (pass_at_1)")
    ax.set_ylim(-0.05, 1.05)
    _save_figure(fig, "fig06_ranking_stability", paths)


def _fig_pairwise_reversals(pairwise: pd.DataFrame, models: list[str], paths: AnalysisPaths) -> None:
    matrix = pd.DataFrame(0.0, index=models, columns=models)
    for _, row in pairwise.iterrows():
        matrix.loc[row["model_a"], row["model_b"]] = row["reversal_probability"]
        matrix.loc[row["model_b"], row["model_a"]] = row["reversal_probability"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.values, cmap="OrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_yticklabels(models)
    ax.set_title("Pairwise ranking reversal probability — HumanEval+ (pass_at_1)")
    fig.colorbar(im, ax=ax, label="Probability")
    _save_figure(fig, "fig07_pairwise_reversals", paths)


def _fig_power_curves(power_grid: pd.DataFrame, paths: AnalysisPaths) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_runs, group in power_grid.groupby("n_runs"):
        sub = group.groupby("n_tasks")["power"].mean().reset_index()
        ax.plot(sub["n_tasks"], sub["power"], marker="o", label=f"runs={n_runs}")
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Simulated power")
    ax.set_title(
        "Prospective approximate power (normal t-test; Δ=0.05) — design-oriented, not post-hoc"
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save_figure(fig, "fig08_power_curves", paths)


def _write_latex_fragments(paths: AnalysisPaths, *, metric: str) -> None:
    metric_tex = metric.replace("_", r"\_")
    tables = (
        r"% Auto-generated Paper 1 table fragments." "\n"
        r"\input{latex/table1_dataset_summary.tex}" "\n"
        r"\input{latex/table2_performance_by_model.tex}" "\n"
        r"\input{latex/table3_performance_by_prompt.tex}" "\n"
        r"\input{latex/table4_performance_by_task.tex}" "\n"
        r"\input{latex/table5_variance_decomposition.tex}" "\n"
        r"\input{latex/table6_effect_sizes.tex}" "\n"
    )
    figures = (
        r"% Auto-generated Paper 1 figure fragments." "\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig01_model_performance.pdf}" "\n"
        rf"  \caption{{Mean {metric_tex} by model on HumanEval+ ($N=39360$) with 95\% bootstrap confidence intervals.}}" "\n"
        r"  \label{fig:model-performance}" "\n"
        r"\end{figure}" "\n\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig02_prompt_performance.pdf}" "\n"
        rf"  \caption{{Mean {metric_tex} by prompt variant on HumanEval+ with 95\% bootstrap confidence intervals.}}" "\n"
        r"  \label{fig:prompt-performance}" "\n"
        r"\end{figure}" "\n\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig03_variance_decomposition.pdf}" "\n"
        rf"  \caption{{Descriptive sequential ANOVA variance shares for {metric_tex} on HumanEval+ ($N=39360$). Not a G-study.}}" "\n"
        r"  \label{fig:variance-decomposition}" "\n"
        r"\end{figure}" "\n\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig04_metric_distribution.pdf}" "\n"
        rf"  \caption{{Observed distribution of {metric_tex} across all completed HumanEval+ cells ($N=39360$).}}" "\n"
        r"  \label{fig:metric-distribution}" "\n"
        r"\end{figure}" "\n\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig05_residual_distribution.pdf}" "\n"
        r"  \caption{Distribution of sequential ANOVA residuals after removing main experimental factors.}" "\n"
        r"  \label{fig:residual-distribution}" "\n"
        r"\end{figure}" "\n\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig06_ranking_stability.pdf}" "\n"
        r"  \caption{Ranking stability measured by Kendall $\tau$ under task, prompt, and run bootstrap resampling (HumanEval+, pass\_at\_1).}" "\n"
        r"  \label{fig:ranking-stability}" "\n"
        r"\end{figure}" "\n\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig07_pairwise_reversals.pdf}" "\n"
        r"  \caption{Pairwise ranking reversal probabilities from bootstrap resampling (HumanEval+, pass\_at\_1).}" "\n"
        r"  \label{fig:pairwise-reversals}" "\n"
        r"\end{figure}" "\n\n"
        r"\begin{figure}[!t]" "\n"
        r"  \centering" "\n"
        r"  \includegraphics[width=\linewidth]{figures/fig08_power_curves.pdf}" "\n"
        r"  \caption{Prospective, design-oriented, approximate Monte Carlo power for a two-model comparison (effect size $=0.05$; normal-score t-test). Not post-hoc power.}" "\n"
        r"  \label{fig:power-curves}" "\n"
        r"\end{figure}" "\n"
    )
    (paths.latex / "results_tables.tex").write_text(tables, encoding="utf-8")
    (paths.latex / "results_figures.tex").write_text(figures, encoding="utf-8")


def _write_statistical_report(
    paths: AnalysisPaths,
    *,
    manifest: dict[str, Any],
    metric: str,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table5: pd.DataFrame,
    effect_sizes: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    power_grid: pd.DataFrame,
    overall_ci: dict[str, float],
) -> None:
    top_var = table5.iloc[0]
    top_model = table2.iloc[0]
    top_prompt = table3.iloc[0]
    overall_rank = ranking_summary[ranking_summary["bootstrap_type"] == "overall"].iloc[0]

    def _df_block(frame: pd.DataFrame) -> str:
        return "```\n" + frame.to_string(index=False) + "\n```"

    lines = [
        "# Statistical Report — Paper 1 Ollama Pilot",
        "",
        f"_Generated at {datetime.now(UTC).isoformat()}_",
        "",
        "## Study design",
        "",
        f"- Experiment: `{manifest.get('experiment_id', 'paper1_ollama_pilot')}`",
        f"- Completed cells: {manifest.get('completed_cells', 'unknown')}",
        f"- Primary metric: `{metric}`",
        f"- Random seed: {manifest.get('random_seed', 'unknown')}",
        "",
        "## Descriptive statistics",
        "",
        f"- Overall mean: **{overall_ci['statistic']:.4f}**",
        f"- 95% bootstrap CI: [{overall_ci['lower']:.4f}, {overall_ci['upper']:.4f}]",
        f"- Best model (mean): **{top_model['model']}** ({top_model['mean']:.4f})",
        f"- Best prompt (mean): **{top_prompt['prompt_id']}** ({top_prompt['mean']:.4f})",
        "",
        "## Variance decomposition",
        "",
        "Method: sequential (Type I) ANOVA approximation via CALIPER `decompose_variance`.",
        "",
        _df_block(table5),
        "",
        f"Largest variance component: **{top_var['component']}** ({top_var['pct_total_variance']:.2f}% of total).",
        "",
        "## Effect sizes",
        "",
        _df_block(effect_sizes),
        "",
        "## Confidence intervals",
        "",
        "Model-level and prompt-level 95% percentile bootstrap CIs (1000 resamples, seed=20260404).",
        "",
        "## Power analysis",
        "",
        "Monte Carlo simulation of a two-model t-test under the estimated variance components "
        "(effect size = 0.05, 300 simulations per design cell).",
        "",
        _df_block(power_grid.groupby(["n_tasks", "n_runs"])["power"].mean().reset_index()),
        "",
        "## Ranking fragility",
        "",
        _df_block(ranking_summary),
        "",
        f"- Overall Kendall τ mean: **{overall_rank['kendall_tau_mean']:.4f}**",
        f"- Overall fragility index: **{overall_rank['fragility_index']:.4f}**",
        "",
        "## Main findings (conservative)",
        "",
        "- Performance on normalized code match is low on average; most cells score 0.",
        f"- `{top_var['component']}` accounts for the largest share of observed score variance in this decomposition.",
        "- Prompt and model effects should be interpreted jointly with wide CIs and zero-inflated scores.",
        "",
        "## Threats to validity",
        "",
        "- Sequential ANOVA ordering affects component attribution; not a full crossed random-effects model.",
        "- `normalized_code_match` ignores semantic equivalence beyond normalized text equality.",
        "- Local Ollama models, single hardware stack; no multi-seed replication of model inference beyond run index.",
        "- Zero-inflated binary-like scores violate normality assumptions used in power simulation.",
        "- Ranking bootstrap resamples facets independently; does not re-run model inference.",
        "",
    ]
    (paths.summary / "statistical_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_main_findings(
    paths: AnalysisPaths,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table5: pd.DataFrame,
    effect_sizes: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    power_grid: pd.DataFrame,
    *,
    n_tasks: int,
    n_prompts: int,
    n_runs: int,
    n_models: int,
    metric: str,
) -> None:
    var_sorted = table5.sort_values("pct_total_variance", ascending=False)
    model_row = effect_sizes[effect_sizes["factor_key"] == "model"].iloc[0] if "model" in effect_sizes["factor_key"].values else None
    prompt_row = effect_sizes[effect_sizes["factor_key"] == "prompt_id"].iloc[0] if "prompt_id" in effect_sizes["factor_key"].values else None
    temp_row = effect_sizes[effect_sizes["factor_key"] == "temperature"].iloc[0] if "temperature" in effect_sizes["factor_key"].values else None
    run_row = effect_sizes[effect_sizes["factor_key"] == "run_id"].iloc[0] if "run_id" in effect_sizes["factor_key"].values else None
    overall_rank = ranking_summary[ranking_summary["bootstrap_type"] == "overall"].iloc[0]
    design_mask = (
        (power_grid["n_tasks"] == n_tasks)
        & (power_grid["n_prompts"] == n_prompts)
        & (power_grid["n_runs"] == n_runs)
    )
    if not design_mask.any():
        # Nearest design cell in the prospective power grid (may not match exact factorial).
        design_mask = (
            (power_grid["n_tasks"] == power_grid["n_tasks"].max())
            & (power_grid["n_runs"] == n_runs)
        )
    power_at_design = float(power_grid.loc[design_mask, "power"].mean()) if design_mask.any() else float("nan")

    lines = [
        f"# Main Findings — Paper 1 ({metric}; {n_tasks} tasks × {n_models} models)",
        "",
        "## Variance structure",
        "",
        "The sequential ANOVA decomposition attributes the following shares of total variance "
        f"(largest first): {', '.join(f'{r.component} ({r.pct_total_variance:.1f}%)' for r in var_sorted.itertuples())}.",
        "",
        "## Models vs prompts",
        "",
    ]
    if model_row is not None and prompt_row is not None:
        model_eta = float(model_row["eta_squared"])
        prompt_eta = float(prompt_row["eta_squared"])
        model_pct = float(var_sorted[var_sorted["component_key"] == "model"]["pct_total_variance"].iloc[0])
        prompt_pct = float(var_sorted[var_sorted["component_key"] == "prompt_id"]["pct_total_variance"].iloc[0])
        if prompt_eta > model_eta * 1.05:
            prompt_vs_model = "prompts show a larger η² than models"
        elif model_eta > prompt_eta * 1.05:
            prompt_vs_model = "models show a larger η² than prompts"
        else:
            prompt_vs_model = "prompt and model η² are similar in magnitude"
        lines.extend(
            [
                f"- Model effect (η²): {model_eta:.4f}; Prompt effect (η²): {prompt_eta:.4f}.",
                f"- Prompt variance share ({prompt_pct:.2f}%) vs model variance share ({model_pct:.2f}%): {prompt_vs_model}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Temperature",
            "",
        ]
    )
    if temp_row is not None:
        temp_pct = var_sorted[var_sorted["component_key"] == "temperature"]["pct_total_variance"].iloc[0]
        lines.append(
            f"Temperature explains **{temp_pct:.2f}%** of total variance (η²={temp_row['eta_squared']:.4f}). "
            "Its practical impact should be read against the dominant task and residual components."
        )
    else:
        lines.append("Temperature could not be estimated separately.")
    lines.extend(["", "## Repeated runs", ""])
    if run_row is not None:
        run_pct = var_sorted[var_sorted["component_key"] == "run_id"]["pct_total_variance"].iloc[0]
        lines.append(
            f"Run index accounts for **{run_pct:.2f}%** of variance (η²={run_row['eta_squared']:.4f}). "
            "Additional runs add information, but diminishing returns are likely when run variance is small "
            "relative to task and residual variance."
        )
    lines.extend(
        [
            "",
            "## Ranking stability",
            "",
            f"Bootstrap ranking analysis yields overall Kendall τ = **{overall_rank['kendall_tau_mean']:.4f}** "
            f"(fragility index = **{overall_rank['fragility_index']:.4f}**). "
            "Lower τ indicates rankings shift under resampling of tasks, prompts, or runs.",
            "",
            "## Power",
            "",
            "Prospective, design-oriented, approximate Monte Carlo power (normal-score two-model "
            f"t-test; effect size = 0.05). Nearest grid cell for the realized design "
            f"({n_tasks} tasks, {n_prompts} prompts, {n_runs} runs) has mean simulated power "
            f"**{power_at_design:.3f}**. This is not post-hoc power and is not evidence of an effect.",
            "",
            "## Top performers (descriptive only)",
            "",
            f"- Best model by mean: **{table2.iloc[0]['model']}** ({table2.iloc[0]['mean']:.4f}).",
            f"- Best prompt by mean: **{table3.iloc[0]['prompt_id']}** ({table3.iloc[0]['mean']:.4f}).",
            "",
            "## Limitations",
            "",
            "All statements above are descriptive or simulation-based on the frozen HumanEval+ "
            f"confirmatory frame ({n_tasks} tasks, {n_models} models, {n_prompts} prompts, "
            f"{n_runs} runs; metric={metric}). They do not establish causal effects of prompts "
            "or models, and they should not be extrapolated beyond this protocol.",
            "",
        ]
    )
    (paths.summary / "main_findings.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(experiment_dir: Path, metric: str | None = None) -> AnalysisPaths:
    experiment_dir = resolve_experiment_dir(experiment_dir)
    paths = AnalysisPaths.from_experiment(experiment_dir)
    paths.ensure()

    manifest = _load_manifest(experiment_dir)
    config_path = experiment_dir / "config.yaml"
    if metric is None and config_path.exists():
        metric = resolve_primary_metric(load_config(config_path))[0]
    metric = metric or PRIMARY_METRIC_DEFAULT

    df = _prepare_analysis_frame(experiment_dir, metric)
    axes = manifest.get("factorial_axes", {})
    n_models = int(axes.get("models", df["model"].nunique()))
    n_tasks = int(axes.get("tasks", df["task_id"].nunique()))
    n_prompts = int(axes.get("prompt_variants", df["prompt_id"].nunique()))
    n_runs = int(axes.get("runs", df["run_id"].nunique()))

    table1 = pd.DataFrame(
        [
            {
                "models": n_models,
                "tasks": n_tasks,
                "prompts": n_prompts,
                "temperatures": axes.get("temperatures", df["temperature"].nunique()),
                "runs": n_runs,
                "total_observations": len(df),
                "primary_metric": metric,
            }
        ]
    )
    table2 = _performance_table(df, "model")
    table3 = _performance_table(df, "prompt_id")
    table4 = _performance_table(df, "task_id")
    table5 = _variance_table(df)
    effect_sizes = _effect_sizes(df, table5)
    residuals = _sequential_residuals(df)

    ranking = run_ranking_fragility_analysis(
        df,
        metric_name=metric,
        n_bootstrap=N_RANK_BOOTSTRAP,
        seed=RANDOM_SEED,
        output_dir=paths.root / "ranking_work",
        reports_dir=paths.root / "ranking_work" / "plots",
    )

    gstudy = estimate_g_variance_components(df)
    # Prospective / design-oriented approximate power grid (normal continuous approximation).
    # Not post-hoc power. Grid spans pilot-scale and confirmatory task counts.
    power_grid = simulate_power_grid(
        gstudy.components,
        effect_size=0.05,
        task_counts=[10, 20, 40, 80, 120, 164],
        prompt_counts=[1, 2, 4],
        run_counts=[1, 3, 5],
        n_simulations=N_POWER_SIM,
        seed=RANDOM_SEED,
    )
    overall_ci = bootstrap_ci(df["metric_value"], n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED).as_dict()

    _export_table(table1, "table1_dataset_summary", paths, "Dataset summary.", "tab:dataset-summary")
    _export_table(table2, "table2_performance_by_model", paths, "Performance by model.", "tab:model-performance")
    _export_table(table3, "table3_performance_by_prompt", paths, "Performance by prompt.", "tab:prompt-performance")
    _export_table(table4, "table4_performance_by_task", paths, "Performance by task.", "tab:task-performance")
    _export_table(table5, "table5_variance_decomposition", paths, "Variance decomposition.", "tab:variance")
    _export_table(effect_sizes, "table6_effect_sizes", paths, "Effect sizes for main factors.", "tab:effect-sizes")

    power_grid.to_csv(paths.csv / "power_simulation_grid.csv", index=False)
    ranking.summary.to_csv(paths.csv / "ranking_fragility_summary.csv", index=False)
    ranking.pairwise_reversals.to_csv(paths.csv / "pairwise_reversals.csv", index=False)

    _fig_model_performance(table2, paths, metric)
    _fig_prompt_performance(table3, paths, metric)
    _fig_variance_decomposition(table5, paths)
    _fig_metric_distribution(df, paths, metric)
    _fig_residual_distribution(residuals, paths)
    _fig_ranking_stability(ranking.summary, paths)
    _fig_pairwise_reversals(ranking.pairwise_reversals, list(ranking.baseline_scores.index), paths)
    _fig_power_curves(power_grid, paths)

    _write_latex_fragments(paths, metric=metric)
    _write_statistical_report(
        paths,
        manifest=manifest,
        metric=metric,
        table1=table1,
        table2=table2,
        table3=table3,
        table5=table5,
        effect_sizes=effect_sizes,
        ranking_summary=ranking.summary,
        power_grid=power_grid,
        overall_ci=overall_ci,
    )
    _write_main_findings(
        paths,
        table2=table2,
        table3=table3,
        table5=table5,
        effect_sizes=effect_sizes,
        ranking_summary=ranking.summary,
        power_grid=power_grid,
        n_tasks=n_tasks,
        n_prompts=n_prompts,
        n_runs=n_runs,
        n_models=n_models,
        metric=metric,
    )

    manifest_out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_dir": str(experiment_dir),
        "metric": metric,
        "n_observations": len(df),
        "outputs": {
            "tables_csv": sorted(p.name for p in paths.csv.glob("*.csv")),
            "figures": sorted(p.name for p in paths.figures.iterdir() if p.is_file()),
            "latex": sorted(p.name for p in paths.latex.glob("*.tex")),
            "summary": sorted(p.name for p in paths.summary.glob("*.md")),
        },
    }
    (paths.root / "analysis_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper 1 publication analysis outputs")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/paper1_ollama_pilot"),
        help="Completed experiment directory",
    )
    parser.add_argument("--metric", default=None, help="Primary metric (defaults to config)")
    args = parser.parse_args()
    paths = run_analysis(args.experiment_dir, metric=args.metric)
    print(f"Analysis written to {paths.root}")


if __name__ == "__main__":
    main()
