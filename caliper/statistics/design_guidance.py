"""Actionable design guidance for Paper 1 (evidence-supported only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from caliper.runners.experiment_paths import resolve_experiment_dir
from caliper.statistics.gtheory import estimate_g_variance_components, simulate_d_study_grid
from caliper.statistics.prepare import load_analysis_frame
from caliper.statistics.variance import decompose_variance

PLACEHOLDER_ROWS = [
    {
        "recommendation": "minimum_tasks_for_g_0_80",
        "value": None,
        "status": "not_supported",
        "notes": "Current D-study G estimator is invalid (flat G across designs).",
    },
    {
        "recommendation": "minimum_tasks_for_g_0_90",
        "value": None,
        "status": "not_supported",
        "notes": "Current D-study G estimator is invalid (flat G across designs).",
    },
    {
        "recommendation": "minimum_tasks_for_kendall_tau_0_90",
        "value": None,
        "status": "pending_task_sampling",
        "notes": "Populate from task-sampling analysis after full benchmark completes.",
    },
    {
        "recommendation": "minimum_tasks_for_kendall_tau_0_95",
        "value": None,
        "status": "pending_task_sampling",
        "notes": "Populate from task-sampling analysis after full benchmark completes.",
    },
    {
        "recommendation": "marginal_benefit_additional_prompts",
        "value": None,
        "status": "pending_experiment_completion",
        "notes": "Use descriptive prompt variance share / effect size, not defective G.",
    },
    {
        "recommendation": "marginal_benefit_additional_runs",
        "value": None,
        "status": "pending_experiment_completion",
        "notes": "Use descriptive run variance share / effect size, not defective G.",
    },
]


@dataclass(frozen=True)
class DesignGuidancePaths:
    root: Path

    @classmethod
    def from_experiment(cls, experiment_dir: Path) -> DesignGuidancePaths:
        return cls(root=experiment_dir / "paper1_analysis" / "design_guidance")

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


def _load_results_frame(experiment_dir: Path, metric: str = "pass_at_1") -> pd.DataFrame | None:
    stats_path = experiment_dir / "statistical_dataset.parquet"
    if not stats_path.exists():
        return None
    return load_analysis_frame(
        experiment_dir,
        metric_name=metric,
        require_statistical_dataset=True,
    )


def _resolve_task_sampling_table(experiment_dir: Path, explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    base = experiment_dir / "paper1_analysis" / "task_sampling" / "csv"
    candidates.extend(
        [
            base / "table_recommended_task_counts.csv",
            base / "task_count_recommendations.csv",
            base / "reliability_thresholds.csv",
        ]
    )
    for path in candidates:
        if path is not None and path.exists():
            return path
    return None


def _kendall_recommendations_from_csv(path: Path) -> list[dict[str, Any]]:
    table = pd.read_csv(path)
    rows: list[dict[str, Any]] = []

    if {"metric", "target", "recommended_min_tasks_80pct_subsets"}.issubset(table.columns):
        for _, rec in table.iterrows():
            if str(rec["metric"]) != "kendall_tau":
                continue
            rows.append(
                {
                    "recommendation": f"minimum_tasks_for_kendall_tau_{str(rec['target']).replace('.', '_')}",
                    "value": rec.get("recommended_min_tasks_80pct_subsets"),
                    "status": "estimated",
                    "notes": (
                        "From repeated task-subset simulation on full HumanEval+ "
                        "(B=1000, seed=20260404); ≥80% of subsets meeting τ target."
                    ),
                }
            )
        return rows

    if {"kendall_tau_target", "min_tasks_80pct_subsets"}.issubset(table.columns):
        for _, rec in table.iterrows():
            target = str(rec["kendall_tau_target"]).replace(".", "_")
            rows.append(
                {
                    "recommendation": f"minimum_tasks_for_kendall_tau_{target}",
                    "value": rec.get("min_tasks_80pct_subsets"),
                    "status": "estimated",
                    "notes": (
                        "From repeated task-subset simulation on full HumanEval+ "
                        "(B=1000, seed=20260404); ≥80% of subsets meeting τ target."
                    ),
                }
            )
        return rows

    if {"metric", "target", "n_tasks", "fraction_meeting_target"}.issubset(table.columns):
        recs = table[table["metric"] == "kendall_tau_recommendation"]
        for _, rec in recs.iterrows():
            target = str(rec["target"]).replace(".", "_")
            rows.append(
                {
                    "recommendation": f"minimum_tasks_for_kendall_tau_{target}",
                    "value": rec.get("n_tasks"),
                    "status": "estimated",
                    "notes": (
                        "From repeated task-subset simulation on full HumanEval+ "
                        "(B=1000, seed=20260404); ≥80% of subsets meeting τ target."
                    ),
                }
            )
        return rows

    return rows


def build_design_recommendations(
    experiment_dir: Path | str,
    *,
    metric: str = "pass_at_1",
    task_sampling_recommendations: Path | None = None,
) -> pd.DataFrame:
    """Build design guidance table from completed analysis outputs when available."""
    experiment_dir = Path(experiment_dir)
    try:
        experiment_dir = resolve_experiment_dir(experiment_dir)
    except Exception:
        return pd.DataFrame(PLACEHOLDER_ROWS)

    frame = _load_results_frame(experiment_dir, metric=metric)
    rows: list[dict[str, Any]] = []

    if frame is None or frame.empty:
        return pd.DataFrame(PLACEHOLDER_ROWS)

    # Retain D-study grid export for audit, but do not treat flat G as actionable.
    components = estimate_g_variance_components(frame).components
    grid = simulate_d_study_grid(
        components,
        task_counts=[10, 20, 40, 60, 80, 100, 120, 140, 164],
        prompt_counts=[1, 2, 4],
        run_counts=[1, 3, 5],
    )
    guidance_dir = experiment_dir / "paper1_analysis" / "design_guidance"
    guidance_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(guidance_dir / "d_study_grid.csv", index=False)
    g_unique = int(grid["g_coefficient"].nunique(dropna=True))
    rows.extend(
        [
            {
                "recommendation": "minimum_tasks_for_g_0_80",
                "value": None,
                "status": "not_supported",
                "notes": (
                    f"Invalid: D-study G is constant across designs "
                    f"(nunique={g_unique}; compute_g_coefficient ignores replication counts)."
                ),
            },
            {
                "recommendation": "minimum_tasks_for_g_0_90",
                "value": None,
                "status": "not_supported",
                "notes": (
                    f"Invalid: D-study G is constant across designs "
                    f"(nunique={g_unique}; compute_g_coefficient ignores replication counts)."
                ),
            },
        ]
    )

    # Marginal benefit from descriptive sequential ANOVA shares (not G-theory).
    vc = decompose_variance(frame)
    total = max(vc.total_variance, 1e-12)
    rows.extend(
        [
            {
                "recommendation": "marginal_benefit_additional_prompts",
                "value": json.dumps(
                    {
                        "prompt_variance_share_pct": round(100.0 * vc.prompt_variance / total, 4),
                        "prompt_variance": vc.prompt_variance,
                        "interpretation": (
                            "Near-zero descriptive prompt share under this HumanEval+ protocol; "
                            "additional prompts are unlikely to move pass@1 means."
                        ),
                    }
                ),
                "status": "estimated_descriptive_anova",
                "notes": (
                    "Scope: HumanEval+ confirmatory protocol only. "
                    "Not a G-theory marginal reliability gain."
                ),
            },
            {
                "recommendation": "marginal_benefit_additional_runs",
                "value": json.dumps(
                    {
                        "run_variance_share_pct": round(100.0 * vc.run_variance / total, 4),
                        "run_variance": vc.run_variance,
                        "interpretation": (
                            "Near-zero descriptive run share; extra repetitions add little "
                            "beyond Monte Carlo noise reduction for pass@1 means."
                        ),
                    }
                ),
                "status": "estimated_descriptive_anova",
                "notes": (
                    "Scope: HumanEval+ confirmatory protocol only. "
                    "Not a G-theory marginal reliability gain."
                ),
            },
        ]
    )

    ts_path = _resolve_task_sampling_table(experiment_dir, task_sampling_recommendations)
    kendall_rows = _kendall_recommendations_from_csv(ts_path) if ts_path is not None else []
    if kendall_rows:
        rows.extend(kendall_rows)
    else:
        rows.extend(
            row
            for row in PLACEHOLDER_ROWS
            if row["recommendation"].startswith("minimum_tasks_for_kendall")
        )

    rows.append(
        {
            "recommendation": "scope_note",
            "value": "HumanEval+_pass_at_1_factorial_protocol",
            "status": "stated",
            "notes": (
                "All recommendations are restricted to the evaluated HumanEval+ "
                "6×164×4×2×5 design with metric=pass_at_1; do not generalize to other benchmarks."
            ),
        }
    )

    return pd.DataFrame(rows)


def export_design_guidance(
    experiment_dir: Path | str,
    *,
    metric: str = "pass_at_1",
) -> DesignGuidancePaths:
    experiment_path = Path(experiment_dir)
    try:
        resolved = resolve_experiment_dir(experiment_dir)
    except Exception:
        resolved = experiment_path
    paths = DesignGuidancePaths.from_experiment(resolved)
    paths.ensure()
    table = build_design_recommendations(resolved, metric=metric)
    csv_path = paths.root / "table_design_recommendations.csv"
    table.to_csv(csv_path, index=False)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Design guidance from task-sampling and descriptive variance analyses "
        r"(G-theory numeric thresholds not supported).}",
        r"\label{tab:design-recommendations}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Recommendation & Value & Status \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        value = "" if pd.isna(row["value"]) else str(row["value"]).replace("%", r"\%").replace("_", r"\_")
        tex_lines.append(
            f"{str(row['recommendation']).replace('_', r'_')} & {value} & "
            f"{str(row['status']).replace('_', r'_')} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (paths.root / "table_design_recommendations.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# Design guidance",
        "",
        "Evidence-supported recommendations for the frozen HumanEval+ confirmatory protocol.",
        "G-theory numeric thresholds are explicitly **not supported** by the current estimator.",
        "",
        "| recommendation | value | status | notes |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in table.iterrows():
        value = "" if pd.isna(row.get("value")) else str(row["value"])
        notes = str(row.get("notes", ""))
        md_lines.append(
            f"| {row['recommendation']} | {value} | {row['status']} | {notes} |"
        )
    md_lines.append("")
    (paths.root / "design_guidance.md").write_text("\n".join(md_lines), encoding="utf-8")
    return paths
