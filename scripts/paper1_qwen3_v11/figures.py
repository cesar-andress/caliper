"""Publication-quality figures for qwen3 v1.1 corrective analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import ANALYSIS_DIR, DPI


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=DPI if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def fig_panel_bars(panel_metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    metrics = [
        ("pass_at_1_mean", "pass@1"),
        ("empty_response_rate", "empty rate"),
        ("latency_median_ms", "median latency (ms)"),
    ]
    x = np.arange(len(panel_metrics))
    labels = panel_metrics["panel"].tolist()
    for ax, (col, title) in zip(axes, metrics):
        vals = panel_metrics[col].to_numpy(dtype=float)
        ax.bar(x, vals, color=["#4C78A8", "#F58518", "#54A24B"][: len(vals)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_panel_metrics", out_dir)


def fig_task_scatter(task_delta: pd.DataFrame, out_dir: Path, stem: str, ycol: str = "pass_delta") -> None:
    if task_delta.empty or "difficulty" not in task_delta.columns:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(task_delta["difficulty"], task_delta[ycol], alpha=0.65, s=22, edgecolor="none")
    ax.set_xlabel("Task difficulty proxy (1 − freeze pass@1)")
    ax.set_ylabel("Pass@1 delta (right − left)")
    ax.axhline(0, color="black", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, stem, out_dir)


def fig_bland_altman(paired: pd.DataFrame, left: str, right: str, out_dir: Path, stem: str) -> None:
    pl = paired[f"pass_at_1_{left}"].to_numpy(dtype=float)
    pr = paired[f"pass_at_1_{right}"].to_numpy(dtype=float)
    # For binary pass, BA is coarse; plot task-aggregated means instead if available
    if "task_id" in paired.columns:
        g = paired.groupby("task_id").agg(l=(f"pass_at_1_{left}", "mean"), r=(f"pass_at_1_{right}", "mean"))
        mean = (g["l"] + g["r"]) / 2
        diff = g["r"] - g["l"]
    else:
        mean = (pl + pr) / 2
        diff = pr - pl
    md = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(mean, diff, alpha=0.65, s=22, edgecolor="none")
    ax.axhline(md, color="black", lw=1)
    ax.axhline(md - 1.96 * sd, color="gray", lw=1, ls="--")
    ax.axhline(md + 1.96 * sd, color="gray", lw=1, ls="--")
    ax.set_xlabel(f"Mean pass@1 ({left}, {right})")
    ax.set_ylabel(f"Pass@1 difference ({right} − {left})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, stem, out_dir)


def fig_empty_vs_pass(panel_metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(
        panel_metrics["empty_response_rate"],
        panel_metrics["pass_at_1_mean"],
        s=80,
    )
    for _, row in panel_metrics.iterrows():
        ax.annotate(row["panel"], (row["empty_response_rate"], row["pass_at_1_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Empty response rate")
    ax.set_ylabel("pass@1 mean")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_empty_vs_pass", out_dir)


def fig_done_reason(arm_a: pd.DataFrame, arm_b: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, df, title in [
        (axes[0], arm_a, "Arm A"),
        (axes[1], arm_b, "Arm B"),
    ]:
        counts = df["done_reason"].fillna("null").astype(str).value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#4C78A8")
        ax.set_title(title)
        ax.set_ylabel("cells")
        ax.tick_params(axis="x", rotation=30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_done_reason", out_dir)


def generate_all_figures(
    panel_metrics: pd.DataFrame,
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    paired_a: pd.DataFrame,
    paired_b: pd.DataFrame,
    paired_ab: pd.DataFrame,
    hetero_tables_dir: Path,
    out_dir: Path | None = None,
) -> list[str]:
    out_dir = out_dir or (ANALYSIS_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_panel_bars(panel_metrics, out_dir)
    fig_empty_vs_pass(panel_metrics, out_dir)
    fig_done_reason(arm_a, arm_b, out_dir)
    fig_bland_altman(paired_a, "freeze", "A", out_dir, "fig_bland_altman_A_vs_freeze")
    fig_bland_altman(paired_b, "freeze", "B", out_dir, "fig_bland_altman_B_vs_freeze")
    fig_bland_altman(paired_ab, "A", "B", out_dir, "fig_bland_altman_A_vs_B")

    task_a = hetero_tables_dir / "A_vs_freeze_task_pass_delta.csv"
    if task_a.exists():
        fig_task_scatter(pd.read_csv(task_a), out_dir, "fig_difficulty_vs_delta_A")
    task_b = hetero_tables_dir / "B_vs_freeze_task_pass_delta.csv"
    if task_b.exists():
        fig_task_scatter(pd.read_csv(task_b), out_dir, "fig_difficulty_vs_delta_B")

    return sorted(p.name for p in out_dir.glob("fig_*"))
