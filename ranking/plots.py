"""Matplotlib plots for ranking fragility analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_kendall_tau_distribution(
    bootstrap_samples: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Kendall τ vs Baseline Ranking",
) -> Path:
    """Save histogram of Kendall tau by bootstrap type."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for btype, group in bootstrap_samples.groupby("bootstrap_type", observed=True):
        taus = group.drop_duplicates(subset=["iteration"])["kendall_tau"]
        ax.hist(taus, bins=30, alpha=0.5, label=str(btype), density=True)

    ax.set_xlabel("Kendall τ")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_rank_probability_heatmap(
    rank_probs: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "P(Model Rank | Bootstrap)",
) -> Path:
    """Save heatmap of rank occupancy probabilities."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, rank_probs.shape[1]), max(4, rank_probs.shape[0] * 0.5)))

    im = ax.imshow(rank_probs.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(rank_probs.shape[1]))
    ax.set_xticklabels([str(c) for c in rank_probs.columns])
    ax.set_yticks(range(rank_probs.shape[0]))
    ax.set_yticklabels(rank_probs.index.tolist())
    ax.set_xlabel("Rank")
    ax.set_ylabel("Model")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_pairwise_reversal_heatmap(
    pairwise: pd.DataFrame,
    models: list[str],
    output_path: Path,
    *,
    title: str = "Pairwise Rank Reversal Probability",
) -> Path:
    """Save heatmap of pairwise reversal probabilities."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(models)
    matrix = pd.DataFrame(0.0, index=models, columns=models)

    for _, row in pairwise.iterrows():
        a, b, prob = row["model_a"], row["model_b"], row["reversal_probability"]
        matrix.loc[a, b] = prob
        matrix.loc[b, a] = prob

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n * 0.6)))
    im = ax.imshow(matrix.values, cmap="OrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(models)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Reversal probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_baseline_rankings(
    baseline_scores: pd.Series,
    output_path: Path,
    *,
    title: str = "Baseline Model Rankings",
) -> Path:
    """Save bar chart of baseline mean scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = baseline_scores.sort_values(ascending=True)
    ax.barh(ordered.index.astype(str), ordered.values)
    ax.set_xlabel("Mean score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
