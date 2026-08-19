"""Scientific comparisons: A vs freeze, B vs freeze, A vs B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ANALYSIS_DIR, FACTOR_KEYS, RANDOM_SEED
from .loaders import join_key_frame


def _safe_mean(s: pd.Series) -> float | None:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if len(s) else None


def panel_metrics(df: pd.DataFrame, label: str) -> dict[str, Any]:
    empty = df["empty"].astype(bool)
    return {
        "panel": label,
        "n": int(len(df)),
        "pass_at_1_mean": _safe_mean(df["pass_at_1"]),
        "syntax_validity_mean": _safe_mean(df["syntax_check"]),
        "empty_response_rate": float(empty.mean()),
        "budget_exhaustion_rate": float(df["budget_exhausted"].mean())
        if "budget_exhausted" in df.columns
        else None,
        "latency_mean_ms": _safe_mean(df["latency_ms"]),
        "latency_median_ms": float(pd.to_numeric(df["latency_ms"], errors="coerce").median()),
        "thinking_length_mean": _safe_mean(df.get("thinking_length", pd.Series(dtype=float))),
        "thinking_length_median": float(
            pd.to_numeric(df.get("thinking_length", pd.Series(dtype=float)), errors="coerce").median()
        )
        if "thinking_length" in df.columns
        else None,
        "eval_count_mean": _safe_mean(df.get("eval_count", pd.Series(dtype=float))),
        "eval_count_median": float(
            pd.to_numeric(df.get("eval_count", pd.Series(dtype=float)), errors="coerce").median()
        )
        if "eval_count" in df.columns and df["eval_count"].notna().any()
        else None,
        "visible_chars_mean": _safe_mean(df["visible_chars"]),
        "visible_chars_median": float(df["visible_chars"].median()),
        "done_reason_counts": df["done_reason"].fillna("null").astype(str).value_counts().to_dict()
        if "done_reason" in df.columns
        else {},
    }


def _agreement_table(left: pd.DataFrame, right: pd.DataFrame, level: str) -> pd.DataFrame:
    """Mean metric agreement after grouping by a factor level."""
    keys = {
        "task": ["task_id"],
        "prompt": ["prompt_id"],
        "temperature": ["temperature"],
        "run": ["run_index"],
    }[level]
    l = left.groupby(keys, as_index=False).agg(
        pass_l=("pass_at_1", "mean"),
        empty_l=("empty", "mean"),
    )
    r = right.groupby(keys, as_index=False).agg(
        pass_r=("pass_at_1", "mean"),
        empty_r=("empty", "mean"),
    )
    m = l.merge(r, on=keys, how="inner")
    m["pass_delta"] = m["pass_r"] - m["pass_l"]
    m["empty_delta"] = m["empty_r"] - m["empty_l"]
    m["level"] = level
    return m


def paired_cell_frame(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    L = join_key_frame(left)
    R = join_key_frame(right)
    # Prefer cell_id join when both have it and overlap is large
    if "cell_id" in L.columns and "cell_id" in R.columns:
        overlap = len(set(L["cell_id"]) & set(R["cell_id"]))
        if overlap >= 0.95 * min(len(L), len(R)):
            cols_r = ["cell_id", "pass_at_1", "syntax_check", "empty", "budget_exhausted",
                      "latency_ms", "thinking_length", "eval_count", "visible_chars", "done_reason",
                      "prediction"]
            cols_r = [c for c in cols_r if c in R.columns]
            merged = L.merge(R[cols_r], on="cell_id", how="inner", suffixes=(f"_{left_name}", f"_{right_name}"))
            return merged
    cols = list(FACTOR_KEYS) + ["model"]
    keep_l = cols + [
        c
        for c in [
            "cell_id",
            "pass_at_1",
            "syntax_check",
            "empty",
            "budget_exhausted",
            "latency_ms",
            "thinking_length",
            "eval_count",
            "visible_chars",
            "done_reason",
            "prediction",
            "factor_key",
        ]
        if c in L.columns
    ]
    keep_r = [c for c in keep_l if c in R.columns]
    return L[keep_l].merge(R[keep_r], on=cols, how="inner", suffixes=(f"_{left_name}", f"_{right_name}"))


def compare_panels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    paired = paired_cell_frame(left, right, left_name, right_name)
    out: dict[str, Any] = {
        "left": panel_metrics(left, left_name),
        "right": panel_metrics(right, right_name),
        "n_paired": int(len(paired)),
        "agreements": {},
    }
    # Cell-level deltas when paired
    pl = f"pass_at_1_{left_name}"
    pr = f"pass_at_1_{right_name}"
    el = f"empty_{left_name}"
    er = f"empty_{right_name}"
    if pl in paired.columns and pr in paired.columns:
        out["paired_pass_delta_mean"] = float((paired[pr] - paired[pl]).mean())
        out["paired_pass_agreement_rate"] = float((paired[pl] == paired[pr]).mean())
    if el in paired.columns and er in paired.columns:
        out["paired_empty_delta_mean"] = float((paired[er].astype(float) - paired[el].astype(float)).mean())
        out["paired_empty_agreement_rate"] = float((paired[el] == paired[er]).mean())

    for level in ("task", "prompt", "temperature", "run"):
        # rebuild left/right with consistent names
        l2 = left.copy()
        r2 = right.copy()
        table = _agreement_table(l2, r2, level)
        out["agreements"][level] = {
            "n_groups": int(len(table)),
            "pass_delta_mean": float(table["pass_delta"].mean()) if len(table) else None,
            "pass_delta_std": float(table["pass_delta"].std()) if len(table) else None,
            "empty_delta_mean": float(table["empty_delta"].mean()) if len(table) else None,
            "spearman_pass": float(table["pass_l"].corr(table["pass_r"], method="spearman"))
            if len(table) > 2
            else None,
        }
        out[f"agreement_table_{level}"] = table
    out["paired"] = paired
    return out


def run_scientific_validation(
    freeze: pd.DataFrame,
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or (ANALYSIS_DIR / "scientific")
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    comparisons = {
        "A_vs_freeze": compare_panels(freeze, arm_a, left_name="freeze", right_name="A"),
        "B_vs_freeze": compare_panels(freeze, arm_b, left_name="freeze", right_name="B"),
        "A_vs_B": compare_panels(arm_a, arm_b, left_name="A", right_name="B"),
    }

    summary_rows = []
    for name, cmp_ in comparisons.items():
        summary_rows.append(
            {
                "comparison": name,
                "n_paired": cmp_["n_paired"],
                "left_pass": cmp_["left"]["pass_at_1_mean"],
                "right_pass": cmp_["right"]["pass_at_1_mean"],
                "left_empty": cmp_["left"]["empty_response_rate"],
                "right_empty": cmp_["right"]["empty_response_rate"],
                "left_syntax": cmp_["left"]["syntax_validity_mean"],
                "right_syntax": cmp_["right"]["syntax_validity_mean"],
                "left_latency_median_ms": cmp_["left"]["latency_median_ms"],
                "right_latency_median_ms": cmp_["right"]["latency_median_ms"],
                "left_thinking_mean": cmp_["left"]["thinking_length_mean"],
                "right_thinking_mean": cmp_["right"]["thinking_length_mean"],
                "left_eval_count_mean": cmp_["left"]["eval_count_mean"],
                "right_eval_count_mean": cmp_["right"]["eval_count_mean"],
                "paired_pass_agreement": cmp_.get("paired_pass_agreement_rate"),
                "paired_empty_agreement": cmp_.get("paired_empty_agreement_rate"),
            }
        )
        for level in ("task", "prompt", "temperature", "run"):
            table = cmp_[f"agreement_table_{level}"]
            table.to_csv(tables / f"{name}_agreement_{level}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "comparison_summary.csv", index=False)

    panel_rows = [comparisons["A_vs_freeze"]["left"], comparisons["A_vs_freeze"]["right"],
                  comparisons["B_vs_freeze"]["right"]]
    # dedupe freeze
    panels = pd.DataFrame(
        [
            comparisons["A_vs_freeze"]["left"],
            comparisons["A_vs_freeze"]["right"],
            comparisons["B_vs_freeze"]["right"],
        ]
    )
    panels.to_csv(tables / "panel_metrics.csv", index=False)

    # JSON without heavy frames
    serializable = {}
    for name, cmp_ in comparisons.items():
        serializable[name] = {
            k: v
            for k, v in cmp_.items()
            if k not in {"paired"} and not k.startswith("agreement_table_")
        }
    (out_dir / "scientific_validation_summary.json").write_text(
        json.dumps(serializable, indent=2, default=str),
        encoding="utf-8",
    )

    # Persist paired frames for statistical tests
    for name, cmp_ in comparisons.items():
        cmp_["paired"].to_parquet(tables / f"{name}_paired.parquet", index=False)

    return {
        "summary": summary,
        "comparisons": comparisons,
        "out_dir": out_dir,
    }
