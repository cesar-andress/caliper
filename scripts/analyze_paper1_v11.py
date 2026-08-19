#!/usr/bin/env python3
"""Compare Paper1 v1.0 frozen qwen3 vs v1.1 diagnostic arm(s).

Separates implementation artifact metrics (empty-rate, done_reason) from
scientific metrics (pass@1 on regenerated protocol).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caliper.statistics.glmm_analysis import run_pass_fail_glmm_analysis
from caliper.statistics.gtheory import estimate_g_variance_components


def _empty_mask(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.len() == 0


def _model_col(df: pd.DataFrame) -> str:
    return "model" if "model" in df.columns else "model_id"


def _value_col(df: pd.DataFrame) -> str:
    return "metric_value" if "metric_value" in df.columns else "score"


def _prompt_col(df: pd.DataFrame) -> str:
    return "prompt_id" if "prompt_id" in df.columns else "prompt_variant_id"


def summarize_panel(df: pd.DataFrame, label: str) -> dict:
    mcol = _model_col(df)
    vcol = _value_col(df)
    rows = []
    for model, g in df.groupby(mcol):
        empty = _empty_mask(g["prediction"])
        rows.append(
            {
                "panel": label,
                "model": model,
                "n": int(len(g)),
                "empty_rate": float(empty.mean()),
                "empty_n": int(empty.sum()),
                "pass_mean": float(g[vcol].mean()),
                "pass_mean_nonempty": float(g.loc[~empty, vcol].mean()) if (~empty).any() else None,
                "latency_median_ms": float(g["latency_ms"].median()),
                "budget_exhausted_n": int((g["status"] == "budget_exhausted").sum())
                if "status" in g
                else None,
            }
        )
    return {"label": label, "models": rows}


def compare_qwen3(v10: pd.DataFrame, v11: pd.DataFrame) -> dict:
    mcol10 = _model_col(v10)
    mcol11 = _model_col(v11)
    vcol10 = _value_col(v10)
    vcol11 = _value_col(v11)
    q10 = v10[v10[mcol10] == "qwen3_32b"]
    q11 = v11[v11[mcol11] == "qwen3_32b"]
    e10 = _empty_mask(q10["prediction"])
    e11 = _empty_mask(q11["prediction"])
    out = {
        "implementation_artifact": {
            "v10_empty_rate": float(e10.mean()),
            "v11_empty_rate": float(e11.mean()),
            "empty_rate_delta": float(e11.mean() - e10.mean()),
            "v10_status_counts": q10["status"].value_counts().to_dict(),
            "v11_status_counts": q11["status"].value_counts().to_dict(),
        },
        "scientific_outcomes": {
            "v10_pass_mean_all_cells": float(q10[vcol10].mean()),
            "v11_pass_mean_all_cells": float(q11[vcol11].mean()),
            "v10_pass_mean_nonempty": float(q10.loc[~e10, vcol10].mean()) if (~e10).any() else None,
            "v11_pass_mean_nonempty": float(q11.loc[~e11, vcol11].mean()) if (~e11).any() else None,
            "v10_latency_median_ms": float(q10["latency_ms"].median()),
            "v11_latency_median_ms": float(q11["latency_ms"].median()),
        },
        "interpretation_guardrails": {
            "do_not_claim": "qwen3 failed compliance",
            "preferred_claim": (
                "Under v1.0 settings (thinking default ON, num_predict=1024), "
                "the recorded visible response was frequently empty because the "
                "shared generation budget was exhausted in the thinking channel."
            ),
        },
    }
    if "done_reason" in q11.columns:
        out["implementation_artifact"]["v11_done_reason_counts"] = (
            q11["done_reason"].fillna("null").value_counts().to_dict()
        )
    return out


def try_glmm(df: pd.DataFrame, out_csv: Path) -> dict:
    work = df.copy()
    mcol = _model_col(work)
    vcol = _value_col(work)
    pcol = _prompt_col(work)
    work = work.rename(
        columns={
            mcol: "model",
            pcol: "prompt_id",
            vcol: "metric_value",
            "task_id": "task_id",
            "run_index": "run_index",
            "temperature": "temperature",
        }
    )
    work["metric_name"] = "pass_at_1"
    try:
        result = run_pass_fail_glmm_analysis(work, metric="pass_at_1")
        coef = result.coefficients if hasattr(result, "coefficients") else None
        if coef is not None and len(coef):
            coef.to_csv(out_csv, index=False)
        return {"ok": True, "method": getattr(result, "method", None)}
    except Exception as exc:  # analysis must not crash the report pipeline
        return {"ok": False, "error": str(exc)}


def try_variance(df: pd.DataFrame, out_json: Path) -> dict:
    work = df.copy()
    mcol = _model_col(work)
    vcol = _value_col(work)
    work = work.rename(columns={mcol: "model", vcol: "metric_value"})
    work["metric_name"] = "pass_at_1"
    try:
        comps = estimate_g_variance_components(work, metric="pass_at_1")
        payload = comps if isinstance(comps, dict) else {"result": str(comps)}
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v10",
        type=Path,
        default=Path("artifacts/paper1/frozen/statistical_dataset.parquet"),
    )
    parser.add_argument("--v11", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../paper1/paper1_analysis_v11"),
    )
    parser.add_argument("--arm-label", default="v11")
    args = parser.parse_args()

    out = args.output_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    v10 = pd.read_parquet(args.v10)
    v11 = pd.read_parquet(args.v11)

    panel10 = summarize_panel(v10, "v1.0_frozen")
    panel11 = summarize_panel(v11, args.arm_label)
    pd.DataFrame(panel10["models"] + panel11["models"]).to_csv(
        out / "tables" / "model_summary_v10_vs_v11.csv", index=False
    )

    cmp = compare_qwen3(v10, v11)
    glmm = try_glmm(v11, out / "tables" / "glmm_coef_v11.csv")
    var = try_variance(v11, out / "tables" / "variance_components_v11.json")

    # Rank stability: model ordering by mean pass@1
    mcol = _model_col(v11)
    vcol = _value_col(v11)
    ranks = (
        v11.groupby(mcol)[vcol]
        .mean()
        .sort_values(ascending=False)
        .rename("pass_mean")
        .reset_index()
    )
    ranks["rank"] = np.arange(1, len(ranks) + 1)
    ranks.to_csv(out / "tables" / "rank_order_v11.csv", index=False)

    ranks10 = (
        v10.groupby(_model_col(v10))[_value_col(v10)]
        .mean()
        .sort_values(ascending=False)
        .rename("pass_mean")
        .reset_index()
    )
    ranks10["rank"] = np.arange(1, len(ranks10) + 1)
    ranks10.to_csv(out / "tables" / "rank_order_v10.csv", index=False)

    report = {
        "panels": {"v10": panel10, "v11": panel11},
        "qwen3_comparison": cmp,
        "glmm": glmm,
        "variance": var,
        "artifact_vs_science": {
            "implementation_artifact_metrics": [
                "empty_rate",
                "done_reason",
                "budget_exhausted",
                "thinking_length",
            ],
            "scientific_metrics": [
                "pass_at_1 under declared protocol arm",
                "latency under declared protocol arm",
                "GLMM / variance on the chosen analysis panel",
            ],
        },
    }
    (out / "comparison_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "qwen3": cmp}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
