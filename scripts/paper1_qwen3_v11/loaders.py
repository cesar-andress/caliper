from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    ARM_OUTPUT_ROOTS,
    EXPECTED_CELLS,
    FROZEN_PARQUET,
    QWEN3_MODEL,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_arm_run_dir(arm: str) -> Path | None:
    root = ARM_OUTPUT_ROOTS[arm]
    if not root.exists():
        return None
    # Nested layout: outputs/<exp>/<exp>/
    nested = root / root.name
    if (nested / "results.jsonl").exists() or (nested / "results.parquet").exists():
        return nested
    if (root / "results.jsonl").exists() or (root / "results.parquet").exists():
        return root
    # Prefer deepest directory that looks like a run
    candidates = [p for p in root.rglob("results.jsonl")]
    if candidates:
        return candidates[0].parent
    return nested if nested.exists() else root


def _score_from_scores(scores: Any, key: str) -> float | None:
    if isinstance(scores, dict):
        val = scores.get(key)
        return float(val) if val is not None else None
    return None


def normalize_arm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "model_id": "model",
        "prompt_variant_id": "prompt_id",
        "metric": "metric_name",
        "score": "metric_value",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns and v not in out.columns})
    if "model" not in out.columns and "model_id" in df.columns:
        out["model"] = df["model_id"]
    if "prompt_id" not in out.columns and "prompt_variant_id" in df.columns:
        out["prompt_id"] = df["prompt_variant_id"]
    if "metric_value" not in out.columns and "score" in df.columns:
        out["metric_value"] = df["score"]
    if "metric_name" not in out.columns:
        out["metric_name"] = "pass_at_1"

    # Expand nested scores for syntax etc.
    if "scores" in out.columns:
        out["pass_at_1"] = out["scores"].map(lambda s: _score_from_scores(s, "pass_at_1"))
        out["syntax_check"] = out["scores"].map(lambda s: _score_from_scores(s, "syntax_check"))
        out["token_count"] = out["scores"].map(lambda s: _score_from_scores(s, "token_count"))
    else:
        out["pass_at_1"] = out.get("metric_value")
        out["syntax_check"] = None
        out["token_count"] = None

    out["prediction"] = out["prediction"].fillna("").astype(str)
    out["empty"] = out["prediction"].str.len() == 0
    out["budget_exhausted"] = out["status"].astype(str).eq("budget_exhausted")
    out["visible_chars"] = out["prediction"].str.len()
    if "eval_count" in out.columns:
        out["visible_tokens_proxy"] = out["eval_count"]
    else:
        out["visible_tokens_proxy"] = out["token_count"]
    if "thinking_length" not in out.columns:
        out["thinking_length"] = 0
    return out


def normalize_freeze_qwen3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    model_col = "model" if "model" in out.columns else "model_id"
    out = out[out[model_col] == QWEN3_MODEL].copy()
    if model_col != "model":
        out = out.rename(columns={model_col: "model"})
    if "prompt_id" not in out.columns and "prompt_variant_id" in out.columns:
        out = out.rename(columns={"prompt_variant_id": "prompt_id"})
    if "metric_value" not in out.columns and "score" in out.columns:
        out["metric_value"] = out["score"]
    if "scores" in out.columns:
        out["pass_at_1"] = out["scores"].map(lambda s: _score_from_scores(s, "pass_at_1"))
        out["syntax_check"] = out["scores"].map(lambda s: _score_from_scores(s, "syntax_check"))
        out["token_count"] = out["scores"].map(lambda s: _score_from_scores(s, "token_count"))
    else:
        out["pass_at_1"] = out["metric_value"]
        out["syntax_check"] = None
        out["token_count"] = None
    out["prediction"] = out["prediction"].fillna("").astype(str)
    out["empty"] = out["prediction"].str.len() == 0
    out["budget_exhausted"] = False
    out["visible_chars"] = out["prediction"].str.len()
    out["done_reason"] = None
    out["eval_count"] = None
    out["prompt_eval_count"] = None
    out["thinking_length"] = 0
    out["thinking_sha256"] = None
    out["panel"] = "v1.0_freeze"
    return out


def load_freeze_qwen3() -> pd.DataFrame:
    df = pd.read_parquet(FROZEN_PARQUET)
    return normalize_freeze_qwen3(df)


def load_arm_results(arm: str) -> pd.DataFrame:
    run_dir = resolve_arm_run_dir(arm)
    if run_dir is None:
        raise FileNotFoundError(f"Arm {arm} output directory not found")
    parquet = run_dir / "results.parquet"
    jsonl = run_dir / "results.jsonl"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif jsonl.exists():
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise FileNotFoundError(f"No results for arm {arm} under {run_dir}")
    out = normalize_arm_frame(df)
    out["panel"] = f"v1.1_arm_{arm}"
    return out


def arm_completion_status(arm: str) -> dict[str, Any]:
    run_dir = resolve_arm_run_dir(arm)
    if run_dir is None:
        return {
            "arm": arm,
            "exists": False,
            "complete": False,
            "n_results": 0,
            "expected": EXPECTED_CELLS,
            "run_dir": None,
            "has_statistical_dataset": False,
            "has_evaluations": False,
            "has_manifest": False,
        }
    jsonl = run_dir / "results.jsonl"
    n = 0
    if jsonl.exists():
        n = sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    complete = n == EXPECTED_CELLS and (run_dir / "statistical_dataset.parquet").exists()
    return {
        "arm": arm,
        "exists": True,
        "complete": complete,
        "n_results": n,
        "expected": EXPECTED_CELLS,
        "run_dir": str(run_dir),
        "has_statistical_dataset": (run_dir / "statistical_dataset.parquet").exists(),
        "has_evaluations": (run_dir / "evaluations.parquet").exists()
        or (run_dir / "evaluations.jsonl").exists(),
        "has_manifest": (run_dir / "manifest.json").exists(),
        "has_results_parquet": (run_dir / "results.parquet").exists(),
    }


def join_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure join keys exist; prefer cell_id when available."""
    out = df.copy()
    for col in ("task_id", "prompt_id", "temperature", "run_index", "model"):
        if col not in out.columns:
            raise KeyError(f"missing join column {col}")
    out["factor_key"] = (
        out["task_id"].astype(str)
        + "|"
        + out["prompt_id"].astype(str)
        + "|"
        + out["temperature"].map(lambda x: f"{float(x):.6f}")
        + "|"
        + out["run_index"].astype(str)
    )
    return out
