#!/usr/bin/env python3
"""Build immutable-safe Paper1 v1.1 merged statistical dataset.

v1.0 frozen rows for non-qwen3 models are copied unchanged.
qwen3 rows are replaced from a completed v1.1 arm (A or B).

Never writes into artifacts/paper1/frozen/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

FROZEN = Path("artifacts/paper1/frozen/statistical_dataset.parquet")
EXPECTED_FROZEN_SHA256 = (
    "95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9"
)
QWEN3 = "qwen3_32b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_col(df: pd.DataFrame) -> str:
    if "model" in df.columns:
        return "model"
    if "model_id" in df.columns:
        return "model_id"
    raise KeyError("missing model column")


def _normalize_v11(df: pd.DataFrame, *, arm: str, experiment_id: str) -> pd.DataFrame:
    """Map v1.1 results schema toward frozen statistical_dataset schema."""
    out = df.copy()
    rename = {
        "model_id": "model",
        "prompt_variant_id": "prompt_id",
        "metric": "metric_name",
        "score": "metric_value",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "metric_name" not in out.columns:
        out["metric_name"] = "pass_at_1"
    if "metric_value" not in out.columns and "score" in df.columns:
        out["metric_value"] = df["score"]
    out["experiment_id"] = experiment_id
    meta = out.get("metadata")
    if meta is not None:
        # ensure dict-like metadata survives parquet
        out["metadata"] = [
            {
                **(m if isinstance(m, dict) else {}),
                "caliper_dataset_version": "v1.1",
                "qwen3_protocol_arm": arm,
                "replaced_from_v10": True,
            }
            for m in meta
        ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["a", "b"], required=True)
    parser.add_argument(
        "--arm-results",
        type=Path,
        required=True,
        help="Path to arm results.parquet or statistical_dataset.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper1/v1.1"),
    )
    args = parser.parse_args()

    if not FROZEN.exists():
        raise SystemExit(f"missing frozen v1.0 dataset: {FROZEN}")
    frozen_sha = _sha256(FROZEN)
    if frozen_sha != EXPECTED_FROZEN_SHA256:
        raise SystemExit(
            f"REFUSING TO BUILD: frozen SHA-256 mismatch\n"
            f"  expected {EXPECTED_FROZEN_SHA256}\n"
            f"  got      {frozen_sha}"
        )

    frozen = pd.read_parquet(FROZEN)
    mcol = _model_col(frozen)
    non_qwen = frozen[frozen[mcol] != QWEN3].copy()
    arm_df = pd.read_parquet(args.arm_results)
    arm_mcol = _model_col(arm_df)
    qwen = arm_df[arm_df[arm_mcol] == QWEN3].copy()
    if len(qwen) != 6560:
        raise SystemExit(f"expected 6560 qwen3 rows, got {len(qwen)}")

    experiment_id = f"paper1_confirmatory_humaneval_qwen3_v11_arm_{args.arm}"
    qwen_norm = _normalize_v11(qwen, arm=args.arm, experiment_id=experiment_id)

    # Align columns: keep union, prefer frozen column order where possible.
    for col in frozen.columns:
        if col not in qwen_norm.columns:
            qwen_norm[col] = None
    # retain extra v1.1 diagnostic columns
    extra = [c for c in qwen_norm.columns if c not in frozen.columns]
    ordered = list(frozen.columns) + extra
    qwen_norm = qwen_norm[ordered]
    non_qwen = non_qwen.reindex(columns=ordered)

    merged = pd.concat([non_qwen, qwen_norm], ignore_index=True)
    if len(merged) != 39360:
        raise SystemExit(f"expected 39360 merged rows, got {len(merged)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = args.output_dir / f"statistical_dataset_v11_arm_{args.arm}.parquet"
    merged.to_parquet(out_parquet, index=False)

    manifest = {
        "dataset_version": "v1.1",
        "arm": args.arm,
        "n_rows": int(len(merged)),
        "n_qwen3_replaced": 6560,
        "n_non_qwen3_from_v10": int(len(non_qwen)),
        "frozen_v10_path": str(FROZEN.resolve()),
        "frozen_v10_sha256": frozen_sha,
        "arm_results_path": str(args.arm_results.resolve()),
        "arm_results_sha256": _sha256(args.arm_results),
        "output_path": str(out_parquet.resolve()),
        "output_sha256": _sha256(out_parquet),
        "immutability": "v1.0 frozen bytes unchanged; merge writes only under v1.1 paths",
    }
    (args.output_dir / f"merge_manifest_arm_{args.arm}.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
