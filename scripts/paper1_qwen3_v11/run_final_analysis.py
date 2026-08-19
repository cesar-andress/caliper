#!/usr/bin/env python3
"""Single-command final analysis for Paper1 qwen3 v1.1 corrective arms.

Refuses to emit scientific outputs until both arms are complete,
unless --integrity-only or --allow-partial is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running as script or module
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paper1_qwen3_v11.constants import (  # type: ignore
        ANALYSIS_DIR,
        EXPECTED_FROZEN_SHA256,
        FROZEN_PARQUET,
    )
    from scripts.paper1_qwen3_v11.figures import generate_all_figures  # type: ignore
    from scripts.paper1_qwen3_v11.heterogeneity import analyze_heterogeneity  # type: ignore
    from scripts.paper1_qwen3_v11.loaders import (  # type: ignore
        arm_completion_status,
        load_arm_results,
        load_freeze_qwen3,
        sha256_file,
    )
    from scripts.paper1_qwen3_v11.postrun_integrity import run_and_write  # type: ignore
    from scripts.paper1_qwen3_v11.provenance import write_provenance  # type: ignore
    from scripts.paper1_qwen3_v11.scientific_validation import (  # type: ignore
        run_scientific_validation,
    )
    from scripts.paper1_qwen3_v11.statistical_comparison import (  # type: ignore
        run_statistical_comparisons,
    )
else:
    from .constants import ANALYSIS_DIR, EXPECTED_FROZEN_SHA256, FROZEN_PARQUET
    from .figures import generate_all_figures
    from .heterogeneity import analyze_heterogeneity
    from .loaders import arm_completion_status, load_arm_results, load_freeze_qwen3, sha256_file
    from .postrun_integrity import run_and_write
    from .provenance import write_provenance
    from .scientific_validation import run_scientific_validation
    from .statistical_comparison import run_statistical_comparisons


def _update_ready_checklist(status: dict) -> None:
    path = ANALYSIS_DIR / "READY_FOR_FINAL_ANALYSIS.md"
    # Preserve human checklist structure; rewrite status block at top.
    body = path.read_text(encoding="utf-8") if path.exists() else ""
    stamp = datetime.now(tz=UTC).isoformat()
    auto = [
        "<!-- AUTO-STATUS-START -->",
        f"Last pipeline status update: `{stamp}`",
        "",
        f"- Arm A complete: **{status['arm_a']['complete']}** ({status['arm_a']['n_results']}/6560)",
        f"- Arm B complete: **{status['arm_b']['complete']}** ({status['arm_b']['n_results']}/6560)",
        f"- Freeze immutable: **{status['freeze_ok']}**",
        f"- Final analysis executed: **{status['final_analysis_executed']}**",
        f"- Gate open: **{status['gate_open']}**",
        "<!-- AUTO-STATUS-END -->",
    ]
    import re

    if "<!-- AUTO-STATUS-START -->" in body:
        body = re.sub(
            r"<!-- AUTO-STATUS-START -->.*?<!-- AUTO-STATUS-END -->",
            "\n".join(auto),
            body,
            flags=re.S,
        )
    else:
        body = "\n".join(auto) + "\n\n" + body
    path.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Run integrity (+ provenance) for available arms; do not run science.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Dangerous: run science on incomplete arms (forbidden for publication).",
    )
    parser.add_argument(
        "--arms",
        default="a,b",
        help="Comma-separated arms for integrity (default a,b).",
    )
    args = parser.parse_args(argv)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    freeze_sha = sha256_file(FROZEN_PARQUET) if FROZEN_PARQUET.exists() else None
    freeze_ok = freeze_sha == EXPECTED_FROZEN_SHA256
    if not freeze_ok:
        print("FATAL: frozen v1.0 SHA-256 mismatch or missing; refusing to proceed.", file=sys.stderr)
        print(f"  expected {EXPECTED_FROZEN_SHA256}", file=sys.stderr)
        print(f"  got      {freeze_sha}", file=sys.stderr)
        return 2

    write_provenance()
    integrity = run_and_write(arms)

    st_a = arm_completion_status("a")
    st_b = arm_completion_status("b")
    gate_open = bool(st_a["complete"] and st_b["complete"] and freeze_ok)

    status = {
        "arm_a": st_a,
        "arm_b": st_b,
        "freeze_ok": freeze_ok,
        "gate_open": gate_open,
        "final_analysis_executed": False,
    }

    if args.integrity_only:
        _update_ready_checklist(status)
        print(json.dumps({"mode": "integrity_only", "gate_open": gate_open, "arms": {
            a: integrity.get(a, {}).get("summary") for a in arms
        }}, indent=2))
        return 0

    if not gate_open and not args.allow_partial:
        status["final_analysis_executed"] = False
        _update_ready_checklist(status)
        (ANALYSIS_DIR / "FINAL_ANALYSIS_BLOCKED.json").write_text(
            json.dumps(
                {
                    "blocked": True,
                    "reason": "Arms incomplete or freeze integrity failed",
                    "arm_a": st_a,
                    "arm_b": st_b,
                    "freeze_ok": freeze_ok,
                    "next_step": "Re-run this command after both arms finish.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "BLOCKED: final scientific analysis requires both arms complete.\n"
            f"  Arm A: {st_a['n_results']}/6560 complete={st_a['complete']}\n"
            f"  Arm B: {st_b['n_results']}/6560 complete={st_b['complete']}\n"
            "  Tip: python scripts/run_qwen3_v11_final_analysis.py --integrity-only",
            file=sys.stderr,
        )
        return 3

    if args.allow_partial:
        print(
            "WARNING: --allow-partial set. Outputs must NOT be used for publication.",
            file=sys.stderr,
        )

    freeze = load_freeze_qwen3()
    arm_a = load_arm_results("a")
    arm_b = load_arm_results("b")

    sci = run_scientific_validation(freeze, arm_a, arm_b)
    paired_a = sci["comparisons"]["A_vs_freeze"]["paired"]
    paired_b = sci["comparisons"]["B_vs_freeze"]["paired"]
    paired_ab = sci["comparisons"]["A_vs_B"]["paired"]

    hetero = analyze_heterogeneity(
        freeze, arm_a, arm_b, paired_a, paired_b, paired_ab
    )
    stats = run_statistical_comparisons(
        freeze, arm_a, arm_b, paired_a, paired_b, paired_ab
    )
    import pandas as pd

    panel_metrics = pd.read_csv(ANALYSIS_DIR / "scientific" / "tables" / "panel_metrics.csv")
    figs = generate_all_figures(
        panel_metrics,
        arm_a,
        arm_b,
        paired_a,
        paired_b,
        paired_ab,
        ANALYSIS_DIR / "heterogeneity" / "tables",
    )

    # Master index
    index = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "gate_open": gate_open,
        "allow_partial": args.allow_partial,
        "freeze_sha256": freeze_sha,
        "integrity_summaries": {a: integrity.get(a, {}).get("summary") for a in ("a", "b")},
        "outputs": {
            "integrity_report": str(ANALYSIS_DIR / "postrun_integrity_report.md"),
            "scientific_dir": str(ANALYSIS_DIR / "scientific"),
            "heterogeneity_dir": str(ANALYSIS_DIR / "heterogeneity"),
            "statistics_dir": str(ANALYSIS_DIR / "statistics"),
            "figures": figs,
            "provenance": str(ANALYSIS_DIR / "provenance_manifest.json"),
        },
        "publication_gate": {
            "numbers_authorized": gate_open and not args.allow_partial,
            "note": "Only use tables/figures for manuscript when numbers_authorized is true.",
        },
    }
    (ANALYSIS_DIR / "final_analysis_index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )

    # Human-readable executive summary (mechanism vs science separated)
    summary_md = ANALYSIS_DIR / "FINAL_ANALYSIS_SUMMARY.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Final analysis summary — qwen3 v1.1",
                "",
                f"Generated: {index['generated_at_utc']}",
                f"Publication numbers authorized: **{index['publication_gate']['numbers_authorized']}**",
                "",
                "## Artifact metrics (implementation)",
                "See `scientific/tables/comparison_summary.csv` columns `*_empty*` and `statistics/statistical_comparisons.json` → `mcnemar_empty`.",
                "",
                "## Scientific metrics (protocol-conditioned)",
                "See pass@1 / syntax columns in the same tables. Arms are **not** reinsertable into the v1.0 factorial.",
                "",
                "## Heterogeneity",
                "See `heterogeneity/heterogeneity_summary.json` and per-factor CSVs.",
                "",
                "## Figures",
                *[f"- `{name}`" for name in figs],
                "",
                "## Interpretive contract",
                "- Corrective arms establish mechanism and measurement-error magnitude.",
                "- They must not enter variance decomposition, GLMM, or n_t(δ).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    status["final_analysis_executed"] = True
    status["gate_open"] = gate_open
    _update_ready_checklist(status)
    print(json.dumps(index, indent=2))
    # silence unused
    _ = (hetero, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
