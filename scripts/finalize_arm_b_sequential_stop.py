#!/usr/bin/env python3
"""Finalize Arm B after sequential STOP (run once stop flag honored)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "outputs"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
)
OUT = ROOT.parent / "paper1" / "paper1_analysis_v11"
SEQ = OUT / "arm_b_sequential" / "latest.json"
PROTOCOL = OUT / "arm_b_sequential_protocol.md"


def main() -> int:
    stop = RUN / "STOP_REQUESTED"
    if not stop.exists():
        print("No STOP_REQUESTED; refuse to finalize stop audit.")
        return 2
    report = json.loads(SEQ.read_text(encoding="utf-8")) if SEQ.exists() else {}
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    proto_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() if PROTOCOL.exists() else None
    n = report.get("n_terminal")
    primary = report.get("primary", {})
    md = [
        "# Arm B stop audit",
        "",
        f"- timestamp_utc: `{datetime.now(tz=UTC).isoformat()}`",
        f"- git_commit: `{commit}`",
        f"- protocol_id: `arm_b_sequential_v1.0`",
        f"- protocol_sha256: `{proto_hash}`",
        f"- n_terminal: **{n}**",
        f"- theta_hat: **{primary.get('theta_hat')}**",
        f"- wilson_95: `{primary.get('wilson_95')}`",
        f"- region: `{primary.get('region')}`",
        f"- recommendation_at_stop: `{report.get('recommendation')}`",
        "",
        "## Stopping rationale",
    ]
    for j in report.get("justification", []):
        md.append(f"- {j}")
    md += [
        "",
        "## Achieved precision",
        f"- Wilson width: {None if not primary.get('wilson_95') else primary['wilson_95']['high']-primary['wilson_95']['low']}",
        "",
        "## Remaining uncertainty",
        "- Arm B is a sequential diagnostic sample, not a 6560-cell census.",
        "- Heterogeneity maps at full factorial resolution are incomplete.",
        "- Primary rate uncertainty is summarized by the Wilson interval above.",
        "",
        f"STOP_REQUESTED path: `{stop}`",
        "",
    ]
    (OUT / "arm_b_stop_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote", OUT / "arm_b_stop_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
