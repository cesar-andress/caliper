#!/usr/bin/env python3
"""Run a single paper1 qwen3 v1.1 diagnostic arm (A or B).

After the arm finishes, automatically runs post-run integrity for that arm
and refreshes provenance. If both arms are then complete, runs the full
final scientific analysis with one additional command invocation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from caliper.config.loader import load_config
from caliper.runners.experiment import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]
ARM_CONFIGS = {
    "a": Path("configs/paper1/paper1_confirmatory_humaneval_qwen3_v11_arm_a.yaml"),
    "b": Path("configs/paper1/paper1_confirmatory_humaneval_qwen3_v11_arm_b.yaml"),
}


def _post_arm_hooks(arm: str) -> int:
    """Integrity for this arm + attempt final analysis gate."""
    py = sys.executable
    # Integrity + provenance for available arms (does not require both complete)
    cmd_integrity = [
        py,
        str(ROOT / "scripts" / "run_qwen3_v11_final_analysis.py"),
        "--integrity-only",
        "--arms",
        arm,
    ]
    print(f"post-run: {' '.join(cmd_integrity)}", flush=True)
    rc = subprocess.call(cmd_integrity, cwd=str(ROOT))
    if rc != 0:
        return rc
    # Attempt full final analysis; exits 3 if the other arm is still running
    cmd_final = [py, str(ROOT / "scripts" / "run_qwen3_v11_final_analysis.py")]
    print(f"post-run: attempting final analysis gate: {' '.join(cmd_final)}", flush=True)
    rc_final = subprocess.call(cmd_final, cwd=str(ROOT))
    # 0 = success; 3 = blocked waiting for other arm (acceptable); else error
    if rc_final in {0, 3}:
        return 0
    return rc_final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=sorted(ARM_CONFIGS))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--skip-post-hooks",
        action="store_true",
        help="Skip automatic integrity / final-analysis trigger.",
    )
    args = parser.parse_args()
    config_path = ARM_CONFIGS[args.arm]
    if not config_path.exists():
        print(f"missing config: {config_path}", file=sys.stderr)
        return 2
    config = load_config(config_path)
    runner = ExperimentRunner(
        config,
        config_path=config_path,
        resume_dir=args.resume,
    )
    print(
        f"starting arm={args.arm} experiment_id={config.experiment_id} "
        f"output={runner.output_dir}",
        flush=True,
    )
    manifest = runner.run()
    print(
        f"finished arm={args.arm} status={manifest.status} "
        f"completed={manifest.completed_cells} failed={manifest.failed_cells} "
        f"skipped={manifest.skipped_cells} total={manifest.total_cells}",
        flush=True,
    )
    if manifest.status not in {"completed", "partial"}:
        return 1
    if args.skip_post_hooks:
        return 0
    return _post_arm_hooks(args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
