#!/usr/bin/env python3
"""Watchdog: keep Arm B runner + sequential monitor alive without duplicate runners.

Does not alter scientific protocol. Safe to run indefinitely in the background.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "bin" / "python"
RUN_DIR = (
    ROOT
    / "outputs"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
)
LOGS = ROOT / "logs"


def pids(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    found: list[int] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if "/bin/bash" in cmd or "pgrep" in cmd or "extglob" in cmd:
            continue
        if "python" not in cmd:
            continue
        if "watchdog_arm_b_sequential.py" in cmd and "monitor" in pattern:
            continue
        found.append(pid)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=180)
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    while True:
        runners = pids("scripts/run_qwen3_v11_arm.py b")
        monitors = pids("scripts/monitor_arm_b_sequential.py")
        # Exclude watchdog itself from false positives if pattern overlaps — it doesn't.

        if len(runners) > 1:
            print(f"WARN duplicate runners: {runners}", flush=True)
        elif not runners:
            # Resume only; never restart from zero
            if (RUN_DIR / "STOP_REQUESTED").exists():
                print("STOP_REQUESTED present; not restarting runner.", flush=True)
            else:
                print("Runner missing; resuming Arm B...", flush=True)
                log = open(LOGS / "arm_b_v11.log", "a", encoding="utf-8")
                subprocess.Popen(
                    [
                        str(VENV_PY),
                        "-u",
                        "scripts/run_qwen3_v11_arm.py",
                        "b",
                        "--resume",
                        str(RUN_DIR),
                    ],
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

        if not monitors:
            print("Monitor missing; starting sequential monitor...", flush=True)
            log = open(LOGS / "arm_b_sequential_monitor.log", "a", encoding="utf-8")
            subprocess.Popen(
                [
                    str(VENV_PY),
                    "-u",
                    "scripts/monitor_arm_b_sequential.py",
                    "--interval-seconds",
                    "120",
                ],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        print(
            f"watchdog ok runners={pids('scripts/run_qwen3_v11_arm.py b')} "
            f"monitors={pids('scripts/monitor_arm_b_sequential.py')}",
            flush=True,
        )
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
