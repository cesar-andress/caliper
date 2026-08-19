#!/usr/bin/env python3
"""Live progress monitor for qwen3 v1.1 Arm B (causal validation).

Writes PRELIMINARY snapshots only. Never authorizes publication numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "outputs"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
)
LOG = ROOT / "logs" / "arm_b_v11.log"
OUT_DIR = ROOT.parent / "paper1" / "paper1_analysis_v11" / "arm_b_live"
EXPECTED = 6560


def snapshot() -> dict:
    now = datetime.now(tz=UTC).isoformat()
    jsonl = RUN / "results.jsonl"
    n = 0
    empty = 0
    budget = 0
    failed = 0
    scores: list[float] = []
    lat: list[float] = []
    eval_counts: list[float] = []
    think_lens: list[float] = []
    done: Counter[str] = Counter()
    status: Counter[str] = Counter()

    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n += 1
                pred = row.get("prediction") or ""
                if not pred:
                    empty += 1
                st = str(row.get("status") or "")
                status[st] += 1
                if st == "budget_exhausted":
                    budget += 1
                if st == "failed":
                    failed += 1
                scores.append(float(row.get("score") or 0.0))
                lat.append(float(row.get("latency_ms") or 0.0))
                if row.get("eval_count") is not None:
                    eval_counts.append(float(row["eval_count"]))
                if row.get("thinking_length") is not None:
                    think_lens.append(float(row["thinking_length"]))
                done[str(row.get("done_reason"))] += 1

    ep = {}
    if LOG.exists():
        finishes = [ln for ln in LOG.read_text(encoding="utf-8").splitlines() if "cell.finish" in ln]
        if finishes:
            try:
                ep = json.loads(finishes[-1]).get("execution_progress", {})
            except json.JSONDecodeError:
                ep = {}

    def _med(xs: list[float]) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        return xs[len(xs) // 2]

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    complete = n == EXPECTED and (RUN / "statistical_dataset.parquet").exists()
    return {
        "timestamp_utc": now,
        "label": "PRELIMINARY — not for publication",
        "n": n,
        "expected": EXPECTED,
        "pct": round(100.0 * n / EXPECTED, 2) if EXPECTED else None,
        "complete": complete,
        "failed": failed,
        "empty_n": empty,
        "empty_rate": (empty / n) if n else None,
        "budget_exhausted_n": budget,
        "budget_exhausted_rate": (budget / n) if n else None,
        "pass_at_1_mean_PRELIMINARY": _mean(scores),
        "latency_median_ms": _med(lat),
        "latency_mean_ms": _mean(lat),
        "eval_count_median": _med(eval_counts),
        "eval_count_mean": _mean(eval_counts),
        "thinking_length_median": _med(think_lens),
        "thinking_length_mean": _mean(think_lens),
        "done_reason": dict(done),
        "status": dict(status),
        "execution_progress": ep,
        "eta_hours": (ep.get("eta_seconds") / 3600.0) if ep.get("eta_seconds") is not None else None,
        "throughput_cells_per_second": ep.get("throughput_cells_per_second"),
        "artifacts": {
            "statistical_dataset": (RUN / "statistical_dataset.parquet").exists(),
            "manifest": (RUN / "manifest.json").exists(),
            "evaluations": (RUN / "evaluations.parquet").exists()
            or (RUN / "evaluations.jsonl").exists(),
        },
        "causal_note": (
            "Arm B tests whether think=true + num_predict=4096 removes the empty-response "
            "artifact. pass@1 here is incidental, not an optimization target."
        ),
    }


def write_snapshot(data: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest_json = OUT_DIR / "latest.json"
    latest_md = OUT_DIR / "latest.md"
    hist = OUT_DIR / "history.jsonl"
    latest_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with hist.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data) + "\n")

    lines = [
        "# Arm B live progress — PRELIMINARY",
        "",
        f"Updated: `{data['timestamp_utc']}`",
        "",
        f"**{data['n']}/{data['expected']} ({data['pct']}%)** — complete={data['complete']}",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| empty rate | {data['empty_rate']} |",
        f"| budget_exhausted | {data['budget_exhausted_n']} |",
        f"| failed | {data['failed']} |",
        f"| pass@1 (PRELIMINARY) | {data['pass_at_1_mean_PRELIMINARY']} |",
        f"| latency median ms | {data['latency_median_ms']} |",
        f"| eval_count median | {data['eval_count_median']} |",
        f"| thinking_length median | {data['thinking_length_median']} |",
        f"| ETA hours | {data['eta_hours']} |",
        "",
        f"done_reason: `{data['done_reason']}`",
        "",
        f"status: `{data['status']}`",
        "",
        "> Do not use these numbers in the manuscript.",
        "",
    ]
    latest_md.write_text("\n".join(lines), encoding="utf-8")
    return latest_md


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--stop-when-complete", action="store_true", default=True)
    args = parser.parse_args()

    while True:
        data = snapshot()
        path = write_snapshot(data)
        print(
            f"[{data['timestamp_utc']}] {data['n']}/{EXPECTED} "
            f"empty={data['empty_rate']} pass_PRELIM={data['pass_at_1_mean_PRELIMINARY']} "
            f"-> {path}",
            flush=True,
        )
        if args.once:
            return 0
        if args.stop_when_complete and data["complete"]:
            print("Arm B complete; monitor exiting.", flush=True)
            return 0
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
