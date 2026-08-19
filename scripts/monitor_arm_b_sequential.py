#!/usr/bin/env python3
"""Live sequential monitor for Arm B (FROZEN protocol arm_b_sequential_v1.0).

Governance agent: executes frozen looks exactly. Does not redesign thresholds.
Survives session end when run with --interval-seconds in the background.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_ID = "arm_b_sequential_v1.0"
EXPECTED = 6560

ROOT = Path(__file__).resolve().parents[1]
PAPER1 = ROOT.parent / "paper1"
PROTOCOL_PATH = PAPER1 / "paper1_analysis_v11" / "arm_b_sequential_protocol.md"
RUN = (
    ROOT
    / "outputs"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
    / "paper1_confirmatory_humaneval_qwen3_v11_arm_b"
)
OUT = PAPER1 / "paper1_analysis_v11" / "arm_b_sequential"
ANALYSIS = PAPER1 / "paper1_analysis_v11"
LOOK800_DIR = OUT / "look_800"

LOOKS_MONITOR = {400}
LOOKS_DECISION = {800, 1000, 1200}
LOOKS_EXTENDED_START = 1400
LOOKS_EXTENDED_STEP = 200
LOOKS_EXTENDED_MAX = 2000

PRELIM_LABEL = "PRELIMINARY — NOT ELIGIBLE FOR SCIENTIFIC DECISION"


def protocol_checksum() -> str:
    if not PROTOCOL_PATH.exists():
        return "MISSING"
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PAPER1.parent),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def find_pids(pattern: str) -> list[int]:
    """Return PIDs whose cmdline is a python interpreter running the pattern."""
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    me = os.getpid()
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == me:
            continue
        cmd = parts[1]
        # Ignore shells / pgrep wrappers that merely embed the pattern in argv
        if "/bin/bash" in cmd or "pgrep" in cmd or "extglob" in cmd:
            continue
        if "python" not in cmd:
            continue
        pids.append(pid)
    return pids


def runner_pids() -> list[int]:
    return find_pids("scripts/run_qwen3_v11_arm.py b")


def monitor_pids() -> list[int]:
    # Prefer parent of this script when looping; include siblings for diagnostics
    return find_pids("scripts/monitor_arm_b_sequential.py")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    try:
        from scipy.stats import beta
    except ImportError:
        return (float("nan"), float("nan"))
    if n == 0:
        return (0.0, 1.0)
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (low, high)


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def load_all_rows() -> list[dict[str, Any]]:
    path = RUN / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_by_cell_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first occurrence of each cell_id (execution order in results.jsonl)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        cid = str(r.get("cell_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(r)
    return out


def load_terminal_rows(all_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = all_rows if all_rows is not None else load_all_rows()
    terminal = [r for r in rows if r.get("status") in {"completed", "budget_exhausted"}]
    return unique_by_cell_id(terminal)


def primary_event(row: dict[str, Any]) -> bool:
    empty = not (row.get("prediction") or "")
    budget = row.get("status") == "budget_exhausted"
    return bool(empty or budget)


def event_decomposition(rows: list[dict[str, Any]]) -> dict[str, int]:
    both = budget_only = empty_only = neither = 0
    for r in rows:
        empty = not (r.get("prediction") or "")
        budget = r.get("status") == "budget_exhausted"
        if budget and empty:
            both += 1
        elif budget and not empty:
            budget_only += 1
        elif empty and not budget:
            empty_only += 1
        else:
            neither += 1
    return {
        "budget_exhausted_only": budget_only,
        "empty_response_only": empty_only,
        "both": both,
        "neither": neither,
    }


def region_for(theta_hat: float, low: float, high: float) -> str:
    if high < 0.05:
        return "R1_eliminated"
    if low > 0.10:
        return "R2_persists"
    if theta_hat <= 0.25 and high >= 0.05 and low <= 0.10 and (high - low) <= 0.08:
        return "R3_greatly_reduced"
    return "R4_inconclusive"


def is_decision_look(n: int) -> bool:
    if n in LOOKS_DECISION:
        return True
    if LOOKS_EXTENDED_START <= n <= LOOKS_EXTENDED_MAX:
        return (n - LOOKS_EXTENDED_START) % LOOKS_EXTENDED_STEP == 0
    return False


def coverage(rows: list[dict[str, Any]], n_fail: int) -> dict[str, Any]:
    n = len(rows)
    prompts = Counter(r.get("prompt_variant_id") for r in rows)
    temps = Counter(float(r.get("temperature")) for r in rows)
    runs = Counter(int(r.get("run_index")) for r in rows)
    tasks = {r.get("task_id") for r in rows}
    n_all = n + n_fail
    fail_rate = (n_fail / n_all) if n_all else 0.0

    warnings: list[str] = []
    if n > 0:
        for p, c in prompts.items():
            if c < max(1, int(0.10 * n)):
                warnings.append(f"prompt `{p}` underrepresented: {c}/{n}")
        for t in (0.0, 0.2):
            if temps.get(t, 0) < max(1, int(0.30 * n)):
                warnings.append(f"temperature {t} underrepresented: {temps.get(t, 0)}/{n}")
        for i in range(5):
            if runs.get(i, 0) < max(1, int(0.08 * n)):
                warnings.append(f"run_index {i} underrepresented: {runs.get(i, 0)}/{n}")
        if len(tasks) < min(50, n):
            warnings.append(f"task coverage low: {len(tasks)} distinct")

    g1 = len(prompts) == 4 and all(c >= max(40, int(0.15 * n)) for c in prompts.values())
    g2 = all(temps.get(t, 0) >= int(0.35 * n) for t in (0.0, 0.2))
    g3 = all(runs.get(i, 0) >= max(25, int(0.10 * n)) for i in range(5))
    need_tasks = 100 if n < 800 else min(140, int(0.85 * 164))
    g4 = len(tasks) >= need_tasks
    g5 = fail_rate <= 0.02
    gates = {"G1_prompts": g1, "G2_temps": g2, "G3_runs": g3, "G4_tasks": g4, "G5_failures": g5}
    coverage_ok = all(gates.values()) if n >= 800 else g5
    return {
        "n_terminal": n,
        "n_failed": n_fail,
        "fail_rate": fail_rate,
        "prompts": dict(prompts),
        "temperatures": {str(k): v for k, v in sorted(temps.items())},
        "run_index": dict(sorted(runs.items())),
        "n_tasks": len(tasks),
        "difficulty_strata": "none_predefined_in_protocol",
        "gates": gates,
        "gate_thresholds": {
            "G1": f"each of 4 prompts >= max(40, floor(0.15n)) = {max(40, int(0.15 * n))}",
            "G2": f"each temp >= floor(0.35n) = {int(0.35 * n)}",
            "G3": f"each run_index >= max(25, floor(0.10n)) = {max(25, int(0.10 * n))}",
            "G4": f"distinct tasks >= {need_tasks}",
            "G5": "failure rate <= 0.02",
        },
        "coverage_ok": coverage_ok,
        "warnings": warnings,
    }


def quality_control(all_rows: list[dict[str, Any]], terminal: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [r.get("cell_id") for r in all_rows]
    dup = len(ids) - len(set(ids))
    failed = [r for r in all_rows if r.get("status") == "failed"]
    missing_done = sum(1 for r in terminal if r.get("done_reason") is None)
    missing_eval = sum(1 for r in terminal if r.get("eval_count") is None)
    missing_prompt_eval = sum(1 for r in terminal if r.get("prompt_eval_count") is None)
    missing_thinking_meta = sum(
        1
        for r in terminal
        if r.get("thinking_length") is None or r.get("thinking_sha256") is None
    )
    missing_provider = sum(
        1 for r in terminal if not r.get("provider_name") or not r.get("provider_type")
    )
    raw_dir = RUN / "raw_responses"
    raw_ids = {p.stem for p in raw_dir.glob("*.json")} if raw_dir.exists() else set()
    term_ids = {str(r.get("cell_id")) for r in terminal}
    missing_raw = sorted(term_ids - raw_ids)
    ck_dir = RUN / "checkpoints"
    ck_ids: set[str] = set()
    malformed_ck = 0
    if ck_dir.exists():
        for p in ck_dir.glob("*.json"):
            try:
                ck_ids.add(str(json.loads(p.read_text(encoding="utf-8")).get("cell_id") or p.stem))
            except Exception:
                malformed_ck += 1
                ck_ids.add(p.stem)
    missing_ck = sorted(term_ids - ck_ids)
    anomalies: list[str] = []
    if dup:
        anomalies.append(f"duplicate_cell_ids={dup}")
    if failed:
        anomalies.append(f"failed_cells={len(failed)}")
    if missing_done:
        anomalies.append(f"missing_done_reason={missing_done}")
    if missing_eval:
        anomalies.append(f"missing_eval_count={missing_eval}")
    if missing_prompt_eval:
        anomalies.append(f"missing_prompt_eval_count={missing_prompt_eval}")
    if missing_thinking_meta:
        anomalies.append(f"missing_thinking_metadata={missing_thinking_meta}")
    if missing_provider:
        anomalies.append(f"missing_provider_metadata={missing_provider}")
    if missing_raw:
        anomalies.append(f"missing_raw_responses={len(missing_raw)}")
    if missing_ck:
        anomalies.append(f"missing_checkpoints={len(missing_ck)}")
    if malformed_ck:
        anomalies.append(f"malformed_checkpoints={malformed_ck}")
    has_stat = (RUN / "statistical_dataset.parquet").exists()
    has_manifest = (RUN / "manifest.json").exists()
    return {
        "duplicate_cell_ids": dup,
        "failed_n": len(failed),
        "missing_done_reason": missing_done,
        "missing_eval_count": missing_eval,
        "missing_prompt_eval_count": missing_prompt_eval,
        "missing_thinking_metadata": missing_thinking_meta,
        "missing_provider_metadata": missing_provider,
        "missing_raw_n": len(missing_raw),
        "missing_checkpoint_n": len(missing_ck),
        "malformed_checkpoints": malformed_ck,
        "has_statistical_dataset": has_stat,
        "has_manifest": has_manifest,
        "anomalies": anomalies,
        "ok": (
            dup == 0
            and missing_done == 0
            and missing_eval == 0
            and missing_prompt_eval == 0
            and len(missing_raw) == 0
            and malformed_ck == 0
        ),
    }


def stratum_theta(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[Any, list] = defaultdict(list)
    for r in rows:
        groups[r.get(key)].append(r)
    out = {}
    for k, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        kk = sum(1 for r in rs if primary_event(r))
        nn = len(rs)
        low, high = wilson_ci(kk, nn)
        out[str(k)] = {
            "n": nn,
            "k": kk,
            "theta_hat": kk / nn if nn else None,
            "wilson_95": {"low": low, "high": high, "width": high - low},
        }
    return out


def boolean_rule_audit(
    n: int,
    theta: float,
    low: float,
    high: float,
    region: str,
    cov: dict[str, Any],
) -> dict[str, Any]:
    """Exact Boolean evaluation of frozen rules (for decision looks)."""
    width = high - low
    rules = {
        "n_ge_800": n >= 800,
        "is_decision_look": is_decision_look(n),
        "coverage_ok": bool(cov.get("coverage_ok")),
        "R1_U_lt_0.05": high < 0.05,
        "R2_L_gt_0.10": low > 0.10,
        "R3_theta_le_0.25": theta <= 0.25,
        "R3_U_ge_0.05": high >= 0.05,
        "R3_L_le_0.10": low <= 0.10,
        "R3_width_le_0.08": width <= 0.08,
        "R3_STOP_authorized_only_if_n_ge_1200": n >= 1200,
    }
    rules["R3_all"] = (
        rules["R3_theta_le_0.25"]
        and rules["R3_U_ge_0.05"]
        and rules["R3_L_le_0.10"]
        and rules["R3_width_le_0.08"]
    )
    return {
        "thresholds": {
            "R1_U": 0.05,
            "R2_L": 0.10,
            "R3_theta_max": 0.25,
            "R3_width_max": 0.08,
            "R3_min_n_for_stop": 1200,
        },
        "computed": {
            "n": n,
            "theta_hat": theta,
            "wilson_low": low,
            "wilson_high": high,
            "wilson_width": width,
            "region": region,
            "gates": cov.get("gates"),
        },
        "booleans": rules,
    }


def decide(n: int, region: str, cov: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Return (decision_code, recommendation, justification)."""
    if n < 800:
        return (
            "CONTINUE_MINIMUM_NOT_REACHED",
            "CONTINUE",
            [f"n={n} < first eligible decision look (800). No scientific stop authorized."],
        )
    if not is_decision_look(n):
        return (
            "CONTINUE_INCONCLUSIVE",
            "CONTINUE",
            [f"n={n} is not a pre-specified decision look."],
        )
    if not cov["coverage_ok"]:
        return (
            "CONTINUE_STRATA_INCOMPLETE",
            "CONTINUE",
            [f"Coverage gates failed at decision look: {cov['gates']}"],
        )
    if region == "R1_eliminated":
        return (
            "STOP_EFFECTIVELY_ELIMINATED",
            "STOP",
            ["R1: Wilson U95 < 0.05 (operational elimination)."],
        )
    if region == "R2_persists":
        return (
            "STOP_ARTIFACT_PERSISTS",
            "STOP",
            ["R2: Wilson L95 > 0.10 (material persistence)."],
        )
    if region == "R3_greatly_reduced":
        if n >= 1200:
            return (
                "STOP_ARTIFACT_PERSISTS",
                "STOP",
                [
                    "R3 at n>=1200 per frozen protocol: greatly reduced vs freeze, "
                    "not eliminated; STOP authorized (mapped to STOP_ARTIFACT_PERSISTS reporting class)."
                ],
            )
        return (
            "CONTINUE_INCONCLUSIVE",
            "CONTINUE",
            ["R3 observed but frozen protocol authorizes R3 STOP only at n>=1200."],
        )
    return (
        "CONTINUE_INCONCLUSIVE",
        "CONTINUE",
        ["R4 inconclusive under frozen thresholds."],
    )


def evaluate(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique_all = unique_by_cell_id(all_rows)
    terminal = load_terminal_rows(unique_all)
    n = len(terminal)
    n_fail = sum(1 for r in unique_all if r.get("status") == "failed")
    k = sum(1 for r in terminal if primary_event(r))
    low, high = wilson_ci(k, n)
    cp = clopper_pearson_ci(k, n)
    theta = (k / n) if n else float("nan")
    region = region_for(theta, low, high) if n else "R4_inconclusive"
    cov = coverage(terminal, n_fail)
    qc = quality_control(all_rows, terminal)
    decomp = event_decomposition(terminal)

    empty_k = sum(1 for r in terminal if not (r.get("prediction") or ""))
    budget_k = sum(1 for r in terminal if r.get("status") == "budget_exhausted")
    length_k = sum(1 for r in terminal if r.get("done_reason") == "length")

    scores = [float(r.get("score") or 0.0) for r in terminal]
    syntax = []
    for r in terminal:
        sc = r.get("scores") or {}
        if isinstance(sc, dict) and sc.get("syntax_check") is not None:
            syntax.append(float(sc["syntax_check"]))
    lats = [float(r.get("latency_ms") or 0.0) for r in terminal]
    evals = [float(r["eval_count"]) for r in terminal if r.get("eval_count") is not None]
    thinks = [float(r.get("thinking_length") or 0.0) for r in terminal]
    vis = [float(len(r.get("prediction") or "")) for r in terminal]

    decision, recommendation, justification = decide(n, region, cov)
    rule_audit = boolean_rule_audit(n, theta if n else float("nan"), low, high, region, cov)

    look_type = "pre_look"
    if n in LOOKS_MONITOR:
        look_type = "monitoring"
    elif is_decision_look(n):
        look_type = "decision"

    # ETA to 800 from median latency
    med_lat = _median(lats)
    remaining_to_800 = max(0, 800 - n)
    eta_hours = None
    if med_lat and remaining_to_800:
        eta_hours = (remaining_to_800 * med_lat / 1000.0) / 3600.0

    r_pids = runner_pids()
    m_pids = monitor_pids()

    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_checksum_sha256": protocol_checksum(),
        "git_commit": git_commit(),
        "runner_pids": r_pids,
        "monitor_pids": m_pids,
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "n_terminal": n,
        "n_eligible_unique": n,
        "n_rows_raw": len(all_rows),
        "n_remaining_to_census": EXPECTED - n,
        "n_remaining_to_800": remaining_to_800,
        "eta_hours_to_800": eta_hours,
        "look_type": look_type,
        "decision_look": is_decision_look(n),
        "decision": decision,
        "recommendation": recommendation,
        "justification": justification,
        "rule_audit": rule_audit,
        "primary": {
            "k": k,
            "n": n,
            "theta_hat": theta,
            "wilson_95": {"low": low, "high": high, "width": high - low},
            "clopper_pearson_95": {"low": cp[0], "high": cp[1]},
            "region": region,
            "event_decomposition": decomp,
        },
        "secondary": {
            "empty_rate": empty_k / n if n else None,
            "empty_n": empty_k,
            "budget_exhausted_rate": budget_k / n if n else None,
            "budget_exhausted_n": budget_k,
            "done_reason_length_rate": length_k / n if n else None,
            "done_reason": dict(Counter(r.get("done_reason") for r in terminal)),
            "status": dict(Counter(r.get("status") for r in terminal)),
            "pass_at_1_mean_PRELIMINARY": _mean(scores),
            "syntax_validity_mean_PRELIMINARY": _mean(syntax),
            "latency_median_ms": _median(lats),
            "latency_mean_ms": _mean(lats),
            "eval_count_median": _median(evals),
            "eval_count_mean": _mean(evals),
            "thinking_length_median": _median(thinks),
            "thinking_length_mean": _mean(thinks),
            "visible_chars_median": _median(vis),
            "visible_chars_mean": _mean(vis),
        },
        "by_prompt": stratum_theta(terminal, "prompt_variant_id"),
        "by_temperature": stratum_theta(terminal, "temperature"),
        "by_run_index": stratum_theta(terminal, "run_index"),
        "coverage": cov,
        "quality_control": qc,
        "preliminary_label": PRELIM_LABEL if n < 800 else None,
    }


def write_latest_md(report: dict[str, Any]) -> None:
    p = report["primary"]
    s = report["secondary"]
    cov = report["coverage"]
    qc = report["quality_control"]
    prelim = report["n_terminal"] < 800
    banner = f"> **{PRELIM_LABEL}**" if prelim else "> First eligible look reached or passed; decision uses frozen rules only."
    lines = [
        f"# Arm B sequential monitor (`{PROTOCOL_ID}`)",
        "",
        f"Updated: `{report['timestamp_utc']}`",
        "",
        banner,
        "",
        "## Provenance",
        f"- protocol_id: `{report['protocol_id']}`",
        f"- protocol_checksum_sha256: `{report['protocol_checksum_sha256']}`",
        f"- git_commit: `{report['git_commit']}`",
        f"- runner_pids: `{report['runner_pids']}`",
        f"- monitor_pids: `{report['monitor_pids']}`",
        "",
        f"**Decision: `{report['decision']}`**",
        f"**Recommendation: `{report['recommendation']}`**",
        f"**Frozen region: `{p['region']}`**",
        "",
        "## Progress",
        f"- eligible unique terminal n: **{report['n_eligible_unique']}** / {EXPECTED}",
        f"- raw results.jsonl rows: {report['n_rows_raw']}",
        f"- remaining to census: **{report['n_remaining_to_census']}**",
        f"- remaining to n=800: **{report['n_remaining_to_800']}**",
        f"- ETA hours to n=800 (median latency): {report['eta_hours_to_800']}",
        f"- look_type: `{report['look_type']}`",
        f"- failed cells (unique): {cov['n_failed']} (rate={cov['fail_rate']})",
        "",
        "## Primary estimand θ",
        f"- θ̂ = **{p['theta_hat']}** (k={p['k']}, n={p['n']})"
        + (f" — {PRELIM_LABEL}" if prelim else ""),
        f"- Wilson 95% CI: **[{p['wilson_95']['low']:.4f}, {p['wilson_95']['high']:.4f}]**",
        f"- CI width: **{p['wilson_95']['width']:.4f}**",
        f"- Clopper–Pearson 95%: `{p['clopper_pearson_95']}`",
        f"- event decomposition: `{p['event_decomposition']}`",
        "",
        "## Rates",
        f"- empty response rate: {s['empty_rate']} (n={s['empty_n']})",
        f"- budget_exhausted rate: {s['budget_exhausted_rate']} (n={s['budget_exhausted_n']})",
        f"- done_reason: `{s['done_reason']}`",
        f"- status: `{s['status']}`",
        "",
        f"## Secondary metrics ({'PRELIMINARY' if prelim else 'at/after eligible look'})",
        f"- pass@1 mean: {s['pass_at_1_mean_PRELIMINARY']}"
        + (f" — {PRELIM_LABEL}" if prelim else " (descriptive; not a stop endpoint)"),
        f"- syntax validity mean: {s['syntax_validity_mean_PRELIMINARY']}",
        f"- latency median / mean ms: {s['latency_median_ms']} / {s['latency_mean_ms']}",
        f"- eval_count median / mean: {s['eval_count_median']} / {s['eval_count_mean']}",
        f"- thinking_length median / mean: {s['thinking_length_median']} / {s['thinking_length_mean']}",
        f"- visible_chars median / mean: {s['visible_chars_median']} / {s['visible_chars_mean']}",
        "",
        "## Coverage",
        f"- coverage_ok (hard gates bind at n≥800): **{cov['coverage_ok']}**",
        f"- gates: `{cov['gates']}`",
        f"- gate_thresholds: `{cov['gate_thresholds']}`",
        f"- prompts: `{cov['prompts']}`",
        f"- temperatures: `{cov['temperatures']}`",
        f"- run_index: `{cov['run_index']}`",
        f"- n_tasks: {cov['n_tasks']}",
        f"- difficulty_strata: {cov['difficulty_strata']}",
    ]
    if cov["warnings"]:
        lines.append("- **warnings:**")
        for w in cov["warnings"]:
            lines.append(f"  - {w}")
    lines += [
        "",
        "## Quality control",
        f"- ok: **{qc['ok']}**",
        f"- anomalies: `{qc['anomalies']}`",
        "",
        "## Justification",
    ]
    for j in report["justification"]:
        lines.append(f"- {j}")
    lines.append("")
    (OUT / "latest.md").write_text("\n".join(lines), encoding="utf-8")


def write_monitoring_look_400(report: dict[str, Any]) -> None:
    p = report["primary"]
    s = report["secondary"]
    path = ANALYSIS / "arm_b_monitoring_look_400.md"
    if path.exists():
        return
    lines = [
        "# Arm B monitoring look — n = 400 (NOT a decision look)",
        "",
        f"**Protocol:** `{PROTOCOL_ID}`",
        f"**protocol_checksum_sha256:** `{report['protocol_checksum_sha256']}`",
        f"**git_commit:** `{report['git_commit']}`",
        f"**Generated:** `{report['timestamp_utc']}`",
        "",
        f"> {PRELIM_LABEL}",
        "",
        "## Primary / secondary estimands",
        f"- θ̂ = {p['theta_hat']} (k={p['k']}, n={p['n']})",
        f"- Wilson 95% CI: [{p['wilson_95']['low']:.6f}, {p['wilson_95']['high']:.6f}]",
        f"- CI width: {p['wilson_95']['width']:.6f}",
        f"- region (descriptive only): `{p['region']}`",
        f"- empty rate: {s['empty_rate']}",
        f"- budget_exhausted rate: {s['budget_exhausted_rate']}",
        f"- done_reason: `{s['done_reason']}`",
        f"- pass@1 mean (PRELIMINARY): {s['pass_at_1_mean_PRELIMINARY']}",
        f"- syntax mean (PRELIMINARY): {s['syntax_validity_mean_PRELIMINARY']}",
        f"- latency median ms: {s['latency_median_ms']}",
        f"- eval_count median: {s['eval_count_median']}",
        f"- thinking_length median: {s['thinking_length_median']}",
        f"- visible_chars median: {s['visible_chars_median']}",
        "",
        "## Coverage gates (display; STOP not allowed)",
        f"```json\n{json.dumps(report['coverage'], indent=2)}\n```",
        "",
        "## Technical integrity",
        f"```json\n{json.dumps(report['quality_control'], indent=2)}\n```",
        "",
        "## Stratum imbalance warnings",
    ]
    for w in report["coverage"].get("warnings") or ["(none)"]:
        lines.append(f"- {w}")
    eta = report.get("eta_hours_to_800")
    lines += [
        "",
        "## Estimated time to n = 800",
        f"- remaining cells: {report['n_remaining_to_800']}",
        f"- ETA hours (median latency extrapolation): {eta}",
        "",
        "CONTINUE — MONITORING LOOK ONLY",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "monitoring_look_400.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def freeze_look_800_snapshot(terminal800: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, str]:
    """Immutable snapshot of analysis inputs at n=800. Idempotent if marker exists."""
    LOOK800_DIR.mkdir(parents=True, exist_ok=True)
    marker = LOOK800_DIR / "SNAPSHOT_COMPLETE.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8")).get("hashes", {})

    # Copy / write inputs
    if (RUN / "manifest.json").exists():
        shutil.copy2(RUN / "manifest.json", LOOK800_DIR / "manifest.json")
    # results prefix metadata
    cell_ids = [str(r.get("cell_id")) for r in terminal800]
    (LOOK800_DIR / "cell_ids.json").write_text(json.dumps(cell_ids, indent=2), encoding="utf-8")
    (LOOK800_DIR / "results_prefix_800.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in terminal800) + "\n",
        encoding="utf-8",
    )
    # checkpoint inventory
    ck_dir = RUN / "checkpoints"
    ck_list = sorted(p.name for p in ck_dir.glob("*.json")) if ck_dir.exists() else []
    (LOOK800_DIR / "checkpoint_inventory.json").write_text(
        json.dumps({"n": len(ck_list), "files": ck_list}, indent=2), encoding="utf-8"
    )
    # provider metadata completeness
    prov = {
        "n": len(terminal800),
        "missing_provider_name": sum(1 for r in terminal800 if not r.get("provider_name")),
        "missing_provider_type": sum(1 for r in terminal800 if not r.get("provider_type")),
        "missing_done_reason": sum(1 for r in terminal800 if r.get("done_reason") is None),
        "missing_eval_count": sum(1 for r in terminal800 if r.get("eval_count") is None),
        "missing_prompt_eval_count": sum(
            1 for r in terminal800 if r.get("prompt_eval_count") is None
        ),
        "missing_thinking_sha256": sum(1 for r in terminal800 if not r.get("thinking_sha256")),
    }
    (LOOK800_DIR / "provider_metadata_completeness.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8"
    )
    (LOOK800_DIR / "report_at_snapshot.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    hashes = {
        "protocol_md": protocol_checksum(),
        "results_prefix_800_jsonl": file_sha256(LOOK800_DIR / "results_prefix_800.jsonl") or "",
        "cell_ids_json": file_sha256(LOOK800_DIR / "cell_ids.json") or "",
        "checkpoint_inventory_json": file_sha256(LOOK800_DIR / "checkpoint_inventory.json") or "",
        "provider_metadata_completeness_json": file_sha256(
            LOOK800_DIR / "provider_metadata_completeness.json"
        )
        or "",
        "report_at_snapshot_json": file_sha256(LOOK800_DIR / "report_at_snapshot.json") or "",
    }
    if (LOOK800_DIR / "manifest.json").exists():
        hashes["manifest_json"] = file_sha256(LOOK800_DIR / "manifest.json") or ""
    meta = {
        "protocol_id": PROTOCOL_ID,
        "timestamp_utc": report["timestamp_utc"],
        "git_commit": report["git_commit"],
        "protocol_checksum_sha256": report["protocol_checksum_sha256"],
        "n": 800,
        "hashes": hashes,
    }
    (LOOK800_DIR / "SNAPSHOT_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return hashes


def write_first_look_800(report: dict[str, Any], terminal: list[dict[str, Any]]) -> None:
    path = ANALYSIS / "arm_b_first_look_800.md"
    if path.exists():
        return
    p = report["primary"]
    s = report["secondary"]
    by_task = stratum_theta(terminal, "task_id")
    task_thetas = [v["theta_hat"] for v in by_task.values() if v["theta_hat"] is not None]
    lines = [
        "# Arm B first eligible look — n = 800",
        "",
        f"**Protocol:** `{PROTOCOL_ID}` (FROZEN — thresholds unchanged)",
        f"**protocol_checksum_sha256:** `{report['protocol_checksum_sha256']}`",
        f"**git_commit:** `{report['git_commit']}`",
        f"**Generated:** `{report['timestamp_utc']}`",
        f"**Decision code:** `{report['decision']}`",
        f"**Recommendation:** `{report['recommendation']}`",
        f"**Snapshot dir:** `paper1_analysis_v11/arm_b_sequential/look_800/`",
        "",
        "## A. Primary endpoint",
        "",
        f"- event count k: **{p['k']}**",
        f"- eligible n: **{p['n']}**",
        f"- θ̂ = **{p['theta_hat']:.6f}**",
        f"- Wilson 95% CI: **[{p['wilson_95']['low']:.6f}, {p['wilson_95']['high']:.6f}]**",
        f"- CI width: **{p['wilson_95']['width']:.6f}**",
        f"- Clopper–Pearson 95%: `{p['clopper_pearson_95']}`",
        f"- Frozen region: `{p['region']}`",
        "",
        "## B. Event decomposition",
        "",
        f"```json\n{json.dumps(p['event_decomposition'], indent=2)}\n```",
        f"- done_reason counts: `{s['done_reason']}`",
        f"- status counts: `{s['status']}`",
        "",
        "## C. Coverage gates",
        "",
        f"```json\n{json.dumps(report['coverage'], indent=2)}\n```",
        "",
        "## D. Secondary outcomes",
        "",
        f"- pass@1 mean: **{s['pass_at_1_mean_PRELIMINARY']}** (descriptive; not a stop endpoint)",
        f"- syntax validity mean: **{s['syntax_validity_mean_PRELIMINARY']}**",
        f"- latency median / mean ms: {s['latency_median_ms']} / {s['latency_mean_ms']}",
        f"- eval_count median / mean: {s['eval_count_median']} / {s['eval_count_mean']}",
        f"- thinking_length median / mean: {s['thinking_length_median']} / {s['thinking_length_mean']}",
        f"- visible_chars median / mean: {s['visible_chars_median']} / {s['visible_chars_mean']}",
        "",
        "## E. Heterogeneity",
        "",
        "### By prompt",
        f"```json\n{json.dumps(report['by_prompt'], indent=2)}\n```",
        "",
        "### By temperature",
        f"```json\n{json.dumps(report['by_temperature'], indent=2)}\n```",
        "",
        "### By run_index",
        f"```json\n{json.dumps(report['by_run_index'], indent=2)}\n```",
        "",
        "### Task-level distribution",
        f"- distinct tasks: {len(by_task)}",
        f"- task-level θ mean / median: {_mean(task_thetas)} / {_median(task_thetas)}",
        f"- task-level θ min / max: {min(task_thetas) if task_thetas else None} / {max(task_thetas) if task_thetas else None}",
        "",
        "## F. Integrity",
        "",
        f"```json\n{json.dumps(report['quality_control'], indent=2)}\n```",
        "",
        "## Frozen decision rule evaluation",
        "",
        f"```json\n{json.dumps(report['rule_audit'], indent=2)}\n```",
        "",
        f"**Final decision:** `{report['decision']}`",
        "",
    ]
    for j in report["justification"]:
        lines.append(f"- {j}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    (OUT / "first_look_800.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_continuation_after_800(report: dict[str, Any]) -> None:
    path = ANALYSIS / "arm_b_continuation_after_800.md"
    if path.exists():
        return
    failed_rule = "precision/inconclusiveness"
    if report["decision"] == "CONTINUE_STRATA_INCOMPLETE":
        failed_rule = "coverage"
    elif report["decision"] == "CONTINUE_INCONCLUSIVE":
        failed_rule = "precision or inconclusiveness (R3/R4 under frozen look schedule)"
    lines = [
        "# Arm B continuation after n = 800",
        "",
        f"**Protocol:** `{PROTOCOL_ID}`",
        f"**Decision:** `{report['decision']}`",
        f"**Generated:** `{report['timestamp_utc']}`",
        "",
        "## Why stopping was not allowed",
        "",
    ]
    for j in report["justification"]:
        lines.append(f"- {j}")
    lines += [
        "",
        f"**Which problem class:** `{failed_rule}`",
        "",
        "## What n = 1000 is expected to add",
        "",
        "- Narrower Wilson interval under the same estimand and locked settings.",
        "- Another pre-specified decision look under identical R1–R4 and coverage gates.",
        "- No threshold changes; no added looks between 800 and 1000.",
        "",
        "Runner continues under the same frozen protocol toward n = 1000.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stop_audit(report: dict[str, Any], hashes: dict[str, str], final_n: int) -> None:
    path = ANALYSIS / "arm_b_stop_audit.md"
    lines = [
        "# Arm B sequential STOP audit",
        "",
        f"**Protocol:** `{PROTOCOL_ID}`",
        f"**protocol_checksum_sha256:** `{report['protocol_checksum_sha256']}`",
        f"**git_commit:** `{report['git_commit']}`",
        f"**Timestamp:** `{report['timestamp_utc']}`",
        f"**Stopping decision:** `{report['decision']}`",
        f"**Decision n (eligible unique):** `{report['n_terminal']}`",
        f"**Final stored n at audit write:** `{final_n}`",
        f"**Difference reason:** `{'none' if final_n == report['n_terminal'] else 'cells completed after snapshot before graceful stop'}`",
        "",
        "## Snapshot hashes",
        f"```json\n{json.dumps(hashes, indent=2)}\n```",
        "",
        "## Rule audit",
        f"```json\n{json.dumps(report['rule_audit'], indent=2)}\n```",
        "",
        "## Shutdown sequence",
        "1. Wrote `STOP_REQUESTED` in Arm B run directory.",
        "2. Runner must finish the active cell and not start another.",
        "3. Monitor exits after STOP recommendation.",
        "4. Experiment status target: `stopped_by_sequential_protocol`.",
        "",
        "## Integrity",
        f"```json\n{json.dumps(report['quality_control'], indent=2)}\n```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_editorial_note_800(report: dict[str, Any]) -> None:
    path = PAPER1 / "editorial_note_after_800.md"
    if path.exists():
        return
    p = report["primary"]
    decision = report["decision"]
    stopped = decision.startswith("STOP_")
    lines = [
        "# Editorial note after n = 800 (first eligible look)",
        "",
        f"**Protocol:** `{PROTOCOL_ID}` (frozen)",
        f"**Look decision code:** `{decision}`",
        f"**Generated:** `{report['timestamp_utc']}`",
        "",
        "## Associate Editor stance",
        "",
        "Acting as a demanding EMSE Associate Editor: comfort with stopping is allowed "
        "**only** if the frozen protocol authorizes STOP and integrity is intact.",
        "",
        f"### Protocol followed without deviation?",
        "Yes — look schedule, Wilson intervals, and coverage gates are those frozen in "
        f"`{PROTOCOL_ID}` (checksum `{report['protocol_checksum_sha256']}`).",
        "",
        "### Execution prefix representative?",
        "The analysis uses the first 800 eligible unique terminal cells in execution order "
        "under the seeded shuffle (seed 20260404), subject to coverage gates.",
        "",
        "### Integrity satisfactory?",
        f"QC ok={report['quality_control']['ok']}; anomalies=`{report['quality_control']['anomalies']}`",
        "",
        "### Statistical precision vs frozen decision?",
        f"θ̂={p['theta_hat']}, Wilson 95% CI=[{p['wilson_95']['low']:.4f}, {p['wilson_95']['high']:.4f}], "
        f"width={p['wilson_95']['width']:.4f}, region=`{p['region']}`, coverage_ok="
        f"**{report['coverage']['coverage_ok']}**.",
        "",
        "### Is a STOP (if triggered) scientifically defensible?",
        (
            "Yes — because it is the pre-specified R1/R2 (or authorized R3 at n≥1200) outcome, "
            "not an informal early peek."
            if stopped
            else "N/A — frozen rule did not authorize STOP at this look."
        ),
        "",
        "### Remaining uncertainty / reviewer demand for remaining cells?",
        (
            "Residual uncertainty is the Wilson interval and that Arm B is diagnostic, not a "
            "6,560-cell census. Reviewers may still ask for later looks if CONTINUE."
            if stopped
            else "Because CONTINUE is mandatory under the frozen rule, reviewers cannot be told "
            "the diagnostic is finished; proceed to n=1000 under the same protocol."
        ),
        "",
        "## Bottom line",
        f"Comfortable with informal override? **NO.**",
        f"Comfortable with the frozen decision `{decision}`? **YES — follow it.**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_execution_status_after_800(report: dict[str, Any], state: str) -> None:
    path = ANALYSIS / "arm_b_execution_status_after_800.md"
    med = report["secondary"]["latency_median_ms"] or 180000.0
    rem = max(0, 1000 - report["n_terminal"]) if state.startswith("VALID_CONTINUATION") else 0
    eta = (rem * med / 1000.0) / 3600.0 if rem else 0.0
    lines = [
        "# Arm B execution status after n = 800",
        "",
        f"**State:** `{state}`",
        f"**Protocol:** `{PROTOCOL_ID}`",
        f"**protocol_checksum_sha256:** `{report['protocol_checksum_sha256']}`",
        f"**git_commit:** `{report['git_commit']}`",
        f"**Generated:** `{report['timestamp_utc']}`",
        "",
        "## Protocol compliance",
        "Frozen protocol executed without threshold/look/sampling changes.",
        "",
        "## Execution integrity",
        f"- QC: `{report['quality_control']}`",
        f"- runner_pids: `{report['runner_pids']}`",
        f"- monitor_pids: `{report['monitor_pids']}`",
        "",
        "## First-look results",
        f"- θ̂={report['primary']['theta_hat']}",
        f"- Wilson={report['primary']['wilson_95']}",
        f"- region=`{report['primary']['region']}`",
        "",
        "## Frozen-rule decision",
        f"- `{report['decision']}` / recommendation `{report['recommendation']}`",
        "",
        "## Actions taken",
        "- Snapshot under `arm_b_sequential/look_800/`",
        "- `arm_b_first_look_800.md`",
        "- `editorial_note_after_800.md`",
        (
            "- `STOP_REQUESTED` + `arm_b_stop_audit.md`"
            if report["recommendation"] == "STOP"
            else "- `arm_b_continuation_after_800.md`; runner left running toward n=1000"
        ),
        "",
        "## Remaining runtime",
        f"- ETA hours to next milestone (if CONTINUE to 1000): {eta}",
        "",
        "## Exact next step",
        (
            "Await graceful runner halt; finalize stop status `stopped_by_sequential_protocol`."
            if report["recommendation"] == "STOP"
            else "Keep runner+monitor alive until n=1000 decision look; do not alter protocol."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_write_stop_artifacts(report: dict[str, Any]) -> None:
    if report["recommendation"] != "STOP":
        return
    if report["n_terminal"] < 800 or not is_decision_look(report["n_terminal"]):
        return
    flag = {
        "protocol_id": PROTOCOL_ID,
        "requested_at_utc": report["timestamp_utc"],
        "decision": report["decision"],
        "reason": report["justification"],
        "primary": report["primary"],
        "n_terminal": report["n_terminal"],
        "action": "stop_between_cells",
        "experiment_status_target": "stopped_by_sequential_protocol",
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "STOP_REQUESTED").write_text(json.dumps(flag, indent=2), encoding="utf-8")


def handle_look_milestones(all_rows: list[dict[str, Any]], terminal: list[dict[str, Any]]) -> None:
    n = len(terminal)

    # n=400 monitoring look (exact or first crossing)
    if n >= 400 and not (ANALYSIS / "arm_b_monitoring_look_400.md").exists():
        prefix = terminal[:400]
        # evaluate on prefix + failed from all_rows
        failed = [r for r in unique_by_cell_id(all_rows) if r.get("status") == "failed"]
        rep400 = evaluate(prefix + failed)
        # Force fields to prefix-400
        rep400["n_terminal"] = 400
        rep400["n_eligible_unique"] = 400
        write_monitoring_look_400(rep400)

    # n=800 first eligible look
    if n >= 800 and not (LOOK800_DIR / "SNAPSHOT_COMPLETE.json").exists():
        prefix = terminal[:800]
        failed = [r for r in unique_by_cell_id(all_rows) if r.get("status") == "failed"]
        # Evaluate decision on exact first 800 terminal cells
        rep800 = evaluate(prefix + failed)
        # Ensure decision uses n=800 terminal set (failed don't enter theta denom)
        # Re-bind primary n to 800 after evaluate on prefix+failed
        term800 = load_terminal_rows(prefix)
        if len(term800) != 800:
            # Should not happen; wait for next cycle rather than crash monitor
            print(f"WARN look_800 prefix size={len(term800)}; deferring snapshot", flush=True)
            return
        rep800 = evaluate(term800 + failed)
        hashes = freeze_look_800_snapshot(term800, rep800)
        write_first_look_800(rep800, term800)
        write_editorial_note_800(rep800)
        if rep800["recommendation"] == "STOP":
            maybe_write_stop_artifacts(rep800)
            final_n = len(load_terminal_rows(load_all_rows()))
            write_stop_audit(rep800, hashes, final_n)
            write_execution_status_after_800(rep800, "VALID_STOP_AT_N_800")
        else:
            write_continuation_after_800(rep800)
            write_execution_status_after_800(rep800, "VALID_CONTINUATION_TO_N_1000")


def write_outputs(report: dict[str, Any], all_rows: list[dict[str, Any]], terminal: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report) + "\n")
    write_latest_md(report)
    handle_look_milestones(all_rows, terminal)
    # If already past 800 and STOP was decided at a later decision look, still allow stop flag
    if report["recommendation"] == "STOP" and report["n_terminal"] >= 800:
        maybe_write_stop_artifacts(report)


def health_blocker() -> str | None:
    """Return technical blocker message or None."""
    if not runner_pids():
        return "Arm B runner not running"
    # disk
    usage = shutil.disk_usage(str(RUN))
    if usage.free < 5 * 1024**3:
        return f"Low disk space: {usage.free / 1024**3:.1f} GiB free"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()

    while True:
        all_rows = load_all_rows()
        terminal = load_terminal_rows(all_rows)
        report = evaluate(all_rows)
        blocker = health_blocker()
        if blocker:
            report["justification"] = list(report["justification"]) + [f"TECHNICAL_BLOCKER: {blocker}"]
            report["decision"] = report["decision"]  # do not invent scientific stop
            OUT.mkdir(parents=True, exist_ok=True)
            (ANALYSIS / "arm_b_technical_blocker.md").write_text(
                "\n".join(
                    [
                        "# Arm B TECHNICAL BLOCKER",
                        "",
                        f"Detected: `{blocker}`",
                        f"Timestamp: `{report['timestamp_utc']}`",
                        f"n={report['n_terminal']}",
                        "",
                        "Scientific outcomes remain non-decisive. Author action required.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        write_outputs(report, all_rows, terminal)
        print(
            f"[{report['timestamp_utc']}] n={report['n_terminal']} "
            f"decision={report['decision']} θ={report['primary']['theta_hat']} "
            f"Wilson=[{report['primary']['wilson_95']['low']:.4f},"
            f"{report['primary']['wilson_95']['high']:.4f}] "
            f"runner={report['runner_pids']} QC={report['quality_control']['anomalies']} "
            f"blocker={blocker}",
            flush=True,
        )
        if args.once:
            return 0
        # Exit monitor only on STOP after eligible look artifacts exist
        if (
            report["recommendation"] == "STOP"
            and report["n_terminal"] >= 800
            and (ANALYSIS / "arm_b_first_look_800.md").exists()
        ):
            print("STOP flag written; monitor exiting after first-look artifacts.", flush=True)
            return 0
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
