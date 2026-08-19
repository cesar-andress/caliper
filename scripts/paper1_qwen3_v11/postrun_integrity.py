"""Post-run integrity checks for a completed qwen3 v1.1 arm."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    ANALYSIS_DIR,
    ARM_CONFIGS,
    ARM_PROTOCOL,
    EXPECTED_CELLS,
    REQUIRED_PROVIDER_FIELDS,
)
from .loaders import load_arm_results, resolve_arm_run_dir, sha256_file


def _check(name: str, ok: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"check": name, "ok": ok, "detail": detail, "severity": severity}


def run_arm_integrity(arm: str) -> dict[str, Any]:
    run_dir = resolve_arm_run_dir(arm)
    protocol = ARM_PROTOCOL[arm]
    report: dict[str, Any] = {
        "arm": arm,
        "protocol": protocol,
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "run_dir": str(run_dir) if run_dir else None,
        "checks": [],
        "summary": {},
    }
    checks: list[dict[str, Any]] = report["checks"]

    if run_dir is None or not run_dir.exists():
        checks.append(_check("run_dir_exists", False, "output directory missing"))
        report["summary"] = {"passed": 0, "failed": 1, "complete": False}
        return report

    checks.append(_check("run_dir_exists", True, str(run_dir)))

    jsonl = run_dir / "results.jsonl"
    parquet = run_dir / "results.parquet"
    stat = run_dir / "statistical_dataset.parquet"
    eval_pq = run_dir / "evaluations.parquet"
    eval_jsonl = run_dir / "evaluations.jsonl"
    manifest = run_dir / "manifest.json"
    checkpoint_state = run_dir / "checkpoint_state.json"
    checkpoints = run_dir / "checkpoints"
    raw_dir = run_dir / "raw_responses"
    config_snap = run_dir / "config.yaml"

    try:
        df = load_arm_results(arm)
    except Exception as exc:
        checks.append(_check("load_results", False, str(exc)))
        report["summary"] = {"passed": 0, "failed": 1, "complete": False}
        return report

    n = len(df)
    checks.append(_check("load_results", True, f"loaded {n} rows"))

    # Cell counts
    checks.append(
        _check(
            "expected_cell_count",
            n == EXPECTED_CELLS,
            f"n={n}, expected={EXPECTED_CELLS}",
        )
    )
    status_counts = df["status"].astype(str).value_counts().to_dict()
    completed = int(status_counts.get("completed", 0))
    budget = int(status_counts.get("budget_exhausted", 0))
    failed = int(status_counts.get("failed", 0))
    checks.append(
        _check(
            "completed_or_budget_cells",
            (completed + budget) == n and failed == 0,
            f"status_counts={status_counts}",
        )
    )
    checks.append(_check("failed_cells_zero", failed == 0, f"failed={failed}"))

    # Duplicates / missing
    dup = int(df["cell_id"].duplicated().sum()) if "cell_id" in df.columns else -1
    checks.append(_check("no_duplicate_cell_ids", dup == 0, f"duplicates={dup}"))
    unique = int(df["cell_id"].nunique()) if "cell_id" in df.columns else n
    checks.append(
        _check(
            "unique_cell_ids",
            unique == EXPECTED_CELLS,
            f"unique={unique}, expected={EXPECTED_CELLS}",
        )
    )

    # Factorial coverage
    n_tasks = df["task_id"].nunique()
    n_prompts = df["prompt_id"].nunique()
    n_temps = df["temperature"].nunique()
    n_runs = df["run_index"].nunique()
    checks.append(
        _check(
            "factorial_axes",
            n_tasks == 164 and n_prompts == 4 and n_temps == 2 and n_runs == 5,
            f"tasks={n_tasks}, prompts={n_prompts}, temps={n_temps}, runs={n_runs}",
        )
    )
    expected_combos = 164 * 4 * 2 * 5
    combo_n = df.groupby(["task_id", "prompt_id", "temperature", "run_index"]).ngroups
    checks.append(
        _check(
            "full_factorial_combos",
            combo_n == expected_combos,
            f"combos={combo_n}, expected={expected_combos}",
        )
    )

    # Checkpoint consistency
    if checkpoints.exists():
        ck_files = list(checkpoints.glob("*.json"))
        ck_ids = set()
        for path in ck_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                cid = payload.get("cell_id") or path.stem
                ck_ids.add(cid)
            except Exception:
                continue
        result_ids = set(df["cell_id"].astype(str))
        missing_ck = sorted(result_ids - ck_ids)
        extra_ck = sorted(ck_ids - result_ids)
        checks.append(
            _check(
                "checkpoint_consistency",
                len(missing_ck) == 0,
                f"checkpoint_files={len(ck_files)}, missing_in_ck={len(missing_ck)}, extra_in_ck={len(extra_ck)}",
                severity="warning" if missing_ck else "error",
            )
        )
    else:
        checks.append(_check("checkpoint_dir", False, "checkpoints/ missing", "warning"))

    if checkpoint_state.exists():
        state = json.loads(checkpoint_state.read_text(encoding="utf-8"))
        checks.append(
            _check(
                "checkpoint_state_present",
                True,
                f"completed_cells_field={state.get('completed_cells')}",
            )
        )

    # Parquet / statistical_dataset / evaluations
    checks.append(_check("results_parquet", parquet.exists(), str(parquet)))
    if parquet.exists() and jsonl.exists():
        pq = pd.read_parquet(parquet)
        checks.append(
            _check(
                "parquet_jsonl_rowcount",
                len(pq) == n,
                f"parquet={len(pq)}, jsonl={n}",
            )
        )
        if "cell_id" in pq.columns:
            checks.append(
                _check(
                    "parquet_jsonl_cell_id_set",
                    set(pq["cell_id"]) == set(df["cell_id"]),
                    "cell_id sets compared",
                )
            )

    checks.append(_check("statistical_dataset", stat.exists(), str(stat)))
    if stat.exists():
        sdf = pd.read_parquet(stat)
        checks.append(
            _check(
                "statistical_dataset_rowcount",
                len(sdf) == EXPECTED_CELLS,
                f"n={len(sdf)}",
            )
        )

    has_eval = eval_pq.exists() or eval_jsonl.exists()
    checks.append(_check("evaluations_present", has_eval, f"parquet={eval_pq.exists()} jsonl={eval_jsonl.exists()}"))
    if eval_pq.exists():
        edf = pd.read_parquet(eval_pq)
        checks.append(
            _check(
                "evaluations_rowcount",
                len(edf) >= EXPECTED_CELLS,
                f"n={len(edf)}",
                severity="warning",
            )
        )

    checks.append(_check("manifest_present", manifest.exists(), str(manifest)))
    checks.append(
        _check(
            "config_snapshot",
            config_snap.exists() or ARM_CONFIGS[arm].exists(),
            f"run_snapshot={config_snap.exists()}, source_config={ARM_CONFIGS[arm].exists()}",
        )
    )

    # Provider metadata completeness
    for field in REQUIRED_PROVIDER_FIELDS:
        if field not in df.columns:
            checks.append(_check(f"meta_field_{field}", False, "column missing"))
            continue
        non_null = int(df[field].notna().sum())
        # thinking_sha256 may be null when thinking_length==0
        if field == "thinking_sha256":
            ok = True
            detail = f"non_null={non_null}/{n} (null allowed when no thinking)"
        elif field in {"done_reason", "eval_count", "prompt_eval_count"}:
            ok = non_null == n
            detail = f"non_null={non_null}/{n}"
        else:
            ok = True
            detail = f"non_null={non_null}/{n}"
        checks.append(_check(f"meta_field_{field}", ok, detail))

    done_counts = df["done_reason"].fillna("null").astype(str).value_counts().to_dict()
    checks.append(_check("done_reason_distribution", True, json.dumps(done_counts)))

    # Raw responses
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.json"))
        raw_ids = {p.stem for p in raw_files}
        missing_raw = set(df["cell_id"].astype(str)) - raw_ids
        checks.append(
            _check(
                "raw_responses_complete",
                len(missing_raw) == 0 and len(raw_files) == n,
                f"raw_files={len(raw_files)}, missing={len(missing_raw)}",
            )
        )
        # Spot-check one payload
        if raw_files:
            sample = json.loads(raw_files[0].read_text(encoding="utf-8"))
            has_payload = "raw_metadata" in sample and (
                isinstance(sample["raw_metadata"], dict)
                and ("ollama" in sample["raw_metadata"] or sample["raw_metadata"])
            )
            checks.append(
                _check(
                    "raw_response_provider_payload",
                    has_payload,
                    f"sample_keys={sorted(sample.keys())}",
                )
            )
    else:
        checks.append(_check("raw_responses_dir", False, "raw_responses/ missing"))

    # Protocol metadata
    if "metadata" in df.columns:
        thinks = []
        nums = []
        for meta in df["metadata"]:
            if isinstance(meta, dict):
                thinks.append(meta.get("think"))
                nums.append(meta.get("num_predict"))
        think_ok = all(str(t).lower() in {str(protocol["think"]).lower(), "false", "true"} for t in thinks) if thinks else False
        # normalize: protocol think False should match 'false' or False
        def _as_bool(v: Any) -> bool | None:
            if v is None:
                return None
            if isinstance(v, bool):
                return v
            s = str(v).lower()
            if s == "false":
                return False
            if s == "true":
                return True
            return None

        think_vals = {_as_bool(t) for t in thinks}
        num_vals = set(nums)
        checks.append(
            _check(
                "protocol_think",
                think_vals == {protocol["think"]},
                f"observed_think={sorted(str(x) for x in think_vals)}",
            )
        )
        checks.append(
            _check(
                "protocol_num_predict",
                num_vals == {protocol["num_predict"]},
                f"observed_num_predict={sorted(num_vals)}",
            )
        )

    # Hashes
    hashes = {}
    for path in (jsonl, parquet, stat, manifest, config_snap):
        if path.exists():
            hashes[path.name] = sha256_file(path)
    report["hashes"] = hashes
    checks.append(_check("sha256_recorded", bool(hashes), f"n_hashed_files={len(hashes)}"))

    # Reproducibility metadata
    repro = {
        "experiment_id": str(df["experiment_id"].iloc[0]) if n else None,
        "run_ids": sorted(df["run_id"].astype(str).unique().tolist()) if n else [],
        "seed_unique": sorted(df["seed"].dropna().unique().tolist())[:20] if "seed" in df.columns else [],
        "model": sorted(df["model"].astype(str).unique().tolist()) if "model" in df.columns else [],
    }
    report["reproducibility"] = repro
    checks.append(
        _check(
            "single_model_qwen3",
            repro["model"] == ["qwen3_32b"],
            f"models={repro['model']}",
        )
    )

    # Empty / budget summary (descriptive only — not publication claims)
    empty_n = int(df["empty"].sum()) if "empty" in df.columns else int((df["prediction"].fillna("").astype(str).str.len() == 0).sum())
    report["descriptive_snapshot"] = {
        "n": n,
        "empty_n": empty_n,
        "empty_rate": empty_n / n if n else None,
        "budget_exhausted_n": budget,
        "pass_mean": float(df["pass_at_1"].mean()) if "pass_at_1" in df.columns else None,
        "NOTE": "Snapshot only; publication numbers require both arms + final analysis gate.",
    }

    passed = sum(1 for c in checks if c["ok"])
    failed_n = sum(1 for c in checks if not c["ok"] and c["severity"] == "error")
    warn_n = sum(1 for c in checks if not c["ok"] and c["severity"] == "warning")
    report["summary"] = {
        "passed": passed,
        "failed": failed_n,
        "warnings": warn_n,
        "complete": failed_n == 0 and n == EXPECTED_CELLS and stat.exists(),
        "ready_for_scientific_use": failed_n == 0 and n == EXPECTED_CELLS and stat.exists() and has_eval,
    }
    return report


def write_integrity_markdown(reports: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Post-run integrity report — qwen3 v1.1 corrective arms",
        "",
        f"Generated: {datetime.now(tz=UTC).isoformat()}",
        "",
        "This report is produced by `scripts/paper1_qwen3_v11/postrun_integrity.py`.",
        "It does **not** authorize manuscript number updates by itself.",
        "",
    ]
    for arm, report in reports.items():
        lines.append(f"## Arm {arm.upper()}")
        lines.append("")
        if not report:
            lines.append("Status: **WAITING** — no report yet.")
            lines.append("")
            continue
        summary = report.get("summary", {})
        lines.append(f"- Run dir: `{report.get('run_dir')}`")
        lines.append(f"- Protocol: `{report.get('protocol')}`")
        lines.append(
            f"- Summary: passed={summary.get('passed')} failed={summary.get('failed')} "
            f"warnings={summary.get('warnings')} complete={summary.get('complete')} "
            f"ready_for_scientific_use={summary.get('ready_for_scientific_use')}"
        )
        snap = report.get("descriptive_snapshot")
        if snap:
            lines.append(
                f"- Descriptive snapshot (not for publication): n={snap.get('n')} "
                f"empty_rate={snap.get('empty_rate')} pass_mean={snap.get('pass_mean')}"
            )
        lines.append("")
        lines.append("| Check | OK | Severity | Detail |")
        lines.append("|-------|----|----------|--------|")
        for c in report.get("checks", []):
            detail = str(c["detail"]).replace("|", "\\|")
            lines.append(
                f"| `{c['check']}` | {'✓' if c['ok'] else '✗'} | {c['severity']} | {detail} |"
            )
        lines.append("")
        if report.get("hashes"):
            lines.append("### SHA256")
            lines.append("")
            for name, digest in report["hashes"].items():
                lines.append(f"- `{name}`: `{digest}`")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Also dump JSON
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")


def run_and_write(arms: list[str] | None = None) -> dict[str, Any]:
    arms = arms or ["a", "b"]
    reports = {}
    for arm in arms:
        try:
            reports[arm] = run_arm_integrity(arm)
        except Exception as exc:
            reports[arm] = {
                "arm": arm,
                "error": str(exc),
                "summary": {"complete": False, "failed": 1, "passed": 0},
            }
    out = ANALYSIS_DIR / "postrun_integrity_report.md"
    write_integrity_markdown(reports, out)
    return reports


if __name__ == "__main__":
    run_and_write()
