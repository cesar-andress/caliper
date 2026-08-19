"""Capture full reproducibility provenance for Arms A/B."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

import caliper

from .constants import (
    ANALYSIS_DIR,
    ARM_CONFIGS,
    EXPECTED_FROZEN_SHA256,
    FROZEN_PARQUET,
    RANDOM_SEED,
    REPO_ROOT,
)
from .loaders import resolve_arm_run_dir, sha256_file


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def _http_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> Any | None:
    try:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(url, data=data, headers=headers)
        else:
            req = request.Request(url, headers=headers)
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def collect_provenance(arms: list[str] | None = None) -> dict[str, Any]:
    arms = arms or ["a", "b"]
    ollama_version = _http_json("http://localhost:11434/api/version") or {}
    tags = _http_json("http://localhost:11434/api/tags") or {}
    qwen = None
    for model in tags.get("models", []):
        if model.get("name") == "qwen3:32b":
            qwen = model
            break

    git_commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    git_status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    git_dirty = bool(git_status)

    nvidia = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,cuda_version",
            "--format=csv,noheader",
        ]
    )
    # cuda_version may not exist on all nvidia-smi; fallback
    if nvidia is None:
        nvidia = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )

    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])

    arm_blocks = {}
    for arm in arms:
        run_dir = resolve_arm_run_dir(arm)
        block: dict[str, Any] = {
            "config_path": str(ARM_CONFIGS[arm]),
            "config_sha256": sha256_file(ARM_CONFIGS[arm]) if ARM_CONFIGS[arm].exists() else None,
            "run_dir": str(run_dir) if run_dir else None,
            "random_seed": RANDOM_SEED,
            "experiment_id": f"paper1_confirmatory_humaneval_qwen3_v11_arm_{arm}",
        }
        if run_dir and (run_dir / "manifest.json").exists():
            block["manifest"] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if run_dir and (run_dir / "results.jsonl").exists():
            block["results_jsonl_sha256"] = sha256_file(run_dir / "results.jsonl")
        if run_dir and (run_dir / "statistical_dataset.parquet").exists():
            block["statistical_dataset_sha256"] = sha256_file(run_dir / "statistical_dataset.parquet")
        arm_blocks[arm] = block

    frozen_sha = sha256_file(FROZEN_PARQUET) if FROZEN_PARQUET.exists() else None

    manifest = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "software": {
            "caliper_version": getattr(caliper, "__version__", None),
            "python_version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "os_release": Path("/etc/os-release").read_text(encoding="utf-8")
            if Path("/etc/os-release").exists()
            else None,
        },
        "git": {
            "commit": git_commit,
            "dirty": git_dirty,
            "status_porcelain": git_status,
            "repo": str(REPO_ROOT),
        },
        "provider": {
            "type": "ollama",
            "base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            "ollama_version_api": ollama_version,
            "ollama_cli_version": _run(["ollama", "--version"]),
            "model_name": "qwen3:32b",
            "model_record": qwen,
        },
        "hardware": {
            "nvidia_smi": nvidia,
            "cpu": platform.processor() or platform.machine(),
        },
        "dependencies_pip_freeze": pip_freeze.splitlines() if pip_freeze else None,
        "frozen_v1_0": {
            "path": str(FROZEN_PARQUET),
            "sha256": frozen_sha,
            "expected_sha256": EXPECTED_FROZEN_SHA256,
            "immutable_match": frozen_sha == EXPECTED_FROZEN_SHA256,
        },
        "arms": arm_blocks,
        "analysis_seed": RANDOM_SEED,
        "notes": [
            "v1.0 freeze must remain byte-immutable.",
            "Arms A/B are outside the freeze and use CALIPER >= 1.1.0.",
            "cell_id hash excludes experiment_id; join freeze↔arms on cell_id is valid.",
        ],
    }
    return manifest


def write_provenance(path: Path | None = None) -> dict[str, Any]:
    path = path or (ANALYSIS_DIR / "provenance_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect_provenance()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = path.with_suffix(".md")
    lines = [
        "# Provenance manifest — qwen3 v1.1 corrective arms",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Software",
        f"- CALIPER: `{manifest['software']['caliper_version']}`",
        f"- Python: `{manifest['software']['python_version'].split()[0]}`",
        f"- Platform: `{manifest['software']['platform']}`",
        "",
        "## Git",
        f"- Commit: `{manifest['git']['commit']}`",
        f"- Dirty working tree: `{manifest['git']['dirty']}`",
        "",
        "## Provider / model",
        f"- Ollama API: `{manifest['provider']['ollama_version_api']}`",
        f"- Ollama CLI: `{manifest['provider']['ollama_cli_version']}`",
        f"- Model: `qwen3:32b`",
        f"- Digest: `{((manifest['provider'].get('model_record') or {}).get('digest'))}`",
        f"- Quantization: `{((manifest['provider'].get('model_record') or {}).get('details') or {}).get('quantization_level')}`",
        "",
        "## Hardware",
        f"- `{manifest['hardware']['nvidia_smi']}`",
        "",
        "## Frozen v1.0",
        f"- Path: `{manifest['frozen_v1_0']['path']}`",
        f"- SHA256: `{manifest['frozen_v1_0']['sha256']}`",
        f"- Matches expected: `{manifest['frozen_v1_0']['immutable_match']}`",
        "",
        "## Arms",
    ]
    for arm, block in manifest["arms"].items():
        lines.append(f"### Arm {arm.upper()}")
        lines.append(f"- Experiment ID: `{block['experiment_id']}`")
        lines.append(f"- Config: `{block['config_path']}`")
        lines.append(f"- Config SHA256: `{block['config_sha256']}`")
        lines.append(f"- Run dir: `{block['run_dir']}`")
        lines.append(f"- Seed: `{block['random_seed']}`")
        lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    write_provenance()
