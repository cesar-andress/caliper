"""Export and verify reproducibility artifacts for completed experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from caliper import __version__
from caliper.config.loader import load_config
from caliper.runners.reproducibility import collect_environment, git_commit

logger = structlog.get_logger(__name__)

MIT_LICENSE_TEXT = """\
MIT License

Copyright (c) 2026 César Andrés, David Martín-Moncunill, and José Manuel Baños

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

REQUIRED_ARTIFACT_FILES = (
    "README.md",
    "environment.yml",
    "requirements.txt",
    "Dockerfile",
    "run.sh",
    "reproduce.sh",
    "metadata.json",
    "checksums.txt",
    "CITATION.cff",
    "LICENSE",
)

REQUIRED_DATA_FILES = (
    "data/config.yaml",
    "data/manifest.json",
    "data/results.parquet",
    "data/statistical_dataset.parquet",
    "data/evaluations.parquet",
    "data/report.md",
)

OPTIONAL_DATA_FILES = (
    "data/results.jsonl",
    "data/evaluations.jsonl",
)


@dataclass
class ArtifactVerification:
    """Result of artifact completeness verification."""

    complete: bool
    missing_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checksum_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "missing_files": self.missing_files,
            "warnings": self.warnings,
            "errors": self.errors,
            "checksum_failures": self.checksum_failures,
        }


@dataclass
class ArtifactExportResult:
    """Paths and verification state for an exported artifact bundle."""

    artifact_dir: Path
    verification: ArtifactVerification
    metadata: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return True


def _load_experiment_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_requirements_text(libraries: dict[str, str]) -> str:
    lines = ["# Pinned runtime requirements for artifact reproduction", ""]
    for package in (
        "caliper",
        "click",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
        "python-dotenv",
        "structlog",
        "pandas",
        "pyarrow",
        "numpy",
        "scipy",
        "statsmodels",
        "matplotlib",
    ):
        version = libraries.get(package)
        if version and version not in {"not_installed", "unknown"}:
            lines.append(f"{package}=={version}")
        elif package == "caliper":
            lines.append(f"caliper=={__version__}")
        else:
            lines.append(f"# {package}: version unavailable at export time")
    lines.append("")
    return "\n".join(lines)


def _build_environment_yml(libraries: dict[str, str], experiment_id: str) -> str:
    python_version = libraries.get("python", "3.11")
    major_minor = ".".join(python_version.split(".")[:2])
    pip_lines = []
    for line in _build_requirements_text(libraries).splitlines():
        if line and not line.startswith("#"):
            pip_lines.append(f"    - {line}")
    pip_block = "\n".join(pip_lines) if pip_lines else "    - caliper"
    return f"""\
name: caliper-{experiment_id}
channels:
  - conda-forge
dependencies:
  - python={major_minor}
  - pip
  - pip:
{pip_block}
"""


def _build_dockerfile(experiment_id: str) -> str:
    return f"""\
FROM python:3.11-slim

WORKDIR /artifact
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Full experiment re-run (requires dataset paths from config.yaml)
CMD ["bash", "run.sh"]
"""


def _build_run_sh(experiment_id: str) -> str:
    return f"""\
#!/usr/bin/env bash
# Re-run the full CALIPER experiment from bundled configuration.
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$ROOT"

if ! command -v caliper >/dev/null 2>&1; then
  pip install -r requirements.txt
fi

echo "[run.sh] Executing experiment: {experiment_id}"
caliper run "$ROOT/data/config.yaml"
"""


def _build_reproduce_sh() -> str:
    """Build reproduce.sh with a reliable artifact-dir resolution."""
    return f"""\
#!/usr/bin/env bash
# Regenerate reported tables and figures from frozen experiment outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$ROOT"

if ! command -v caliper >/dev/null 2>&1; then
  pip install -r requirements.txt
fi

echo "[reproduce.sh] Verifying artifact checksums..."
python3 - <<PY
from pathlib import Path
from caliper.runners.artifact_export import verify_artifact

artifact_dir = Path(r"$ROOT")
verification = verify_artifact(artifact_dir)
if verification.errors:
    for err in verification.errors:
        print(f"ERROR: {{err}}")
    raise SystemExit(1)
for warn in verification.warnings:
    print(f"WARNING: {{warn}}")
print("Checksum verification passed.")
PY

mkdir -p output/tables output/figures

echo "[reproduce.sh] Variance decomposition (Paper 1)..."
caliper analyze variance \\
  --results "$ROOT/data/statistical_dataset.parquet" \\
  | tee output/tables/variance_decomposition.txt

echo "[reproduce.sh] Power simulation (Paper 1)..."
caliper analyze power \\
  --results "$ROOT/data/statistical_dataset.parquet" \\
  | tee output/tables/power_simulation.txt

echo "[reproduce.sh] Ranking fragility (Paper 2)..."
caliper ranking-fragility \\
  "$ROOT/data/results.parquet" \\
  --output-dir "$ROOT/output/ranking_fragility" \\
  --reports-dir "$ROOT/output/figures/ranking_fragility"

if [ -d "$ROOT/data/figures" ] && [ "$(ls -A "$ROOT/data/figures" 2>/dev/null)" ]; then
  mkdir -p "$ROOT/output/figures/experiment"
  cp -r "$ROOT/data/figures/." "$ROOT/output/figures/experiment/"
fi

echo "[reproduce.sh] Done. Outputs in output/tables and output/figures."
"""


def _build_readme(experiment_id: str, manifest: dict[str, Any]) -> str:
    exp_hash = manifest.get("experiment_hash", "unknown")
    config_hash = manifest.get("configuration_hash", "unknown")
    git = manifest.get("git_commit", "unknown")
    return f"""\
# CALIPER reproduction artifact: {experiment_id}

This directory contains a self-contained bundle sufficient to reproduce the
tables and figures reported for experiment `{experiment_id}`.

## Contents

| File | Purpose |
|------|---------|
| `data/` | Frozen experiment outputs (config, results, evaluations, report) |
| `reproduce.sh` | Regenerate analysis tables and figures from frozen data |
| `run.sh` | Re-run the full experiment (requires live datasets/API keys) |
| `requirements.txt` | Pinned Python dependencies |
| `environment.yml` | Conda environment specification |
| `Dockerfile` | Container image for isolated reproduction |
| `metadata.json` | Provenance and export metadata |
| `checksums.txt` | SHA-256 checksums for integrity verification |
| `CITATION.cff` | Citation metadata (CFF format) |
| `LICENSE` | MIT License |

## Quick start (tables and figures)

```bash
pip install -r requirements.txt
./reproduce.sh
```

Outputs appear under `output/tables/` and `output/figures/`.

## Full experiment re-run

```bash
pip install -r requirements.txt
./run.sh
```

Note: full re-execution requires dataset paths referenced in `data/config.yaml`
and any configured API keys or local model weights.

## Provenance

- **Experiment hash**: `{exp_hash}`
- **Configuration hash**: `{config_hash}`
- **Git commit**: `{git}`
- **CALIPER version**: {manifest.get("software_version", __version__)}

## Verification

```bash
python3 -c "from pathlib import Path; from caliper.runners.artifact_export import verify_artifact; print(verify_artifact(Path('.')).to_dict())"
```
"""


def _build_citation_cff(experiment_id: str, manifest: dict[str, Any]) -> str:
    return f"""\
cff-version: 1.2.0
message: "If you use this artifact, please cite CALIPER and the accompanying papers."
title: "CALIPER experiment artifact: {experiment_id}"
version: "{manifest.get('software_version', __version__)}"
doi: "10.5281/zenodo.21780089"
date-released: "{datetime.now(tz=UTC).date().isoformat()}"
authors:
  - family-names: Andrés
    given-names: César
    orcid: "https://orcid.org/0009-0001-8968-3404"
  - family-names: Martín-Moncunill
    given-names: David
    orcid: "https://orcid.org/0000-0003-2422-9005"
  - family-names: Baños
    given-names: José Manuel
    orcid: "https://orcid.org/0009-0004-9971-7390"
license: MIT
keywords:
  - large language models
  - evaluation
  - reproducibility
  - factorial design
repository-code: "https://github.com/cesar-andress/caliper"
url: "https://doi.org/10.5281/zenodo.21780089"
abstract: >
  Reproducibility bundle for CALIPER factorial experiment {experiment_id}.
  Includes frozen results and scripts to regenerate reported tables and figures.
  Software archive DOI: https://doi.org/10.5281/zenodo.21780089.
"""


def _bundle_experiment_data(experiment_dir: Path, data_dir: Path) -> list[str]:
    """Copy required experiment outputs into artifact/data/. Returns copied paths."""
    copied: list[str] = []
    sources = {
        "config.yaml": experiment_dir / "config.yaml",
        "manifest.json": experiment_dir / "manifest.json",
        "results.parquet": experiment_dir / "results.parquet",
        "results.jsonl": experiment_dir / "results.jsonl",
        "statistical_dataset.parquet": experiment_dir / "statistical_dataset.parquet",
        "evaluations.parquet": experiment_dir / "evaluations.parquet",
        "evaluations.jsonl": experiment_dir / "evaluations.jsonl",
        "report.md": experiment_dir / "report.md",
    }
    for name, source in sources.items():
        if _copy_if_exists(source, data_dir / name):
            copied.append(f"data/{name}")

    figures_src = experiment_dir / "figures"
    if _copy_if_exists(figures_src, data_dir / "figures"):
        copied.append("data/figures/")
    return copied


def write_checksums(artifact_dir: Path) -> Path:
    """Write SHA-256 checksums for all artifact files except checksums.txt."""
    lines: list[str] = []
    files = sorted(
        path
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "checksums.txt"
    )
    for path in files:
        rel = path.relative_to(artifact_dir).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}")
    checksums_path = artifact_dir / "checksums.txt"
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums_path


def verify_checksums(artifact_dir: Path) -> list[str]:
    """Return relative paths whose checksums do not match checksums.txt."""
    checksums_path = artifact_dir / "checksums.txt"
    if not checksums_path.exists():
        return ["checksums.txt"]

    failures: list[str] = []
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel_path = line.split("  ", 1)
        target = artifact_dir / rel_path
        if not target.exists():
            failures.append(rel_path)
            continue
        if _sha256_file(target) != digest:
            failures.append(rel_path)
    return failures


def verify_artifact(artifact_dir: Path) -> ArtifactVerification:
    """Verify that an artifact bundle is complete and internally consistent."""
    result = ArtifactVerification(complete=True)

    for name in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / name).exists():
            result.missing_files.append(name)
            result.complete = False

    for rel in REQUIRED_DATA_FILES:
        if not (artifact_dir / rel).exists():
            result.missing_files.append(rel)
            result.complete = False

    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            exp_manifest = metadata.get("experiment_manifest", {})
            if exp_manifest.get("status") != "completed":
                result.warnings.append(
                    f"experiment status is '{exp_manifest.get('status')}', not 'completed'"
                )
            if exp_manifest.get("failed_cells", 0) > 0:
                result.warnings.append(
                    f"experiment has {exp_manifest['failed_cells']} failed cell(s)"
                )
            if metadata.get("git_commit") == "unknown":
                result.warnings.append("git commit was unknown at export time")
            if not metadata.get("bundled_files"):
                result.warnings.append("metadata lists no bundled data files")
        except json.JSONDecodeError:
            result.errors.append("metadata.json is not valid JSON")
            result.complete = False
    elif "metadata.json" not in result.missing_files:
        result.errors.append("metadata.json exists but could not be validated")

    figures_dir = artifact_dir / "data" / "figures"
    if not figures_dir.exists() or not any(figures_dir.iterdir()):
        result.warnings.append("no experiment figures bundled in data/figures/")

    checksum_failures = verify_checksums(artifact_dir)
    if checksum_failures:
        result.checksum_failures = checksum_failures
        result.errors.append(
            f"{len(checksum_failures)} file(s) failed checksum verification"
        )
        result.complete = False

    if result.missing_files:
        result.errors.append(
            f"{len(result.missing_files)} required file(s) missing from artifact"
        )

    return result


def export_artifact(
    experiment_dir: Path,
    *,
    artifact_dir: Path | None = None,
    force: bool = False,
) -> ArtifactExportResult:
    """Export a reproduction artifact bundle for a completed experiment."""
    experiment_dir = experiment_dir.resolve()
    if artifact_dir is None:
        artifact_dir = experiment_dir / "artifact"
    else:
        artifact_dir = artifact_dir.resolve()

    exp_manifest = _load_experiment_manifest(experiment_dir)
    experiment_id = exp_manifest.get("experiment_id", experiment_dir.name)
    env = collect_environment()
    libraries = env["libraries"]

    if artifact_dir.exists():
        if force:
            shutil.rmtree(artifact_dir)
        else:
            shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    data_dir = artifact_dir / "data"
    bundled = _bundle_experiment_data(experiment_dir, data_dir)

    config_path = experiment_dir / "config.yaml"
    config_summary: dict[str, Any] = {}
    if config_path.exists():
        try:
            config_summary = load_config(config_path).model_dump(mode="json")
        except Exception as exc:
            logger.warning("artifact.config_load_failed", error=str(exc))

    export_metadata: dict[str, Any] = {
        "artifact_version": "1.0",
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "caliper_version": __version__,
        "git_commit": git_commit(),
        "experiment_hash": exp_manifest.get("experiment_hash"),
        "configuration_hash": exp_manifest.get("configuration_hash"),
        "experiment_manifest": exp_manifest,
        "config_summary": {
            "experiment_id": config_summary.get("experiment_id"),
            "random_seed": config_summary.get("random_seed"),
            "factorial_axes": exp_manifest.get("factorial_axes"),
        },
        "bundled_files": bundled,
        "environment": env,
    }

    (artifact_dir / "README.md").write_text(
        _build_readme(experiment_id, exp_manifest), encoding="utf-8"
    )
    (artifact_dir / "environment.yml").write_text(
        _build_environment_yml(libraries, experiment_id), encoding="utf-8"
    )
    (artifact_dir / "requirements.txt").write_text(
        _build_requirements_text(libraries), encoding="utf-8"
    )
    (artifact_dir / "Dockerfile").write_text(
        _build_dockerfile(experiment_id), encoding="utf-8"
    )
    _write_executable(artifact_dir / "run.sh", _build_run_sh(experiment_id))
    _write_executable(artifact_dir / "reproduce.sh", _build_reproduce_sh())
    (artifact_dir / "metadata.json").write_text(
        json.dumps(export_metadata, indent=2, default=str), encoding="utf-8"
    )
    (artifact_dir / "CITATION.cff").write_text(
        _build_citation_cff(experiment_id, exp_manifest), encoding="utf-8"
    )
    (artifact_dir / "LICENSE").write_text(MIT_LICENSE_TEXT, encoding="utf-8")

    write_checksums(artifact_dir)

    verification = verify_artifact(artifact_dir)
    export_metadata["verification"] = verification.to_dict()
    (artifact_dir / "metadata.json").write_text(
        json.dumps(export_metadata, indent=2, default=str), encoding="utf-8"
    )
    write_checksums(artifact_dir)

    if verification.warnings:
        for warning in verification.warnings:
            logger.warning("artifact.export_warning", message=warning)
    if not verification.complete:
        for error in verification.errors:
            logger.error("artifact.export_error", message=error)

    logger.info(
        "artifact.exported",
        artifact_dir=str(artifact_dir),
        complete=verification.complete,
        warnings=len(verification.warnings),
        missing=len(verification.missing_files),
    )

    return ArtifactExportResult(
        artifact_dir=artifact_dir,
        verification=verification,
        metadata=export_metadata,
    )
