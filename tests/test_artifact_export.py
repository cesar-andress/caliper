"""Tests for experiment artifact export and verification."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2

import yaml

from caliper.config.loader import load_config
from caliper.runners.artifact_export import (
    REQUIRED_ARTIFACT_FILES,
    REQUIRED_DATA_FILES,
    export_artifact,
    verify_artifact,
    verify_checksums,
    write_checksums,
)
from caliper.runners.experiment import ExperimentRunner

EXAMPLE_FACTORIAL = Path("configs/examples/example_factorial.yaml")


def _tiny_config(tmp_path: Path):
    config = load_config(EXAMPLE_FACTORIAL)
    config_dict = config.model_dump(mode="json")
    config_dict["output"] = {"directory": str(tmp_path / "outputs"), "format": "parquet"}
    config_dict["execution"] = {"shuffle": False, "parallel_workers": 1}
    config_dict["number_of_runs"] = 1
    config_dict["models"] = [config_dict["models"][0]]
    config_dict["tasks"] = [config_dict["tasks"][0]]
    config_dict["prompt_variants"] = [config_dict["prompt_variants"][0]]
    config_dict["temperatures"] = [0.0]

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
    return load_config(config_path), config_path


class TestArtifactExport:
    def test_pipeline_auto_exports_artifact(self, tmp_path: Path) -> None:
        loaded, config_path = _tiny_config(tmp_path)
        runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=False,
        )
        runner.run()

        artifact_dir = runner.output_dir / "artifact"
        assert artifact_dir.is_dir()
        for name in REQUIRED_ARTIFACT_FILES:
            assert (artifact_dir / name).exists(), name
        for rel in REQUIRED_DATA_FILES:
            assert (artifact_dir / rel).exists(), rel

        verification = verify_artifact(artifact_dir)
        assert verification.complete
        assert not verification.missing_files
        assert not verification.errors

    def test_export_cli_command(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from caliper.cli import main

        loaded, config_path = _tiny_config(tmp_path)
        runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=False,
        )
        runner.run()

        cli = CliRunner()
        result = cli.invoke(main, ["export-artifact", str(runner.output_dir)])
        assert result.exit_code == 0
        assert "Artifact exported" in result.output
        assert (runner.output_dir / "artifact" / "README.md").exists()

    def test_verify_detects_missing_files(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "README.md").write_text("stub", encoding="utf-8")

        verification = verify_artifact(artifact_dir)
        assert not verification.complete
        assert verification.missing_files
        assert verification.errors

    def test_verify_warns_on_missing_figures(self, tmp_path: Path) -> None:
        loaded, config_path = _tiny_config(tmp_path)
        runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=False,
        )
        runner.run()
        result = export_artifact(runner.output_dir, force=True)
        assert any("figures" in w for w in result.verification.warnings)

    def test_checksum_verification_detects_tampering(self, tmp_path: Path) -> None:
        loaded, config_path = _tiny_config(tmp_path)
        runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=False,
        )
        runner.run()
        artifact_dir = runner.output_dir / "artifact"

        write_checksums(artifact_dir)
        readme = artifact_dir / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8") + "\n# tampered", encoding="utf-8")

        failures = verify_checksums(artifact_dir)
        assert failures
        verification = verify_artifact(artifact_dir)
        assert not verification.complete
        assert verification.checksum_failures

    def test_incomplete_experiment_export_not_complete(self, tmp_path: Path) -> None:
        loaded, config_path = _tiny_config(tmp_path)
        out = tmp_path / "outputs" / loaded.experiment_id
        out.mkdir(parents=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "experiment_id": loaded.experiment_id,
                    "failed_cells": 2,
                    "git_commit": "unknown",
                }
            ),
            encoding="utf-8",
        )
        copy2(config_path, out / "config.yaml")
        result = export_artifact(out)
        assert not result.verification.complete
        assert result.verification.missing_files
        assert any("status" in w for w in result.verification.warnings)
        assert any("failed cell" in w for w in result.verification.warnings)
