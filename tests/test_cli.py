"""Tests for the CLI."""

from click.testing import CliRunner

from caliper.cli import main


class TestCLI:
    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_version(self) -> None:
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.1.0" in result.output

    def test_validate_example_config(self) -> None:
        result = self.runner.invoke(
            main, ["validate", "--config", "configs/examples/basic_experiment.yaml"]
        )
        assert result.exit_code == 0
        assert "valid" in result.output

    def test_plan_example_config(self) -> None:
        result = self.runner.invoke(
            main, ["plan", "--config", "configs/examples/basic_experiment.yaml"]
        )
        assert result.exit_code == 0
        assert "Combinations (48)" in result.output

    def test_dry_run(self) -> None:
        result = self.runner.invoke(
            main, ["run", "--config", "configs/examples/basic_experiment.yaml", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "completed" in result.output
