"""Tests for experiment runner."""

from caliper.config.loader import load_config
from caliper.config.schema import ExperimentConfig
from caliper.runners.experiment import ExperimentRunner


class TestExperimentRunner:
    def test_plan_combinations(self, sample_config: ExperimentConfig) -> None:
        runner = ExperimentRunner(sample_config, dry_run=True)
        combos = runner.plan_combinations()
        # 2 runs × 1 model × 1 temperature × 1 prompt × 1 task = 2
        assert len(combos) == 2
        assert "temperature" in combos[0]

    def test_dry_run_completes(self, sample_config: ExperimentConfig) -> None:
        runner = ExperimentRunner(sample_config, dry_run=True)
        manifest = runner.run()
        assert manifest.status == "completed"
        assert manifest.run_id is not None

    def test_manifest_has_timestamps(self, sample_config: ExperimentConfig) -> None:
        runner = ExperimentRunner(sample_config, dry_run=True)
        manifest = runner.run()
        assert manifest.started_at is not None
        assert manifest.finished_at is not None

    def test_example_config_combinations(self) -> None:
        config = load_config("configs/examples/basic_experiment.yaml")
        runner = ExperimentRunner(config, dry_run=True)
        combos = runner.plan_combinations()
        # 3 runs × 2 models × 2 temps × 2 prompts × 2 tasks = 48
        assert len(combos) == 48

    def test_factorial_config_combinations(self) -> None:
        config = load_config("configs/examples/factorial_power.yaml")
        assert config.total_combinations() == 180
