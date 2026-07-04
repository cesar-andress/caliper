"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from caliper.config.errors import ConfigValidationError
from caliper.config.loader import load_config, validate_config
from caliper.config.schema import ExperimentConfig


class TestExperimentConfig:
    def test_valid_config(self, sample_config: ExperimentConfig) -> None:
        assert sample_config.experiment_id == "test_experiment"
        assert sample_config.random_seed == 42
        assert len(sample_config.models) == 1
        assert len(sample_config.tasks) == 1
        assert sample_config.evaluation_metrics == ["accuracy"]

    def test_requires_at_least_one_model(self, sample_config_dict: dict) -> None:
        sample_config_dict["models"] = []
        with pytest.raises(ValidationError, match="at least one entry"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_requires_at_least_one_task(self, sample_config_dict: dict) -> None:
        sample_config_dict["tasks"] = []
        with pytest.raises(ValidationError, match="at least one entry"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_default_decoding(self, sample_config: ExperimentConfig) -> None:
        assert sample_config.decoding.top_p == 1.0
        assert sample_config.decoding.max_tokens == 1024

    def test_default_number_of_runs(self, sample_config_dict: dict) -> None:
        del sample_config_dict["number_of_runs"]
        config = ExperimentConfig.model_validate(sample_config_dict)
        assert config.number_of_runs == 1

    def test_factorial_axes(self, sample_config: ExperimentConfig) -> None:
        axes = sample_config.factorial_axes()
        assert axes == {
            "models": 1,
            "temperatures": 1,
            "prompt_variants": 1,
            "tasks": 1,
            "runs": 2,
        }

    def test_total_combinations(self, sample_config: ExperimentConfig) -> None:
        assert sample_config.total_combinations() == 2

    def test_iter_combinations(self, sample_config: ExperimentConfig) -> None:
        combos = list(sample_config.iter_combinations())
        assert len(combos) == 2
        assert combos[0].temperature == 0.0
        assert combos[0].model_id == "mock-a"

    def test_metrics_for_task_uses_defaults(self, sample_config: ExperimentConfig) -> None:
        assert sample_config.metrics_for_task("test-task") == ["accuracy"]

    def test_metrics_for_task_override(self, sample_config_dict: dict) -> None:
        sample_config_dict["tasks"][0]["metrics"] = ["exact_match"]
        config = ExperimentConfig.model_validate(sample_config_dict)
        assert config.metrics_for_task("test-task") == ["exact_match"]

    def test_invalid_experiment_id(self, sample_config_dict: dict) -> None:
        sample_config_dict["experiment_id"] = "Bad-ID"
        with pytest.raises(ValidationError, match="experiment_id"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_empty_temperatures(self, sample_config_dict: dict) -> None:
        sample_config_dict["temperatures"] = []
        with pytest.raises(ValidationError, match="at least one value"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_temperature_out_of_range(self, sample_config_dict: dict) -> None:
        sample_config_dict["temperatures"] = [3.0]
        with pytest.raises(ValidationError, match="out of range"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_empty_evaluation_metrics(self, sample_config_dict: dict) -> None:
        sample_config_dict["evaluation_metrics"] = []
        with pytest.raises(ValidationError, match="at least one metric"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_unknown_evaluation_metric(self, sample_config_dict: dict) -> None:
        sample_config_dict["evaluation_metrics"] = ["not_a_metric"]
        with pytest.raises(ValidationError, match="unknown evaluation metric"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_unknown_provider_reference(self, sample_config_dict: dict) -> None:
        sample_config_dict["models"][0]["provider"] = "nonexistent"
        with pytest.raises(ValidationError, match="unknown provider"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_duplicate_model_ids(self, sample_config_dict: dict) -> None:
        sample_config_dict["models"].append(
            {"id": "mock-a", "provider": "mock", "model_id": "mock-v2"}
        )
        with pytest.raises(ValidationError, match="duplicate model id"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_duplicate_task_ids(self, sample_config_dict: dict) -> None:
        sample_config_dict["tasks"].append({"id": "test-task", "dataset": "other"})
        with pytest.raises(ValidationError, match="duplicate task id"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_duplicate_prompt_variant_ids(self, sample_config_dict: dict) -> None:
        sample_config_dict["prompt_variants"].append(
            {"id": "default", "template": "other {question}"}
        )
        with pytest.raises(ValidationError, match="duplicate prompt variant id"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_prompt_variant_requires_template_or_path(self, sample_config_dict: dict) -> None:
        sample_config_dict["prompt_variants"] = [{"id": "empty"}]
        with pytest.raises(ValidationError, match="must specify 'template' or 'path'"):
            ExperimentConfig.model_validate(sample_config_dict)

    def test_task_unknown_metrics(self, sample_config_dict: dict) -> None:
        sample_config_dict["tasks"][0]["metrics"] = ["bad_metric"]
        with pytest.raises(ValidationError, match="unknown metric"):
            ExperimentConfig.model_validate(sample_config_dict)


class TestLoadConfig:
    def test_load_from_yaml(self, config_yaml: Path) -> None:
        config = load_config(config_yaml)
        assert config.experiment_id == "test_experiment"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("/nonexistent/config.yaml")

    def test_load_example_config(self) -> None:
        config = load_config("configs/examples/basic_experiment.yaml")
        assert config.experiment_id == "basic_experiment"
        assert len(config.models) == 2
        assert len(config.tasks) == 2
        assert config.temperatures == [0.0, 0.7]

    def test_load_factorial_example(self) -> None:
        config = load_config("configs/examples/factorial_power.yaml")
        assert config.total_combinations() == 180

    def test_validation_error_has_useful_message(self, tmp_path: Path) -> None:
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text(yaml.dump({"experiment_id": "x"}), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(bad_config)
        assert "Invalid experiment config" in str(exc_info.value)
        assert len(exc_info.value.errors) > 0

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        from caliper.config.errors import ConfigParseError

        bad = tmp_path / "broken.yaml"
        bad.write_text("experiment_id: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigParseError):
            load_config(bad)

    def test_missing_prompt_path(self, tmp_path: Path, sample_config_dict: dict) -> None:
        sample_config_dict["prompt_variants"] = [
            {"id": "file-prompt", "path": "prompts/missing.txt"}
        ]
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.dump(sample_config_dict), encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="file not found"):
            load_config(path)


class TestValidateConfig:
    def test_valid_returns_empty(self, config_yaml: Path) -> None:
        assert validate_config(config_yaml) == []

    def test_invalid_returns_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("experiment_id: bad id\n", encoding="utf-8")
        errors = validate_config(path)
        assert len(errors) > 0

    def test_missing_file(self) -> None:
        errors = validate_config("/nonexistent/config.yaml")
        assert any("not found" in e for e in errors)
