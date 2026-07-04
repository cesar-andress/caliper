"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from caliper.config.schema import ExperimentConfig


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "experiment_id": "test_experiment",
        "description": "A test experiment",
        "random_seed": 42,
        "providers": {
            "mock": {"type": "mock"},
        },
        "models": [
            {
                "id": "mock-a",
                "provider": "mock",
                "model_id": "mock-v1",
            }
        ],
        "tasks": [
            {
                "id": "test-task",
                "dataset": "test",
            }
        ],
        "prompt_variants": [
            {
                "id": "default",
                "template": "Q: {question}\nA:",
            }
        ],
        "temperatures": [0.0],
        "evaluation_metrics": ["accuracy"],
        "number_of_runs": 2,
    }


@pytest.fixture
def sample_config(sample_config_dict: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(sample_config_dict)


@pytest.fixture
def config_yaml(tmp_path: Path, sample_config_dict: dict) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.dump(sample_config_dict), encoding="utf-8")
    return path
