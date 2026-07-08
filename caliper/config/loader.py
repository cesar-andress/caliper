"""Load and validate experiment configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from caliper.config.errors import (
    ConfigParseError,
    ConfigValidationError,
    format_validation_errors,
    pydantic_errors_to_messages,
)
from caliper.config.schema import ExperimentConfig


def load_config(path: str | Path, *, strict: bool = True) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.
        strict: If True, raise on validation failure; if False, same behavior
            (reserved for future warning-only mode).

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigParseError: If the file is not valid YAML.
        ConfigValidationError: If the config fails schema validation.
    """
    config_path = Path(path).resolve()
    raw = _read_yaml(config_path)

    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        errors = pydantic_errors_to_messages(exc)
        if strict:
            raise ConfigValidationError(config_path, errors) from exc
        raise

    _validate_prompt_paths(config, config_path)
    return config


def validate_config(path: str | Path) -> list[str]:
    """Validate a config file and return a list of error messages.

    Returns an empty list when the config is valid.
    """
    config_path = Path(path).resolve()
    errors: list[str] = []

    if not config_path.exists():
        return [f"config file not found: {config_path}"]

    try:
        raw = _read_yaml(config_path)
    except ConfigParseError as exc:
        return [str(exc)]

    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        errors.extend(pydantic_errors_to_messages(exc))
        return errors

    errors.extend(_check_prompt_paths(config, config_path))
    return errors


def format_config_summary(config: ExperimentConfig) -> str:
    """Return a human-readable summary of an experiment config."""
    axes = config.factorial_axes()
    axis_summary = " × ".join(f"{count} {name}" for name, count in axes.items())
    lines = [
        f"Experiment: {config.experiment_id}",
        f"Description: {config.description or '(none)'}",
        f"Random seed: {config.random_seed}",
        f"Providers: {len(config.providers)}",
        f"Models: {len(config.models)}",
        f"Tasks: {len(config.tasks)}",
        f"Prompt variants: {len(config.prompt_variants) or 1}",
        f"Temperatures: {config.temperatures}",
        f"Runs: {config.number_of_runs}",
        f"Metrics: {', '.join(config.evaluation_metrics)}",
        f"Primary metric: {config.primary_metric or '(default: first evaluation metric)'}",
        f"Factorial: {axis_summary} = {config.total_combinations()} combinations",
        f"Output: {config.output.directory}",
    ]
    return "\n".join(lines)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        msg = f"Failed to parse YAML in {path}: {exc}"
        raise ConfigParseError(msg) from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"Config root must be a mapping, got {type(raw).__name__}"
        raise ConfigValidationError(path, [msg])
    return raw


def _validate_prompt_paths(config: ExperimentConfig, config_path: Path) -> None:
    errors = _check_prompt_paths(config, config_path)
    if errors:
        raise ConfigValidationError(config_path, errors)


def _check_prompt_paths(config: ExperimentConfig, config_path: Path) -> list[str]:
    errors: list[str] = []
    base_dir = config_path.parent
    for prompt in config.prompt_variants:
        if prompt.path is None:
            continue
        resolved = prompt.path if prompt.path.is_absolute() else base_dir / prompt.path
        if not resolved.exists():
            errors.append(
                f"prompt_variants.{prompt.id}.path: file not found: {resolved}"
            )
    return errors
