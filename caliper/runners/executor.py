"""Execute factorial experiment cells."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from caliper.config.schema import (
    ExperimentCombination,
    ExperimentConfig,
    ModelConfig,
    PromptVariantConfig,
    ProviderConfig,
    TaskConfig,
)
from caliper.models import create_provider
from caliper.models.base import BaseModelProvider
from caliper.models.retry import ProviderRuntimeConfig, RetryPolicy
from caliper.models.types import ModelRequest
from caliper.prompts.loader import load_prompt
from caliper.runners.cells import expand_cells, make_cell_id
from caliper.runners.results import ExperimentResultRecord, ResultWriter
from caliper.tasks import create_task
from caliper.tasks.base import BaseTask
from caliper.tasks.schema import TaskDomain, TaskMetadata

logger = structlog.get_logger(__name__)

SUPPORTED_PROVIDER_TYPES = frozenset(
    {"mock", "random", "openai", "anthropic", "gemini", "google", "local"}
)

API_PROVIDER_TYPES = frozenset({"openai", "anthropic", "gemini", "google"})

DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}


def resolve_dataset_path(task: TaskConfig, config_dir: Path) -> Path:
    """Resolve a task dataset path relative to the config file."""
    dataset_path = Path(task.dataset)
    if dataset_path.is_absolute():
        return dataset_path
    candidates = [
        config_dir / dataset_path,
        Path.cwd() / dataset_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (config_dir / dataset_path).resolve()


def resolve_task_domain(task: TaskConfig) -> TaskDomain:
    if task.domain is not None:
        return task.domain
    msg = f"task '{task.id}' must specify a domain for factorial execution"
    raise ValueError(msg)


def build_provider(
    config: ExperimentConfig,
    model: ModelConfig,
) -> BaseModelProvider:
    provider_cfg = config.providers[model.provider]
    if provider_cfg.type not in SUPPORTED_PROVIDER_TYPES:
        msg = (
            f"provider type '{provider_cfg.type}' is not supported yet; "
            f"allowed: {', '.join(sorted(SUPPORTED_PROVIDER_TYPES))}"
        )
        raise ValueError(msg)

    decoding = model.decoding or config.decoding
    extra = dict(provider_cfg.extra)
    timeout_seconds = float(extra.pop("timeout_seconds", 60.0))
    max_retries = int(extra.pop("max_retries", 3))
    runtime = ProviderRuntimeConfig(
        timeout_seconds=timeout_seconds,
        retry=RetryPolicy(
            max_retries=max_retries,
            initial_backoff_seconds=float(extra.pop("initial_backoff_seconds", 0.5)),
            backoff_multiplier=float(extra.pop("backoff_multiplier", 2.0)),
            max_backoff_seconds=float(extra.pop("max_backoff_seconds", 30.0)),
        ),
    )
    _ = decoding  # used when constructing ModelRequest in execute_cell

    provider_kwargs: dict[str, Any] = {}
    if provider_cfg.type in API_PROVIDER_TYPES:
        provider_kwargs["api_key_env"] = (
            provider_cfg.api_key_env or DEFAULT_API_KEY_ENVS[provider_cfg.type]
        )
        if provider_cfg.base_url:
            provider_kwargs["base_url"] = provider_cfg.base_url
        provider_kwargs.update(extra)
    elif provider_cfg.type in {"mock", "random"}:
        provider_kwargs["simulated_latency_ms"] = float(extra.pop("simulated_latency_ms", 0.0))
        provider_kwargs.update(extra)
    elif provider_cfg.type == "local":
        model_path = model.extra.get("model_path") or extra.get("model_path")
        if model_path is not None:
            provider_kwargs["model_path"] = model_path
        provider_kwargs.update(extra)
        provider_kwargs.update(
            {k: v for k, v in model.extra.items() if k not in provider_kwargs}
        )

    return create_provider(
        provider_cfg.type,
        model_name=model.model_id,
        provider_name=model.provider,
        runtime=runtime,
        **provider_kwargs,
    )


def build_task(
    config: ExperimentConfig,
    task_cfg: TaskConfig,
    config_dir: Path,
) -> BaseTask:
    domain = resolve_task_domain(task_cfg)
    dataset_path = resolve_dataset_path(task_cfg, config_dir)
    task_config: dict[str, Any] = {"num_samples": task_cfg.num_samples}
    task_config.update(task_cfg.extra)
    return create_task(domain, task_cfg.id, dataset_path, **task_config)


def render_task_prompt(
    prompt_cfg: PromptVariantConfig,
    example: TaskMetadata,
    config_dir: Path,
) -> str:
    if prompt_cfg.path is not None:
        path = prompt_cfg.path if prompt_cfg.path.is_absolute() else config_dir / prompt_cfg.path
        prompt_cfg = prompt_cfg.model_copy(update={"path": path})
    template = load_prompt(prompt_cfg)
    variables = {
        "input": example.input,
        "question": example.input,
        **prompt_cfg.variables,
    }
    return template.render(**variables)


def primary_metric_name(config: ExperimentConfig, task_id: str) -> str:
    return config.metrics_for_task(task_id)[0]


def extract_primary_score(scores: dict[str, float], metric: str) -> float:
    if metric in scores:
        return scores[metric]
    if metric == "accuracy" and "exact_match" in scores:
        return scores["exact_match"]
    if scores:
        return next(iter(scores.values()))
    return 0.0


def execute_cell(
    *,
    config: ExperimentConfig,
    cell: ExperimentCombination,
    run_id: str,
    config_dir: Path,
    providers: dict[str, BaseModelProvider],
    tasks: dict[str, BaseTask],
    prompts: dict[str, PromptVariantConfig],
) -> ExperimentResultRecord:
    """Execute a single factorial cell and return a structured result."""
    cell_id = make_cell_id(config, cell)
    model_cfg = next(m for m in config.models if m.id == cell.model_id)
    task_cfg = next(t for t in config.tasks if t.id == cell.task_id)
    prompt_cfg = prompts[cell.prompt_variant_id]
    provider = providers[cell.model_id]
    task = tasks[cell.task_id]
    metric = primary_metric_name(config, cell.task_id)

    seed = config.random_seed + cell.run_index
    decoding = model_cfg.decoding or config.decoding

    examples = task.load_examples()
    predictions: list[str] = []
    score_accumulator: dict[str, list[float]] = {}
    total_latency = 0.0

    for example in examples:
        prompt_text = render_task_prompt(prompt_cfg, example, config_dir)
        request = ModelRequest(
            prompt=prompt_text,
            prompt_id=cell.prompt_variant_id,
            task_id=cell.task_id,
            run_id=f"{run_id}:{cell_id}",
            temperature=cell.temperature,
            seed=seed,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            max_tokens=decoding.max_tokens,
            stop=decoding.stop,
            metadata={
                "expected_output": example.expected_output,
                "language": example.language,
            },
        )
        response = provider.generate(request)
        predictions.append(response.text)
        total_latency += response.latency_ms

        example_scores = task.score(example, response.text)
        for name, value in example_scores.items():
            score_accumulator.setdefault(name, []).append(value)

    averaged_scores = {
        name: sum(values) / len(values) for name, values in score_accumulator.items()
    }
    primary = extract_primary_score(averaged_scores, metric)

    provider_cfg = config.providers[model_cfg.provider]
    return ExperimentResultRecord(
        cell_id=cell_id,
        experiment_id=config.experiment_id,
        run_id=run_id,
        run_index=cell.run_index,
        model_id=cell.model_id,
        provider_name=model_cfg.provider,
        provider_type=provider_cfg.type,
        task_id=cell.task_id,
        prompt_variant_id=cell.prompt_variant_id,
        temperature=cell.temperature,
        seed=seed,
        metric=metric,
        score=primary,
        scores=averaged_scores,
        prediction=predictions[0] if predictions else "",
        num_examples=len(examples),
        latency_ms=total_latency,
        status="completed",
        metadata={
            "model_id_config": model_cfg.model_id,
            "dataset": task_cfg.dataset,
            "task_domain": resolve_task_domain(task_cfg),
            "num_predictions": len(predictions),
        },
    )


def execute_cell_safe(
    *,
    config: ExperimentConfig,
    cell: ExperimentCombination,
    run_id: str,
    config_dir: Path,
    providers: dict[str, BaseModelProvider],
    tasks: dict[str, BaseTask],
    prompts: dict[str, PromptVariantConfig],
) -> ExperimentResultRecord:
    """Execute a cell, returning a failed record instead of raising."""
    cell_id = make_cell_id(config, cell)
    try:
        return execute_cell(
            config=config,
            cell=cell,
            run_id=run_id,
            config_dir=config_dir,
            providers=providers,
            tasks=tasks,
            prompts=prompts,
        )
    except Exception as exc:
        logger.exception("cell.failed", cell_id=cell_id, error=str(exc))
        model_cfg = next(m for m in config.models if m.id == cell.model_id)
        provider_cfg = config.providers[model_cfg.provider]
        return ExperimentResultRecord(
            cell_id=cell_id,
            experiment_id=config.experiment_id,
            run_id=run_id,
            run_index=cell.run_index,
            model_id=cell.model_id,
            provider_name=model_cfg.provider,
            provider_type=provider_cfg.type,
            task_id=cell.task_id,
            prompt_variant_id=cell.prompt_variant_id,
            temperature=cell.temperature,
            seed=config.random_seed + cell.run_index,
            metric=primary_metric_name(config, cell.task_id),
            score=0.0,
            status="failed",
            error=str(exc),
            metadata={"exception_type": type(exc).__name__},
        )
