"""Build minimal pre-flight experiment configs from confirmatory references."""

from __future__ import annotations

from pathlib import Path

from caliper.benchmarks.materialize import select_task_subset
from caliper.benchmarks.prompts import controlled_prompt_templates
from caliper.config.loader import load_config
from caliper.config.schema import ExperimentConfig, ModelConfig, PromptVariantConfig, TaskConfig

BENCHMARK_ALIASES = {
    "humaneval": "humaneval_plus",
    "humaneval_plus": "humaneval_plus",
    "mbpp": "mbpp",
}

REFERENCE_CONFIGS = {
    "humaneval_plus": Path("configs/paper1/confirmatory_humaneval.yaml"),
    "mbpp": Path("configs/paper1/confirmatory_mbpp.yaml"),
}

DATASET_PATHS = {
    "humaneval_plus": Path("data/benchmarks/humaneval_plus.jsonl"),
    "mbpp": Path("data/benchmarks/mbpp.jsonl"),
}

DEFAULT_MODEL = "qwen25_coder_7b"
DEFAULT_PROMPT = "minimal"
PROMPT_PROTOCOL_VERSION = "controlled_output_format_v1"


def resolve_benchmark(name: str) -> str:
    key = name.strip().lower()
    if key not in BENCHMARK_ALIASES:
        allowed = ", ".join(sorted(BENCHMARK_ALIASES))
        msg = f"Unknown benchmark '{name}'. Allowed: {allowed}"
        raise ValueError(msg)
    return BENCHMARK_ALIASES[key]


def reference_config_path(benchmark: str) -> Path:
    resolved = resolve_benchmark(benchmark)
    return REFERENCE_CONFIGS[resolved]


def dataset_path(benchmark: str) -> Path:
    resolved = resolve_benchmark(benchmark)
    return DATASET_PATHS[resolved]


def build_preflight_config(
    benchmark: str,
    *,
    model_id: str = DEFAULT_MODEL,
    prompt_id: str = DEFAULT_PROMPT,
    temperature: float = 0.0,
    number_of_runs: int = 1,
    num_tasks: int = 3,
    seed: int = 20260404,
) -> tuple[ExperimentConfig, Path, list[str]]:
    """Build a minimal end-to-end config without modifying confirmatory YAML files."""
    resolved = resolve_benchmark(benchmark)
    reference_path = REFERENCE_CONFIGS[resolved]
    reference = load_config(reference_path)
    dataset = dataset_path(benchmark)

    task_ids = select_task_subset(dataset, size=num_tasks, seed=seed)
    dataset_rel = str(dataset.as_posix())

    model = next((m for m in reference.models if m.id == model_id), None)
    if model is None:
        msg = f"Model '{model_id}' not found in {reference_path}"
        raise ValueError(msg)

    prompt = next((p for p in reference.prompt_variants if p.id == prompt_id), None)
    if prompt is None:
        available = [p.id for p in reference.prompt_variants]
        msg = f"Prompt '{prompt_id}' not found. Available: {available}"
        raise ValueError(msg)

    tasks: list[TaskConfig] = []
    for index, benchmark_task_id in enumerate(task_ids, start=1):
        ref_task = reference.tasks[0]
        tasks.append(
            TaskConfig(
                id=f"preflight-{resolved}-{index:03d}",
                domain=ref_task.domain,
                dataset=dataset_rel,
                num_samples=1,
                metrics=list(ref_task.metrics or reference.evaluation_metrics),
                extra={"filter_task_id": benchmark_task_id, **ref_task.extra},
            )
        )

    config = ExperimentConfig(
        experiment_id=f"preflight_validate_{resolved}",
        description=(
            f"Pre-flight validation run for Paper 1 confirmatory study ({resolved}). "
            "Not part of the confirmatory evidence."
        ),
        random_seed=seed,
        providers=reference.providers,
        models=[ModelConfig(id=model.id, provider=model.provider, model_id=model.model_id, extra=model.extra)],
        tasks=tasks,
        prompt_variants=[PromptVariantConfig(id=prompt.id, template=prompt.template, path=prompt.path)],
        temperatures=[temperature],
        number_of_runs=number_of_runs,
        decoding=reference.decoding,
        evaluation_metrics=list(reference.evaluation_metrics),
        primary_metric=reference.primary_metric,
        output=reference.output.model_copy(
            update={"directory": Path(f"experiments/preflight_validate_{resolved}")}
        ),
        logging=reference.logging,
        execution=reference.execution.model_copy(update={"shuffle": False}),
        metadata={
            "study_type": "preflight_validation",
            "reference_config": str(reference_path),
            "prompt_protocol": PROMPT_PROTOCOL_VERSION,
            "benchmark": resolved,
        },
    )
    return config, reference_path, task_ids


def verify_prompt_family() -> list[str]:
    """Return errors if controlled prompts do not share the required output suffix."""
    from caliper.benchmarks.prompts import CONTROLLED_OUTPUT_SUFFIX

    errors: list[str] = []
    for prompt in controlled_prompt_templates():
        rendered = prompt.render("def example():\n    pass\n")
        if CONTROLLED_OUTPUT_SUFFIX.strip() not in rendered:
            errors.append(f"Prompt '{prompt.style}' missing controlled output suffix")
    return errors
