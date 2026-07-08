"""Generate Paper 1 confirmatory experiment YAML configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from caliper.benchmarks.materialize import select_task_subset
from caliper.benchmarks.prompts import controlled_prompt_templates

DEFAULT_MODELS: list[dict[str, str]] = [
    {"id": "qwen25_coder_7b", "provider": "ollama_local", "model_id": "qwen2.5-coder:7b"},
    {"id": "qwen25_coder_14b", "provider": "ollama_local", "model_id": "qwen2.5-coder:14b"},
    {"id": "qwen25_coder_32b", "provider": "ollama_local", "model_id": "qwen2.5-coder:32b"},
    {"id": "deepseek_coder_v2_lite", "provider": "ollama_local", "model_id": "deepseek-coder-v2:lite"},
    {"id": "llama31_8b", "provider": "ollama_local", "model_id": "llama3.1:8b"},
    {"id": "qwen3_32b", "provider": "ollama_local", "model_id": "qwen3:32b"},
]

DEFAULT_TEMPERATURES = [0.0, 0.2]
DEFAULT_RUNS = 5
DEFAULT_TASK_SUBSET_SIZE = 40


def _task_block(task_id: str, benchmark_task_id: str, dataset_rel: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "domain": "executable_code_generation",
        "dataset": dataset_rel,
        "num_samples": 1,
        "metrics": [
            "pass_at_1",
            "syntax_check",
            "normalized_code_match",
            "execution_latency_ms",
            "token_count",
        ],
        "extra": {
            "filter_task_id": benchmark_task_id,
            "execution": {"timeout_seconds": 5.0, "memory_mb": 512},
        },
    }


def build_confirmatory_config(
    *,
    experiment_id: str,
    benchmark_name: str,
    dataset_path: str,
    description: str,
    task_ids: list[str],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a factorial confirmatory experiment configuration dict."""
    prompts = [
        {"id": prompt.style, "template": prompt.template + "\n\nRespond with exactly one Python code block using triple backticks (` ```python `). Do not include explanations, commentary, or text outside the code block."}
        for prompt in controlled_prompt_templates()
    ]

    tasks = [
        _task_block(
            task_id=f"task-{benchmark_name}-{index:03d}",
            benchmark_task_id=benchmark_task_id,
            dataset_rel=dataset_path,
        )
        for index, benchmark_task_id in enumerate(task_ids, start=1)
    ]

    config: dict[str, Any] = {
        "experiment_id": experiment_id,
        "description": description.strip(),
        "random_seed": 20260404,
        "primary_metric": "pass_at_1",
        "evaluation_metrics": [
            "pass_at_1",
            "syntax_check",
            "normalized_code_match",
            "execution_latency_ms",
            "token_count",
        ],
        "providers": {
            "ollama_local": {
                "provider_type": "ollama",
                "base_url": "http://localhost:11434",
                "timeout_seconds": 300,
            },
        },
        "models": DEFAULT_MODELS,
        "prompt_variants": prompts,
        "tasks": tasks,
        "temperatures": DEFAULT_TEMPERATURES,
        "number_of_runs": DEFAULT_RUNS,
        "decoding": {"max_tokens": 1024, "top_p": 1.0},
        "output": {"directory": f"experiments/{experiment_id}", "format": "parquet"},
        "logging": {"level": "INFO", "log_to_file": True, "log_format": "json"},
        "execution": {"shuffle": True, "parallel_workers": 1},
        "study_metadata": {
            "study_type": "confirmatory",
            "benchmark": benchmark_name,
            "prompt_protocol": "controlled_output_format_v1",
            "pilot_reference": "experiments/paper1_ollama_pilot",
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Paper 1 confirmatory study — {benchmark_name}\n"
            f"# {len(task_ids)} tasks × {len(DEFAULT_MODELS)} models × "
            f"{len(prompts)} prompts × {len(DEFAULT_TEMPERATURES)} temperatures × "
            f"{DEFAULT_RUNS} runs = "
            f"{len(task_ids) * len(DEFAULT_MODELS) * len(prompts) * len(DEFAULT_TEMPERATURES) * DEFAULT_RUNS} cells\n"
        )
        output_path.write_text(
            header + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return config


def write_confirmatory_configs(
    *,
    configs_dir: Path | str = Path("configs/paper1"),
    data_dir: Path | str = Path("data/benchmarks"),
    task_subset_size: int = DEFAULT_TASK_SUBSET_SIZE,
    seed: int = 20260404,
) -> dict[str, Path]:
    """Write HumanEval+ and MBPP confirmatory YAML configs."""
    configs_dir = Path(configs_dir)
    data_dir = Path(data_dir)

    humaneval_dataset = data_dir / "humaneval_plus.jsonl"
    mbpp_dataset = data_dir / "mbpp.jsonl"

    humaneval_ids = select_task_subset(humaneval_dataset, size=task_subset_size, seed=seed)
    mbpp_ids = select_task_subset(mbpp_dataset, size=task_subset_size, seed=seed + 1)

    paths = {
        "humaneval": configs_dir / "confirmatory_humaneval.yaml",
        "mbpp": configs_dir / "confirmatory_mbpp.yaml",
    }

    build_confirmatory_config(
        experiment_id="paper1_confirmatory_humaneval",
        benchmark_name="humaneval_plus",
        dataset_path=str(humaneval_dataset.as_posix()),
        description=(
            "Paper 1 confirmatory study on HumanEval+ with controlled prompts, "
            "sandboxed pass@1 evaluation, and factorial local-model design."
        ),
        task_ids=humaneval_ids,
        output_path=paths["humaneval"],
    )
    build_confirmatory_config(
        experiment_id="paper1_confirmatory_mbpp",
        benchmark_name="mbpp",
        dataset_path=str(mbpp_dataset.as_posix()),
        description=(
            "Paper 1 confirmatory study on MBPP with controlled prompts, "
            "sandboxed pass@1 evaluation, and factorial local-model design."
        ),
        task_ids=mbpp_ids,
        output_path=paths["mbpp"],
    )
    return paths


def expected_cell_count(task_count: int) -> int:
    """Return expected factorial cell count for default confirmatory design."""
    prompt_count = len(controlled_prompt_templates())
    return (
        task_count
        * len(DEFAULT_MODELS)
        * prompt_count
        * len(DEFAULT_TEMPERATURES)
        * DEFAULT_RUNS
    )
