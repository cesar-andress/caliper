# CALIPER Architecture

## Overview

CALIPER is a CLI-first Python framework for reproducible LLM evaluation experiments.
It supports two research papers:

1. **Paper 1** — Variance decomposition and statistical power in LLM evaluation
2. **Paper 2** — Ranking fragility in LLM benchmarks

## Module Map

```
caliper/
├── cli.py              # Click CLI entry point
├── config/             # YAML config loading & Pydantic schemas
├── models/             # Model provider ABC + registry
├── tasks/              # Evaluation task ABC
├── prompts/            # Prompt template loading & rendering
├── runners/            # Experiment orchestration
├── evaluation/         # Metric computation
├── statistics/         # Variance decomposition & power analysis (Paper 1)
├── ranking/            # Ranking fragility analysis (Paper 2)
├── storage/            # Result persistence (Parquet/JSONL/CSV)
└── utils/              # Logging and shared helpers
```

## Data Flow

```
YAML config
    │
    ▼
ExperimentRunner
    │
    ├── setup logging (structlog → stdout + file)
    ├── create RunManifest
    │
    ▼
For each (run × model × prompt × task):
    │
    ├── Task.load_examples()
    ├── PromptTemplate.render()
    ├── ModelProvider.generate()
    ├── Task.score()
    └── ResultStore.save()
    │
    ▼
Post-hoc analysis
    ├── caliper analyze variance   (Paper 1)
    └── caliper analyze fragility  (Paper 2)
```

## Configuration

Experiments are fully defined in YAML. See `configs/examples/` for templates.
Key sections:

| Section   | Purpose                                      |
|-----------|----------------------------------------------|
| `models`  | Model name, provider, decoding parameters    |
| `prompts` | Prompt templates (inline or file-based)      |
| `tasks`   | Dataset, split, metrics, sample limits       |
| `run`     | Replication count, shuffling, parallelism    |
| `storage` | Output directory and format                  |
| `logging` | Log level, format, file output               |

## Output Layout

```
outputs/
└── {experiment_name}/
    └── {run_id}/
        ├── run.log          # Structured JSON logs
        ├── manifest.json    # Run metadata
        └── results.parquet  # Long-format scores
```

## Results Schema (planned)

Each row in the results table:

| Column       | Type   | Description                    |
|--------------|--------|--------------------------------|
| run_id       | str    | Unique run identifier          |
| run_index    | int    | Replication index              |
| model        | str    | Model name from config         |
| task         | str    | Task identifier                |
| prompt_id    | str    | Prompt variant                 |
| example_id   | str    | Dataset example ID             |
| score        | float  | Metric score (0–1)             |
| metric       | str    | Metric name                    |
| prediction   | str    | Raw model output (optional)    |
| latency_ms   | float  | Generation latency (optional)  |

## Extension Points

| Component       | How to extend                                    |
|-----------------|--------------------------------------------------|
| Model providers | Subclass `ModelProvider`, `@register_provider`   |
| Tasks           | Subclass `Task`, implement `load_examples/score` |
| Metrics         | Add cases to `compute_metric()`                  |
| Storage         | Subclass `ResultStore`                           |
| Statistics      | Add functions in `statistics/`                   |

## Design Principles

- **CLI-first**: Every operation accessible via `caliper` commands
- **Config-driven**: No hardcoded experiment parameters
- **Reproducible**: Seeded runs, logged manifests, versioned configs
- **Modular**: Papers share infrastructure but have dedicated analysis modules
- **Analysis-ready**: Parquet output for pandas/scipy/R downstream
