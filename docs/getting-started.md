# Getting Started

## Prerequisites

- Python 3.11+
- `make` (optional, for convenience targets)

## Installation

```bash
cd caliper
make install-dev
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys (not needed for mock/dry-run)
```

## Quick Start

### 1. Validate a config

```bash
caliper validate --config configs/examples/basic_experiment.yaml
```

### 2. Preview the experiment plan

```bash
caliper plan --config configs/examples/basic_experiment.yaml
```

### 3. Dry-run an experiment

```bash
caliper run --config configs/examples/basic_experiment.yaml --dry-run
```

### 4. Run tests

```bash
make test
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `caliper run -c CONFIG` | Execute an experiment |
| `caliper run -c CONFIG --dry-run` | Validate and plan without executing |
| `caliper plan -c CONFIG` | List all model × prompt × task × run combos |
| `caliper validate -c CONFIG` | Validate a YAML config file |
| `caliper analyze variance -r RESULTS` | Variance decomposition (Paper 1) |
| `caliper analyze fragility -r RESULTS` | Ranking fragility analysis (Paper 2) |

## Writing a Custom Experiment

Create a YAML file following the schema in `configs/examples/basic_experiment.yaml`.
Key fields:

```yaml
name: my_experiment
seed: 42
models:
  - name: my-model
    provider: openai        # or anthropic, local, mock
    model_id: gpt-4o
    decoding:
      temperature: 0.0
tasks:
  - id: my-task
    dataset: my_dataset
    metrics: [accuracy]
run:
  num_runs: 5
```

## Local Models (GPU)

To run open-weight models on a local GPU (e.g. RTX 4090), see [local-models.md](local-models.md).
Install optional backends with `pip install -e ".[local]"` and configure `type: local` in YAML.

## Reproducing Paper 1 analyses

The frozen HumanEval+ statistical dataset ships under `artifacts/paper1/`:

```bash
python artifacts/paper1/scripts/verify_frozen_dataset.py
python artifacts/paper1/scripts/reproduce_paper1_core_tables.py
```

Details: [`artifacts/paper1/README.md`](../artifacts/paper1/README.md).

## Development

```bash
make lint       # ruff linter
make format     # auto-format
make typecheck  # mypy
make test-cov   # tests with coverage
```

## Project Structure

See [architecture.md](architecture.md) for the full module map and data flow.

## Citation and archival DOI

The CALIPER software artifact is archived on Zenodo
([DOI: https://doi.org/10.5281/zenodo.21780089](https://doi.org/10.5281/zenodo.21780089)).
Prefer that DOI for archival citation; GitHub remains the development repository.
See [`CITATION.cff`](../CITATION.cff) and the README citation block.
