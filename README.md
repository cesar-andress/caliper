# CALIPER

**C**omparative **A**nalysis of **L**LM **I**nference **P**erturbation, **E**valuation, and **R**anking

CALIPER is a reproducible research artifact for designing, executing, and analyzing factorial LLM evaluation experiments with explicit attention to **variance**, **statistical power**, and **ranking stability**.

It is intended to support empirical studies that treat benchmark scores as measurements subject to sampling and protocol variation—not as fixed constants. The software is CLI-first, configuration-driven, and modular; it does not prescribe a single benchmark or model suite.

**Status:** Alpha research software (v0.1.0). APIs and analysis methods may change. See limitations in [`analyses/paper1/README.md`](analyses/paper1/README.md).

---

## Why variance-aware evaluation?

Standard LLM benchmarks often report aggregate scores or leaderboard ranks without describing how much of the observed difference is attributable to models versus tasks, prompts, stochastic runs, or decoding settings. That makes it difficult to assess:

- whether a reported gap between models is stable under resampling;
- how many tasks, prompts, or runs are needed for a comparison to be informative;
- whether a leaderboard ordering reflects durable capability differences or protocol noise.

CALIPER encodes evaluation as a **factorial experiment**: models, tasks, prompt variants, temperatures, and run indices are crossed explicitly, logged structurally, and passed to analysis modules that estimate variance components, generalizability coefficients, power under alternative designs, and rank stability under bootstrap resampling.

This does not replace domain-specific benchmark design or human evaluation. It provides infrastructure for reporting **how** results were produced and **how sensitive** conclusions are to legitimate protocol variation.

---

## Relationship to Paper 1 and Paper 2

CALIPER implements the pipelines described in two companion manuscripts (LaTeX skeletons in the parent repository):

| Paper | Focus | Artifact modules |
|-------|--------|------------------|
| **Paper 1** | Variance decomposition, generalizability (G-/D-study), and statistical power in factorial LLM evaluation | `caliper/statistics/`, `caliper analyze variance`, `caliper analyze power`, [`analyses/paper1/`](analyses/paper1/) |
| **Paper 2** | Ranking fragility: stability of model orderings under task, prompt, and run resampling | `caliper/ranking/`, `caliper ranking-fragility` |

The artifact is the **executable counterpart** to those papers: YAML configs define designs; runners produce result matrices; analysis commands implement the estimators discussed in the manuscripts. Manuscript text and empirical claims are maintained separately; this repository provides tools and examples, not published findings.

---

## Installation

**Requirements:** Python 3.11+, `pip`, optional virtual environment.

```bash
cd caliper
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + development tools
```

Optional dependency groups (install only what you need):

```bash
pip install -e ".[api]"           # OpenAI, Anthropic, Gemini SDKs
pip install -e ".[local]"         # HuggingFace transformers + PyTorch
pip install -e ".[local-llama-cpp]"   # GGUF via llama.cpp
pip install -e ".[local-vllm]"     # vLLM (Linux + CUDA)
pip install -e ".[local-all]"     # all local backends + NVML
```

Or via Makefile:

```bash
make install-dev
```

Copy environment template for API keys and local paths:

```bash
cp .env.example .env
# Edit .env — keys are read from the environment, never committed
```

---

## Quickstart

Validate configuration, inspect the factorial plan, and dry-run without model calls:

```bash
caliper validate --config configs/examples/basic_experiment.yaml
caliper plan --config configs/examples/basic_experiment.yaml
caliper run configs/examples/basic_experiment.yaml --dry-run
```

Run a small mock factorial experiment (no API keys):

```bash
caliper run configs/examples/example_factorial.yaml
```

Post-hoc evaluation and analysis (after a run completes):

```bash
caliper evaluate outputs/example_factorial/<run_id>/results.parquet \
  --config configs/examples/example_factorial.yaml

caliper analyze variance -r outputs/example_factorial/<run_id>/evaluations.parquet
caliper ranking-fragility outputs/example_factorial/<run_id>/evaluations.parquet \
  --metric exact_match
```

Run the test suite:

```bash
make test
```

Further detail: [`docs/getting-started.md`](docs/getting-started.md), [`docs/architecture.md`](docs/architecture.md).

---

## Running mock experiments

Mock and random providers require no external services. They are suitable for pipeline testing, CI, and dry statistical workflows.

| Provider | Behavior |
|----------|----------|
| `mock` | Deterministic outputs from prompt, seed, and model id |
| `random` | Stochastic outputs for testing variability |

Example config: [`configs/examples/example_factorial.yaml`](configs/examples/example_factorial.yaml) (32 factorial cells with sample code tasks).

```bash
caliper run configs/examples/example_factorial.yaml
# Resume after interruption:
caliper run configs/examples/example_factorial.yaml \
  --resume outputs/example_factorial/<run_id>
```

Use `--dry-run` to validate and plan without writing results or calling providers.

---

## Running real API experiments

CALIPER supports `openai`, `anthropic`, and `gemini` (alias `google`) providers. Model identifiers come from YAML (`model_id`); they are not hardcoded in the library.

1. Install API dependencies: `pip install -e ".[api]"`
2. Set keys in `.env` or the shell:

   ```bash
   export OPENAI_API_KEY=...
   export ANTHROPIC_API_KEY=...
   export GEMINI_API_KEY=...
   ```

3. Define providers and models in experiment YAML:

   ```yaml
   providers:
     openai-main:
       type: openai
       api_key_env: OPENAI_API_KEY
       extra:
         max_retries: 3
         timeout_seconds: 60

   models:
     - id: gpt-eval
       provider: openai-main
       model_id: <your-model-id>
   ```

4. Run: `caliper run your_experiment.yaml`

Providers support retries with exponential backoff, timeouts, structured response metadata, optional dry-run mode (`extra.dry_run: true` or `CALIPER_PROVIDER_DRY_RUN=1`), and optional cost estimation hooks when pricing is supplied in config. Integration tests are marked `@pytest.mark.integration` and skipped unless keys are present.

---

## Running local models

Open-weight models can be run via the `local` provider (transformers, llama.cpp, or vLLM). GPU dependencies are optional; the core package installs without them.

1. Install a backend: see [`docs/local-models.md`](docs/local-models.md)
2. Set `LOCAL_MODEL_PATH` or `providers.<name>.extra.model_path` in YAML
3. Example skeleton: [`configs/examples/local_model.yaml`](configs/examples/local_model.yaml)

```yaml
providers:
  local-gpu:
    type: local
    extra:
      model_path: /path/to/model
      backend: transformers    # or llama_cpp, vllm
      device: cuda:0
      nvml: true               # optional power logging (requires pynvml)
```

```bash
pip install -e ".[local]"
caliper run configs/examples/local_model.yaml   # after configuring model_path
```

Local inference logs GPU metadata, latency, quantization settings, and optional NVML energy estimates in `raw_metadata`.

---

## Output structure

Each experiment run writes to:

```text
outputs/<experiment_id>/<run_id>/
├── manifest.json           # run metadata, cell counts, timestamps
├── results.jsonl           # append-only cell records (resume-safe)
├── results.parquet         # finalized experiment table
├── evaluations.jsonl     # after caliper evaluate (if run)
├── evaluations.parquet
└── logs/                   # structured logs (when enabled)
```

A **cell** is one combination of model × task × prompt variant × temperature × run index. Each record includes scores, status, seeds, latency, and provider metadata.

Analysis outputs (variance, ranking fragility) default to `reports/` or paths given on the CLI, e.g.:

```text
reports/ranking_fragility/
├── ranking_fragility_summary.csv
├── bootstrap_samples.parquet
├── rank_probabilities.csv
├── pairwise_reversals.csv
└── plots/
```

Standard analysis columns include `model`, `task_id`, `prompt_id`, `run_id`, `temperature`, `metric_name`, and `metric_value` (see `caliper statistics prepare`).

---

## Reproducibility principles

CALIPER is built around the following conventions:

1. **Declarative configs** — Experiments are fully specified in version-controlled YAML (providers, models, tasks, prompts, temperatures, run counts, seeds).
2. **Explicit factorial designs** — All crossed factors are enumerated before execution (`caliper plan`).
3. **Seeded execution** — Base `random_seed` plus deterministic per-cell seed derivation.
4. **Structured logging** — JSON or console logs via structlog; optional log files per run.
5. **Resumable runs** — Completed cells are skipped on `--resume`.
6. **Analysis separation** — Raw runs, metric evaluation, and statistical analysis are distinct stages with stable table schemas.
7. **No hidden defaults for models** — Provider types and model ids come from config, not library constants.

We recommend archiving the config file, git commit hash, dependency versions, and output manifest alongside any reported result.

---

## CLI reference

| Command | Description |
|---------|-------------|
| `caliper validate -c CONFIG` | Validate YAML schema |
| `caliper plan -c CONFIG` | List factorial combinations |
| `caliper run CONFIG [--dry-run] [--resume DIR]` | Execute experiment |
| `caliper evaluate RESULTS -c CONFIG` | Apply metrics to saved results |
| `caliper analyze variance -r FILE` | Variance decomposition (Paper 1) |
| `caliper analyze power -r FILE` | Monte Carlo power simulation (Paper 1) |
| `caliper ranking-fragility FILE -m METRIC` | Bootstrap ranking fragility (Paper 2) |

Positional config form is also supported: `caliper run configs/examples/basic_experiment.yaml`.

---

## Project layout

```text
caliper/                    # Python package
├── caliper/
│   ├── config/             # YAML schemas and loader
│   ├── models/             # Providers (mock, API, local)
│   ├── tasks/              # Code-gen, repair, summarization tasks
│   ├── runners/            # Factorial experiment runner
│   ├── evaluation/         # Post-hoc metrics
│   ├── statistics/         # Paper 1 analyses
│   ├── ranking/            # Paper 2 analyses
│   └── storage/            # Parquet / JSONL I/O
├── configs/examples/       # Example experiment YAML
├── data/examples/          # Sample task datasets (JSONL)
├── analyses/paper1/        # Standalone analysis scripts
├── docs/                   # Architecture and usage guides
└── tests/
```

---

## Development

```bash
make lint       # ruff
make format     # ruff format + fix
make typecheck  # mypy
make test-cov   # pytest with coverage
```

---

## Citation

If you use CALIPER in academic work, please cite the accompanying papers (BibTeX to be finalized upon publication):

```bibtex
@misc{caliper2026artifact,
  author       = {{CALIPER Team}},
  title        = {{CALIPER}: Reproducible Factorial Evaluation, Variance Analysis,
                  and Ranking Fragility for Large Language Models},
  year         = {2026},
  howpublished = {Software artifact},
  note         = {https://github.com/PLACEHOLDER/caliper --- URL and version TBD}
}

@article{caliper2026paper1,
  author  = {Author, A. and Author, B.},
  title   = {How Much Variance Is Hidden in {LLM} Evaluation?
             A Generalizability Theory Study of Model, Prompt, Task, Run, and Decoding Effects},
  journal = {Information Processing \& Management},
  year    = {2026},
  note    = {In preparation}
}

@article{caliper2026paper2,
  author  = {Author, A. and Author, B.},
  title   = {Ranking Fragility in {LLM} Benchmarks:
             When Model Leaderboards Collapse Under Resampling},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {In preparation}
}
```

Replace placeholders with published bibliographic data when available.

---

## License

Intended distribution: **MIT License** (see [`pyproject.toml`](pyproject.toml)). A `LICENSE` file will be added before public release. Third-party model APIs and weights are subject to their respective terms.

---

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Local models](docs/local-models.md)
- [Development roadmap](docs/roadmap.md) and [issue backlog](docs/issues.md)
- [Paper 1 analysis scripts](analyses/paper1/README.md)
