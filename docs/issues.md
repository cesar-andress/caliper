# CALIPER GitHub Issues Backlog

Copy each issue block below into GitHub Issues. Use labels: `milestone:M0` … `milestone:M9`, `priority:P0`, `effort:S|M|L|XL`, and `status:done` where noted.

**Repository:** `caliper/` (Python package root)

---

## M0: Repository scaffold

---

### M0-01 · Initialize pyproject.toml and package layout

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** S · **Dependencies:** —

**Objective**  
Create an installable Python 3.11+ package with hatchling build, core dependencies, and `caliper/` namespace.

**Acceptance criteria**
- [ ] `pyproject.toml` defines `caliper` entry point, ruff, pytest, mypy config
- [ ] Package imports as `import caliper` after `pip install -e .`
- [ ] `caliper/__init__.py` exports `__version__`
- [ ] `.gitignore` excludes `.venv`, outputs, caches, `.env`

---

### M0-02 · Add Makefile and .env.example

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** S · **Dependencies:** M0-01

**Objective**  
Provide standard dev commands and document environment variables without committing secrets.

**Acceptance criteria**
- [ ] `make install-dev`, `make test`, `make lint` targets work
- [ ] `.env.example` lists API keys, `LOCAL_MODEL_PATH`, CALIPER defaults
- [ ] README references Makefile targets

---

### M0-03 · Implement Click CLI skeleton

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** S · **Dependencies:** M0-01

**Objective**  
Expose `caliper` command group with version and placeholder subcommands.

**Acceptance criteria**
- [ ] `caliper --version` prints version
- [ ] `caliper --help` lists command groups
- [ ] Entry point registered in `pyproject.toml`

---

### M0-04 · Add structlog logging utilities

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** S · **Dependencies:** M0-01

**Objective**  
Configure structured logging for experiments (JSON and console formats).

**Acceptance criteria**
- [ ] `caliper/utils/logging.py` exposes `setup_logging(level, log_format, log_dir)`
- [ ] Experiment runner calls `setup_logging` on start
- [ ] Log level configurable from YAML `logging:` block

---

### M0-05 · Create architecture and getting-started docs

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M0-03

**Objective**  
Document module map and first-run instructions for contributors.

**Acceptance criteria**
- [ ] `docs/architecture.md` describes data flow and extension points
- [ ] `docs/getting-started.md` covers install, validate, dry-run
- [ ] README links to docs

---

### M0-06 · Initial pytest suite and conftest

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M0-01

**Objective**  
Establish test conventions and shared fixtures.

**Acceptance criteria**
- [ ] `tests/conftest.py` with `sample_config` fixture
- [ ] `pytest` runs with `pythonpath = ["."]` in pyproject
- [ ] At least smoke tests for config and CLI

---

## M1: Config and task system

---

### M1-01 · Pydantic schemas for experiment config

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M0-01

**Objective**  
Define typed YAML schema for providers, models, tasks, prompts, decoding, output.

**Acceptance criteria**
- [ ] `caliper/config/schema.py` with `ExperimentConfig`, `ProviderConfig`, `ModelConfig`, `TaskConfig`
- [ ] Validators for experiment_id pattern, non-empty models/tasks
- [ ] `ProviderType` includes mock, random, openai, anthropic, gemini, local

---

### M1-02 · Config loader with validation errors

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M1-01

**Objective**  
Load YAML configs with actionable validation messages.

**Acceptance criteria**
- [ ] `load_config(path)` returns `ExperimentConfig`
- [ ] `validate_config(path)` returns list of errors without raising
- [ ] `ConfigValidationError` includes file path and field context
- [ ] `caliper validate -c CONFIG` CLI command

---

### M1-03 · Factorial combination expansion

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M1-01

**Objective**  
Expand models × tasks × prompts × temperatures × runs into enumerated cells.

**Acceptance criteria**
- [ ] `factorial_axes()`, `total_combinations()`, `iter_combinations()` on config
- [ ] `caliper plan -c CONFIG` prints all combinations
- [ ] Tests for combination count and uniqueness

---

### M1-04 · BaseTask ABC and task registry

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M0-01

**Objective**  
Pluggable task interface with JSONL loading and scoring hook.

**Acceptance criteria**
- [ ] `BaseTask` with `load_examples()`, `score(example, prediction)`
- [ ] `create_task(domain, id, dataset_path, **kwargs)` factory
- [ ] `TaskMetadata` schema for input/expected fields

---

### M1-05 · Code-gen, bug-repair, summarization tasks

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** L · **Dependencies:** M1-04

**Objective**  
Implement three CALIPER task domains with example datasets.

**Acceptance criteria**
- [ ] `CodeGenerationTask`, `BugRepairTask`, `CodeSummarizationTask` registered
- [ ] `data/examples/*.jsonl` sample files load without error
- [ ] `tests/test_tasks.py` covers loading and scoring

---

### M1-06 · Prompt template loader and renderer

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M1-01

**Objective**  
Support inline templates and file paths with `{variable}` substitution.

**Acceptance criteria**
- [ ] `load_prompt(PromptVariantConfig)` returns renderable template
- [ ] `render(**variables)` substitutes task fields
- [ ] Validation requires `template` or `path`

---

### M1-07 · Example configs and config tests

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M1-02, M1-03

**Objective**  
Ship working example YAML files and comprehensive config tests.

**Acceptance criteria**
- [ ] `configs/examples/basic_experiment.yaml` validates
- [ ] `tests/test_config.py` covers invalid configs and factorial math
- [ ] `caliper validate` passes on all example configs

---

## M2: Mock experiment runner

---

### M2-01 · Model provider ABC and registry

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M0-01

**Objective**  
Define provider interface with retry/timeout wrapper and registration decorator.

**Acceptance criteria**
- [ ] `BaseModelProvider` with `_generate_once`, `generate`, `generate_batch`
- [ ] `ModelRequest` / `ModelResponse` pydantic types
- [ ] `@register_provider` and `create_provider(type, ...)`

---

### M2-02 · MockProvider and RandomProvider

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M2-01

**Objective**  
Deterministic and stochastic providers for testing without APIs.

**Acceptance criteria**
- [ ] MockProvider: same seed → same output
- [ ] RandomProvider: distinct outputs across calls
- [ ] `tests/test_models.py` covers both

---

### M2-03 · Cell expansion and cell_id generation

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M1-03

**Objective**  
Stable cell identifiers for resume and logging.

**Acceptance criteria**
- [ ] `expand_cells(config)` returns list of `ExperimentCombination`
- [ ] `make_cell_id(config, cell)` is deterministic
- [ ] `cell_to_dict` suitable for structlog

---

### M2-04 · ExperimentRunner with dry-run

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** L · **Dependencies:** M2-01, M2-03, M1-05

**Objective**  
Orchestrate full factorial grid with optional dry-run mode.

**Acceptance criteria**
- [ ] `ExperimentRunner.run()` executes all cells or skips on `--dry-run`
- [ ] `execute_cell` builds prompt, calls provider, aggregates task scores
- [ ] Failed cells recorded with `status: failed` (no silent drop)
- [ ] `RunManifest` tracks completed/failed/skipped counts

---

### M2-05 · ResultWriter (JSONL + Parquet + manifest)

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M2-04

**Objective**  
Append-only persistence with resume support.

**Acceptance criteria**
- [ ] Writes `results.jsonl` incrementally per cell
- [ ] `finalize()` exports `results.parquet`
- [ ] `load_existing()` skips completed cell_ids on `--resume`
- [ ] `manifest.json` written at run end

---

### M2-06 · example_factorial.yaml and runner tests

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M2-04, M2-05

**Objective**  
End-to-end mock factorial run in CI-friendly config.

**Acceptance criteria**
- [ ] `configs/examples/example_factorial.yaml` runs to completion
- [ ] `tests/test_factorial_runner.py` covers resume and cell counts
- [ ] Output under `outputs/example_factorial/<run_id>/`

---

## M3: Real API providers

---

### M3-01 · Shared API provider utilities

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M2-01

**Objective**  
Common code for API keys, dry-run, cost estimation, error mapping.

**Acceptance criteria**
- [ ] `caliper/models/api_common.py` with `ApiProviderMixin`
- [ ] `caliper/models/cost.py` with YAML-driven pricing hooks
- [ ] Rate-limit errors map to `ProviderGenerationError(retryable=True)`

---

### M3-02 · OpenAIProvider

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M3-01

**Objective**  
OpenAI chat completions via official SDK; model from YAML only.

**Acceptance criteria**
- [ ] `@register_provider("openai")` lazy-imports SDK
- [ ] Reads `OPENAI_API_KEY`; supports dry-run
- [ ] Populates token usage and `raw_metadata`
- [ ] Mocked unit test for response mapping

---

### M3-03 · AnthropicProvider

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M3-01

**Objective**  
Anthropic Messages API integration.

**Acceptance criteria**
- [ ] `@register_provider("anthropic")`
- [ ] Reads `ANTHROPIC_API_KEY`
- [ ] Handles rate limits and timeouts
- [ ] Mocked unit test

---

### M3-04 · GeminiProvider

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M3-01

**Objective**  
Google Gemini via `google-genai`; `GEMINI_API_KEY` with `GOOGLE_API_KEY` fallback.

**Acceptance criteria**
- [ ] `@register_provider("gemini")` and alias `google`
- [ ] Mocked unit test
- [ ] Documented in README and `.env.example`

---

### M3-05 · Wire API providers in executor.build_provider

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** S · **Dependencies:** M3-02, M3-03, M3-04

**Objective**  
Pass `api_key_env`, `base_url`, and `extra` from YAML to providers.

**Acceptance criteria**
- [ ] `SUPPORTED_PROVIDER_TYPES` includes API types
- [ ] Retry/timeout from provider `extra` applied to `ProviderRuntimeConfig`
- [ ] `build_provider` test with OpenAI config

---

### M3-06 · API provider tests (unit + optional integration)

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M3-05

**Objective**  
Test without keys by default; live tests when keys present.

**Acceptance criteria**
- [ ] `tests/test_api_providers.py` with mocked SDKs
- [ ] `tests/test_api_providers_integration.py` marked `@pytest.mark.integration`
- [ ] `pyproject.toml` excludes integration by default in `addopts`
- [ ] Optional dep group `[api]` in pyproject

---

## M4: Local model provider

---

### M4-01 · LocalModelSettings and backend ABC

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M2-01

**Objective**  
Configuration dataclass and backend factory for local inference.

**Acceptance criteria**
- [ ] `LocalModelSettings.from_config()` reads YAML + `LOCAL_MODEL_PATH`
- [ ] `create_local_backend()` selects transformers / llama_cpp / vllm
- [ ] `@register_provider("local")` `LocalModelProvider`

---

### M4-02 · TransformersBackend

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** L · **Dependencies:** M4-01

**Objective**  
HuggingFace causal LM inference with optional 4bit/8bit quantization.

**Acceptance criteria**
- [ ] Lazy import; clear error if `[local]` not installed
- [ ] Greedy decoding at temperature 0 when deterministic
- [ ] Logs inference latency in response metadata

---

### M4-03 · LlamaCppBackend and VllmBackend

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** L · **Dependencies:** M4-01

**Objective**  
GGUF and vLLM backends as optional extras.

**Acceptance criteria**
- [ ] `llama_cpp` loads `.gguf` with `n_gpu_layers`
- [ ] `vllm` single-GPU `tensor_parallel_size=1`
- [ ] Unit tests mock backends

---

### M4-04 · GPU/NVML metadata logging

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M4-01

**Objective**  
Log GPU name, memory, optional power/energy via pynvml.

**Acceptance criteria**
- [ ] `collect_gpu_metadata()` in `local/metadata.py`
- [ ] Optional `nvml: true` in YAML enables power sampling
- [ ] Metadata attached to `ModelResponse.raw_metadata`

---

### M4-05 · local_model.yaml example and docs/local-models.md

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M4-02

**Objective**  
Document installation and configuration for local backends.

**Acceptance criteria**
- [ ] `configs/examples/local_model.yaml` validates (placeholder path OK)
- [ ] `docs/local-models.md` covers `[local]`, `[local-llama-cpp]`, `[local-vllm]`
- [ ] README links to local-models doc

---

## M5: Evaluation metrics

---

### M5-01 · Evaluation runner and result schema

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M2-05

**Objective**  
Post-hoc evaluation producing long-format metric tables.

**Acceptance criteria**
- [ ] `MetricEvaluationRecord` schema
- [ ] `evaluate_results_file()` writes evaluations.parquet/jsonl
- [ ] Columns: model, task_id, prompt_id, run_id, metric_name, metric_value

---

### M5-02 · Code metrics (exact_match, contains_expected)

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M5-01

**Objective**  
Implement primary code evaluation metrics.

**Acceptance criteria**
- [ ] Normalization before string compare
- [ ] Registered in metric registry
- [ ] Tests in `tests/test_evaluation.py`

---

### M5-03 · Summarization metrics (lexical_overlap, length)

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M5-01

**Objective**  
Overlap-based summarization scores for Paper 1 domains.

**Acceptance criteria**
- [ ] `lexical_overlap` and `length` metrics implemented
- [ ] Used in example_factorial summarization task

---

### M5-04 · caliper evaluate CLI command

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** S · **Dependencies:** M5-01

**Objective**  
CLI entry point for evaluation pipeline.

**Acceptance criteria**
- [ ] `caliper evaluate RESULTS -c CONFIG` runs end-to-end
- [ ] `--enable-code-execution` and `--enable-llm-judge` flags exist
- [ ] Documented in README CLI table

---

### M5-05 · Implement test_pass and llm_judge (or document deferral)

**Status:** 🔴 Todo  
**Priority:** P2 · **Effort:** L · **Dependencies:** M5-04

**Objective**  
Replace placeholder metrics with real implementations or formally defer to post-v1.0.

**Acceptance criteria**
- [ ] **Option A:** `test_pass` executes sandboxed tests on generated code
- [ ] **Option B:** Document in README and CLI help that metrics are deferred to v1.1; remove "placeholder" from user-facing text
- [ ] **Option A for llm_judge:** wire optional judge provider from config
- [ ] Tests updated; no misleading `--enable-*` flags if deferred
- [ ] Decision recorded in CHANGELOG or docs/metrics.md

---

## M6: Paper 1 statistics

---

### M6-01 · prepare_results_table and descriptive stats

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M5-01

**Objective**  
Normalize evaluation tables for analysis modules.

**Acceptance criteria**
- [ ] Column alias handling (score → metric_value, etc.)
- [ ] `descriptive_by_factor()` and `descriptive_all_factors()`
- [ ] Tests in `tests/test_statistics_paper1.py`

---

### M6-02 · Sequential ANOVA variance decomposition

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M6-01

**Objective**  
Type-I variance components for model, task, prompt, run, temperature.

**Acceptance criteria**
- [ ] `decompose_variance()` returns `VarianceComponents`
- [ ] Documented factor order in docstring
- [ ] `caliper analyze variance` prints components

---

### M6-03 · Mixed-effects fallback (MixedLM)

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** M · **Dependencies:** M6-02

**Objective**  
statsmodels MixedLM with ANOVA fallback on failure.

**Acceptance criteria**
- [ ] `fit_mixed_model()` in `mixed_effects.py`
- [ ] Graceful fallback logged
- [ ] Test with synthetic data

---

### M6-04 · G-theory G-study and D-study grid

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** L · **Dependencies:** M6-02

**Objective**  
Estimate G coefficients and simulate D-study over n_tasks, n_prompts, n_runs.

**Acceptance criteria**
- [ ] `estimate_g_variance_components()` and `simulate_d_study_grid()`
- [ ] Returns G and Phi for grid points
- [ ] Used by variance_decomposition.py script

---

### M6-05 · Monte Carlo power simulation

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M6-04

**Objective**  
Simulate pairwise model comparison power over design grids.

**Acceptance criteria**
- [ ] `simulate_power_grid()` with configurable effect size and counts
- [ ] `caliper analyze power` CLI
- [ ] `analyses/paper1/power_simulation.py` script

---

### M6-06 · CLI analyze variance/power + paper1 scripts

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M6-02, M6-05

**Objective**  
Ship standalone analysis scripts for Paper 1 reproduction.

**Acceptance criteria**
- [ ] `analyses/paper1/variance_decomposition.py` with `--results`, `--output-dir`
- [ ] `analyses/paper1/README.md` documents methods and limitations
- [ ] Scripts run on example_factorial outputs without error

---

## M7: Paper 2 ranking fragility

---

### M7-01 · Aggregate scores and rank_models

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** S · **Dependencies:** M6-01

**Objective**  
Baseline leaderboard construction from result matrices.

**Acceptance criteria**
- [ ] `aggregate_scores_by_model()` and `rank_models()` in `ranking/aggregate.py`
- [ ] Deterministic tie-breaking
- [ ] Unit tests

---

### M7-02 · Bootstrap over task/prompt/run facets

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M7-01

**Objective**  
Resample facet levels with replacement; recompute ranks per iteration.

**Acceptance criteria**
- [ ] `bootstrap_rankings()` and `bootstrap_all_facets()`
- [ ] Stores long-format samples with iteration, model, rank, tau
- [ ] Fixed random seed support

---

### M7-03 · Kendall tau, fragility index, pairwise reversals

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M7-02

**Objective**  
Implement Paper 2 formal metrics.

**Acceptance criteria**
- [ ] `kendall_tau_between_rankings()`, `ranking_fragility_index()` — F = (1−τ̄)/2
- [ ] `pairwise_reversal_probability()` groups by iteration and bootstrap_type
- [ ] `rank_probability_matrix()` for P(rank=k)

---

### M7-04 · Analysis orchestration, plots, CSV/Parquet outputs

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M7-03

**Objective**  
Full pipeline writing summary tables and matplotlib plots.

**Acceptance criteria**
- [ ] `run_ranking_fragility_analysis()` writes CSV, parquet, jsonl
- [ ] Plots: tau distribution, rank heatmap, pairwise heatmap, baseline bar
- [ ] `matplotlib` only (no seaborn requirement)

---

### M7-05 · ranking-fragility CLI and synthetic tests

**Status:** ✅ Done  
**Priority:** P0 · **Effort:** M · **Dependencies:** M7-04

**Objective**  
CLI command and stable vs unstable synthetic data tests.

**Acceptance criteria**
- [ ] `caliper ranking-fragility RESULTS --metric NAME`
- [ ] `generate_stable_ranking_data()` / `generate_unstable_ranking_data()`
- [ ] `tests/test_ranking_fragility.py` — stable τ > unstable τ
- [ ] 170+ tests pass in full suite

---

## M8: Reproducibility package

---

### M8-01 · Add LICENSE (MIT) and CONTRIBUTING.md

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** S · **Dependencies:** M0-01

**Objective**  
Legal clarity for public GitHub release.

**Acceptance criteria**
- [ ] `LICENSE` file with MIT text and copyright year
- [ ] `CONTRIBUTING.md` with PR process, code style (ruff), test requirement
- [ ] README license section links to LICENSE (remove "TBD" language)

---

### M8-02 · GitHub Actions: lint, typecheck, pytest

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** M · **Dependencies:** M0-06, M8-01

**Objective**  
CI on push/PR to main.

**Acceptance criteria**
- [ ] `.github/workflows/ci.yml` runs on Python 3.11 and 3.12
- [ ] Steps: `pip install -e ".[dev]"`, `ruff check`, `mypy caliper`, `pytest`
- [ ] Badge in README (optional)
- [ ] CI passes on current main

---

### M8-03 · Add analyses/paper2/ standalone script

**Status:** 🔴 Todo  
**Priority:** P1 · **Effort:** M · **Dependencies:** M7-05

**Objective**  
Mirror paper1 scripts for ranking fragility reproduction.

**Acceptance criteria**
- [ ] `analyses/paper2/ranking_fragility.py` with `--results`, `--metric`, `--n-bootstrap`, `--output-dir`
- [ ] `analyses/paper2/README.md` documents inputs/outputs
- [ ] Script equivalent to `caliper ranking-fragility` CLI

---

### M8-04 · Confirmatory configs (paper1, paper2)

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** M · **Dependencies:** M6-06, M7-05, M1-07

**Objective**  
Frozen YAML configs matching preregistration (placeholder paths until datasets locked).

**Acceptance criteria**
- [ ] `configs/paper1_confirmatory.yaml` — models, tasks, prompts, runs, metrics documented in comments
- [ ] `configs/paper2_confirmatory.yaml` — same result matrix requirements for ranking analysis
- [ ] Both pass `caliper validate`
- [ ] Config commit hash referenced in `paper1/preregistration.md` checklist

---

### M8-05 · Reproduction guide (docs/reproduction.md)

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** M · **Dependencies:** M8-03, M8-04

**Objective**  
Step-by-step instructions to reproduce Paper 1 and Paper 2 analyses from mock or published data.

**Acceptance criteria**
- [ ] Section: mock end-to-end (no API keys) in < 30 minutes
- [ ] Section: full pipeline with API/local providers
- [ ] Exact commands with expected output paths
- [ ] Troubleshooting (failed cells, resume, missing columns)
- [ ] Linked from README

---

### M8-06 · Export script for OSF/archival bundle

**Status:** 🔴 Todo  
**Priority:** P1 · **Effort:** M · **Dependencies:** M8-04, M8-05

**Objective**  
Package configs, manifests, evaluation tables, and analysis outputs for upload.

**Acceptance criteria**
- [ ] `scripts/export_osf_bundle.py` (or `make export-bundle`) creates tarball
- [ ] Includes: config YAML, git hash, pyproject version, results parquet, analysis CSVs, plots
- [ ] Excludes: `.env`, API keys, raw secrets
- [ ] `manifest.json` lists bundle contents with SHA256 hashes

---

### M8-07 · GitHub issue and PR templates

**Status:** 🔴 Todo  
**Priority:** P2 · **Effort:** S · **Dependencies:** M8-01

**Objective**  
Standardize contributor reporting using this issues backlog.

**Acceptance criteria**
- [ ] `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`
- [ ] `.github/PULL_REQUEST_TEMPLATE.md` with checklist (tests, docs)
- [ ] Optional: issue template linking to milestone IDs (M0-01 format)

---

### M8-08 · Pin optional extras in CI matrix (api, local smoke)

**Status:** 🔴 Todo  
**Priority:** P2 · **Effort:** M · **Dependencies:** M8-02

**Objective**  
Verify optional dependency groups install without running GPU/API jobs.

**Acceptance criteria**
- [ ] CI job: `pip install -e ".[api]"` + import check only
- [ ] CI job: `pip install -e ".[local]"` + import transformers (no GPU test)
- [ ] Integration tests remain opt-in via workflow_dispatch or label

---

### M8-09 · Global README and roadmap

**Status:** ✅ Done  
**Priority:** P1 · **Effort:** S · **Dependencies:** M6-06, M7-05

**Objective**  
Repository-level documentation for users and reviewers.

**Acceptance criteria**
- [ ] README covers install, mock/API/local, output structure, citation placeholder
- [ ] `docs/roadmap.md` and `docs/issues.md` (this file)
- [ ] No fabricated empirical results

---

### M8-10 · Pre-commit config (ruff)

**Status:** 🔴 Todo  
**Priority:** P2 · **Effort:** S · **Dependencies:** M8-02

**Objective**  
Local hook matching CI lint rules.

**Acceptance criteria**
- [ ] `.pre-commit-config.yaml` with ruff check + format
- [ ] Documented in CONTRIBUTING.md
- [ ] Optional: `make pre-commit-install` target

---

## M9: Submission freeze

---

### M9-01 · Lock Paper 1 preregistration and confirmatory config

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** S · **Dependencies:** M8-04

**Objective**  
Post preregistration to OSF; freeze config commit before empirical collection.

**Acceptance criteria**
- [ ] `paper1/preregistration.md` posted to OSF with DOI
- [ ] `configs/paper1_confirmatory.yaml` committed with final dataset paths and n_m, n_t, n_p, n_r
- [ ] Git tag `paper1-prereg-lock` on config commit
- [ ] Amendments log initialized (empty)

---

### M9-02 · Execute Paper 1 confirmatory experiments

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** XL · **Dependencies:** M9-01, M3-05 or M4-05

**Objective**  
Run full factorial grid for Paper 1 preregistered design.

**Acceptance criteria**
- [ ] All cells attempted; manifest records completed/failed/skipped
- [ ] `caliper evaluate` produces evaluations.parquet
- [ ] `caliper analyze variance` and `analyze power` run on confirmatory data
- [ ] Data audit table: exclusion counts per preregistration rules
- [ ] No optional stopping before completion

---

### M9-03 · Execute Paper 2 confirmatory experiments

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** XL · **Dependencies:** M9-02 (or shared result matrix)

**Objective**  
Ensure result matrix supports ranking fragility analysis for Paper 2.

**Acceptance criteria**
- [ ] Same or superseding result matrix meets Paper 2 preregistration (≥3 models, facet columns present)
- [ ] `caliper ranking-fragility` run on confirmatory evaluations
- [ ] Outputs archived under `reports/paper2_confirmatory/`
- [ ] Bootstrap B and seed match preregistration

---

### M9-04 · Populate manuscript results (paper1/, paper2/)

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** L · **Dependencies:** M9-02, M9-03

**Objective**  
Replace LaTeX placeholders with empirical tables and figures.

**Acceptance criteria**
- [ ] `paper1/sections/06_results.tex` populated; TBD cells removed
- [ ] `paper2/sections/06_results.tex` populated
- [ ] All numbers traceable to archived CSV/Parquet (file + line documented)
- [ ] No results in manuscript that aren't in archived outputs

---

### M9-05 · Generate publication figures from CALIPER outputs

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** M · **Dependencies:** M9-02, M9-03

**Objective**  
Export vector PDFs for journals from CALIPER matplotlib plots.

**Acceptance criteria**
- [ ] Paper 1: variance components, D-study curves, power heatmaps in `paper1/figures/`
- [ ] Paper 2: fragility by facet, rank probability, pairwise, tau distribution in `paper2/figures/`
- [ ] Figure filenames match `\includegraphics` in LaTeX
- [ ] `make pdf` in paper1/ and paper2/ succeeds

---

### M9-06 · Tag caliper v1.0.0 and archive on OSF/Zenodo

**Status:** 🔴 Todo  
**Priority:** P0 · **Effort:** M · **Dependencies:** M9-04, M8-06

**Objective**  
Immutable software release linked from papers.

**Acceptance criteria**
- [ ] Git tag `v1.0.0` on release commit
- [ ] CHANGELOG.md summarizes v1.0.0 scope
- [ ] OSF/Zenodo upload includes bundle from M8-06
- [ ] README citation block updated with DOI and version

---

### M9-07 · Finalize citations and replace README placeholders

**Status:** 🔴 Todo  
**Priority:** P1 · **Effort:** S · **Dependencies:** M9-06

**Objective**  
Remove PLACEHOLDER URLs and unpublished boilerplate where possible.

**Acceptance criteria**
- [ ] GitHub repository URL in README and bibtex
- [ ] Paper bibtex entries updated when DOI available (or "forthcoming" with journal name)
- [ ] `references.bib` in paper1/ and paper2/ finalized

---

### M9-08 · Submission checklist and deviation log

**Status:** 🔴 Todo  
**Priority:** P1 · **Effort:** S · **Dependencies:** M9-01, M9-04

**Objective**  
Document any preregistration deviations and submission readiness.

**Acceptance criteria**
- [ ] `docs/submission_checklist.md` with IP&M and KBS requirements
- [ ] OSF amendments log complete for any analysis changes post-lock
- [ ] Artifact, preregistration, and manuscript cross-linked
- [ ] Co-author sign-off checklist (manual)

---

## Issue creation cheat sheet

**GitHub labels to create:**

```text
milestone:M0 … milestone:M9
priority:P0, priority:P1, priority:P2
effort:S, effort:M, effort:L, effort:XL
status:done, status:todo
area:config, area:runner, area:providers, area:stats, area:ranking, area:docs, area:ci
```

**Suggested GitHub milestones (close when all child issues closed):**

| Milestone | Due | Open issues (approx.) |
|-----------|-----|------------------------|
| M8 Reproducibility package | TBD | 8 |
| M9 Submission freeze | TBD | 8 |

M0–M7 milestones can be created and immediately closed for audit trail, or omitted if using `status:done` labels only.

---

## Cursor implementation prompt template

When implementing an open issue, paste:

```markdown
Implement CALIPER issue **[ID]: [Title]**

Repository: ~/papers/caliper/caliper

Objective: [copy from issue]

Acceptance criteria:
[copy checklist]

Dependencies already merged: [list IDs]

Constraints:
- Minimize scope to this issue only
- Match existing code conventions
- Add/update tests for new behavior
- Do not invent empirical results
- Do not commit unless asked
```

---

*End of issues backlog.*
