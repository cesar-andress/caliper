.PHONY: install install-dev lint format typecheck test test-cov clean run help paper1-analysis paper1-robustness paper1-confirmatory-prep paper1-confirmatory-analysis paper1-confirmatory-robustness paper1-preflight paper1-humaneval-full-prep paper1-humaneval-full-preflight paper1-humaneval-full-analysis paper1-humaneval-full-robustness paper1-humaneval-full-task-sampling paper1-humaneval-full-design-guidance

PYTHON ?= python3.12
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
CALIPER := $(VENV_PYTHON) -m caliper

PAPER1_CONFIRMATORY_DIR ?= experiments/paper1_confirmatory_humaneval/paper1_confirmatory_humaneval
PAPER1_HUMANEVAL_FULL_DIR ?= experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full
PAPER1_HUMANEVAL_FULL_CONFIG ?= configs/paper1/confirmatory_humaneval_full.yaml

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/python ## Install package in production mode
	@$(VENV_PYTHON) -c "import sys; ver=sys.version_info; assert ver >= (3, 11), f'Python 3.11+ required, got {sys.version}'"
	$(PIP) install -e .

install-dev: $(VENV)/bin/python ## Install package with dev dependencies
	@$(VENV_PYTHON) -c "import sys; ver=sys.version_info; assert ver >= (3, 11), f'Python 3.11+ required, got {sys.version}'"
	$(PIP) install -e ".[dev]"

lint: ## Run ruff linter
	$(RUFF) check caliper tests

format: ## Auto-format code with ruff
	$(RUFF) format caliper tests
	$(RUFF) check --fix caliper tests

typecheck: ## Run mypy type checker
	$(MYPY) caliper

test: ## Run test suite
	$(PYTEST) tests/

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=caliper --cov-report=term-missing

run: install ## Run example experiment (dry-run)
	$(CALIPER) run --config configs/examples/basic_experiment.yaml --dry-run

paper1-analysis: ## Generate Paper 1 publication tables/figures from pilot outputs
	$(PYTHON) analyses/paper1/generate_publication_analysis.py \
		--experiment-dir experiments/paper1_ollama_pilot

paper1-robustness: ## Generate Paper 1 robustness analysis (ANOVA, convergence, bootstrap)
	$(PYTHON) analyses/paper1/generate_robustness_analysis.py \
		--experiment-dir experiments/paper1_ollama_pilot

paper1-confirmatory-prep: install ## Materialize benchmarks and write confirmatory YAML configs
	$(CALIPER) benchmarks materialize --output-dir data/benchmarks
	$(CALIPER) benchmarks write-configs --configs-dir configs/paper1 --data-dir data/benchmarks

paper1-confirmatory-analysis: ## Publication analysis for a confirmatory experiment directory
	$(PYTHON) analyses/paper1/generate_publication_analysis.py \
		--experiment-dir $(PAPER1_CONFIRMATORY_DIR)

paper1-confirmatory-robustness: ## Robustness + GLMM analysis for confirmatory experiment
	$(PYTHON) analyses/paper1/generate_robustness_analysis.py \
		--experiment-dir $(PAPER1_CONFIRMATORY_DIR)

paper1-preflight: install ## End-to-end pre-flight validation (requires Ollama + benchmarks)
	$(CALIPER) validate-confirmatory --benchmark humaneval --verbose

paper1-humaneval-full-prep: install ## Write 164-task HumanEval+ config and protocol comparison report
	$(CALIPER) benchmarks write-humaneval-full-config
	$(CALIPER) compare-protocol

paper1-humaneval-full-preflight: install ## Pre-flight for full HumanEval+ protocol (3 tasks, requires Ollama)
	$(CALIPER) validate-confirmatory --benchmark humaneval --verbose \
		--reference-config $(PAPER1_HUMANEVAL_FULL_CONFIG) \
		--expected-total-tasks 164 \
		--tasks 3 --runs 1 --temperature 0.0

paper1-humaneval-full-analysis: ## Publication analysis for full HumanEval+ experiment (after completion)
	$(PYTHON) analyses/paper1/generate_publication_analysis.py \
		--experiment-dir $(PAPER1_HUMANEVAL_FULL_DIR)
	$(PYTHON) analyses/paper1/generate_design_guidance.py \
		--experiment-dir $(PAPER1_HUMANEVAL_FULL_DIR)

paper1-humaneval-full-robustness: ## Robustness + GLMM analysis for full HumanEval+ experiment
	$(PYTHON) analyses/paper1/generate_robustness_analysis.py \
		--experiment-dir $(PAPER1_HUMANEVAL_FULL_DIR)

paper1-humaneval-full-task-sampling: ## Compare 40-task subset against full HumanEval+ benchmark
	$(CALIPER) analyze task-sampling \
		--full-experiment $(PAPER1_HUMANEVAL_FULL_DIR) \
		--subset-experiment $(PAPER1_CONFIRMATORY_DIR)

paper1-humaneval-full-design-guidance: ## Export design-guidance placeholders or populated recommendations
	$(PYTHON) analyses/paper1/generate_design_guidance.py \
		--experiment-dir $(PAPER1_HUMANEVAL_FULL_DIR)

clean: ## Remove build artifacts and caches
	rm -rf $(VENV) dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
