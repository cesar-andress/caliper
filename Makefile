.PHONY: install install-dev lint format typecheck test test-cov clean run help

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
CALIPER := $(VENV)/bin/caliper

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate ## Install package in production mode
	$(PIP) install -e .

install-dev: $(VENV)/bin/activate ## Install package with dev dependencies
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

run: ## Run example experiment (dry-run)
	$(CALIPER) run --config configs/examples/basic_experiment.yaml --dry-run

clean: ## Remove build artifacts and caches
	rm -rf $(VENV) dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
