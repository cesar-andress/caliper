# CALIPER Development Roadmap

This document describes the milestone plan for the CALIPER research artifact (`caliper/`). Detailed GitHub-ready issues are in [`issues.md`](issues.md).

**Last updated:** 2026-07-04  
**Target:** Two companion papers (Paper 1: variance/power; Paper 2: ranking fragility)

---

## Overview

CALIPER is built in layers: configuration and tasks → experiment execution → evaluation → paper-specific analyses → reproducibility packaging → submission freeze. Each milestone maps to a GitHub milestone; issues within milestones are sized for single PRs implementable by Cursor or a human contributor.

```text
M0 Scaffold → M1 Config/Tasks → M2 Runner → M3 API → M4 Local
     → M5 Metrics → M6 Paper 1 → M7 Paper 2 → M8 Repro → M9 Freeze
```

---

## Milestone summary

| Milestone | Name | Status | Issue count | Notes |
|-----------|------|--------|-------------|-------|
| **M0** | Repository scaffold | ✅ Complete | 6 | Package, CLI shell, docs skeleton |
| **M1** | Config and task system | ✅ Complete | 7 | YAML schemas, tasks, prompts |
| **M2** | Mock experiment runner | ✅ Complete | 6 | Factorial runner, resume, manifests |
| **M3** | Real API providers | ✅ Complete | 6 | OpenAI, Anthropic, Gemini |
| **M4** | Local model provider | ✅ Complete | 5 | transformers, llama.cpp, vLLM |
| **M5** | Evaluation metrics | 🟡 Mostly complete | 5 | `test_pass`, `llm_judge` still placeholders |
| **M6** | Paper 1 statistics | ✅ Complete | 6 | Variance, G-theory, power sim |
| **M7** | Paper 2 ranking fragility | ✅ Complete | 5 | Bootstrap, fragility index, CLI |
| **M8** | Reproducibility package | 🔴 In progress | 10 | CI, LICENSE, confirmatory configs, Paper 2 scripts |
| **M9** | Submission freeze | 🔴 Not started | 8 | Empirical runs, tags, OSF, manuscripts |

**Legend:** ✅ Complete · 🟡 Partial · 🔴 Not started / in progress

---

## M0: Repository scaffold

**Goal:** Installable Python package with CLI entry point, dev tooling, and test harness.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M0-01 | Initialize pyproject.toml and package layout | P0 | S | ✅ |
| M0-02 | Add Makefile and .env.example | P0 | S | ✅ |
| M0-03 | Implement Click CLI skeleton | P0 | S | ✅ |
| M0-04 | Add structlog logging utilities | P1 | S | ✅ |
| M0-05 | Create architecture and getting-started docs | P1 | M | ✅ |
| M0-06 | Initial pytest suite and conftest | P0 | M | ✅ |

---

## M1: Config and task system

**Goal:** Validated YAML experiment configs and pluggable task loaders.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M1-01 | Pydantic schemas for experiment config | P0 | M | ✅ |
| M1-02 | Config loader with validation errors | P0 | M | ✅ |
| M1-03 | Factorial combination expansion | P0 | M | ✅ |
| M1-04 | BaseTask ABC and task registry | P0 | M | ✅ |
| M1-05 | Code-gen, bug-repair, summarization tasks | P0 | L | ✅ |
| M1-06 | Prompt template loader and renderer | P1 | M | ✅ |
| M1-07 | Example configs and config tests | P1 | M | ✅ |

---

## M2: Mock experiment runner

**Goal:** End-to-end factorial execution with mock providers and resumable output.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M2-01 | Model provider ABC and registry | P0 | M | ✅ |
| M2-02 | MockProvider and RandomProvider | P0 | M | ✅ |
| M2-03 | Cell expansion and cell_id generation | P0 | M | ✅ |
| M2-04 | ExperimentRunner with dry-run | P0 | L | ✅ |
| M2-05 | ResultWriter (JSONL + Parquet + manifest) | P0 | M | ✅ |
| M2-06 | example_factorial.yaml and runner tests | P0 | M | ✅ |

---

## M3: Real API providers

**Goal:** Production API integrations with retries, timeouts, and optional integration tests.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M3-01 | Shared API provider utilities (retry, dry-run, cost hooks) | P0 | M | ✅ |
| M3-02 | OpenAIProvider | P0 | M | ✅ |
| M3-03 | AnthropicProvider | P0 | M | ✅ |
| M3-04 | GeminiProvider (+ google alias) | P0 | M | ✅ |
| M3-05 | Wire API providers in executor.build_provider | P0 | S | ✅ |
| M3-06 | Mocked unit tests + optional integration tests | P1 | M | ✅ |

---

## M4: Local model provider

**Goal:** Optional GPU inference via transformers, llama.cpp, and vLLM.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M4-01 | LocalModelSettings and backend ABC | P0 | M | ✅ |
| M4-02 | TransformersBackend | P0 | L | ✅ |
| M4-03 | LlamaCppBackend and VllmBackend | P1 | L | ✅ |
| M4-04 | GPU/NVML metadata logging | P1 | M | ✅ |
| M4-05 | local_model.yaml example and docs/local-models.md | P1 | M | ✅ |

---

## M5: Evaluation metrics

**Goal:** Post-hoc metric evaluation pipeline from saved run results.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M5-01 | Evaluation runner and result schema | P0 | M | ✅ |
| M5-02 | Code metrics (exact_match, contains_expected) | P0 | M | ✅ |
| M5-03 | Summarization metrics (lexical_overlap, length) | P1 | M | ✅ |
| M5-04 | caliper evaluate CLI command | P0 | S | ✅ |
| M5-05 | Implement test_pass and llm_judge (or document deferral) | P2 | L | 🔴 |

---

## M6: Paper 1 statistics

**Goal:** Variance decomposition, G-/D-study, and power simulation.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M6-01 | prepare_results_table and descriptive stats | P0 | M | ✅ |
| M6-02 | Sequential ANOVA variance decomposition | P0 | M | ✅ |
| M6-03 | Mixed-effects fallback (MixedLM) | P1 | M | ✅ |
| M6-04 | G-theory G-study and D-study grid | P0 | L | ✅ |
| M6-05 | Monte Carlo power simulation | P0 | M | ✅ |
| M6-06 | CLI analyze variance/power + paper1 scripts | P0 | M | ✅ |

---

## M7: Paper 2 ranking fragility

**Goal:** Bootstrap ranking analysis and fragility metrics.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M7-01 | Aggregate scores and rank_models | P0 | S | ✅ |
| M7-02 | Bootstrap over task/prompt/run facets | P0 | M | ✅ |
| M7-03 | Kendall tau, fragility index, pairwise reversals | P0 | M | ✅ |
| M7-04 | Analysis orchestration, plots, CSV/Parquet outputs | P0 | M | ✅ |
| M7-05 | ranking-fragility CLI and synthetic stable/unstable tests | P0 | M | ✅ |

---

## M8: Reproducibility package

**Goal:** Public-ready repository with CI, legal files, confirmatory configs, and one-command reproduction paths.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M8-01 | Add LICENSE (MIT) and CONTRIBUTING.md | P0 | S | 🔴 |
| M8-02 | GitHub Actions: lint, typecheck, pytest | P0 | M | 🔴 |
| M8-03 | Add analyses/paper2/ standalone script | P1 | M | 🔴 |
| M8-04 | Confirmatory configs (paper1, paper2) | P0 | M | 🔴 |
| M8-05 | Reproduction guide (docs/reproduction.md) | P0 | M | 🔴 |
| M8-06 | Export script for OSF/archival bundle | P1 | M | 🔴 |
| M8-07 | GitHub issue and PR templates | P2 | S | 🔴 |
| M8-08 | Pin optional extras in CI matrix (api, local smoke) | P2 | M | 🔴 |
| M8-09 | Global README and roadmap (this doc) | P1 | S | ✅ |
| M8-10 | Pre-commit config (ruff) | P2 | S | 🔴 |

---

## M9: Submission freeze

**Goal:** Locked empirical study, versioned release, and manuscript submission artifacts.

| ID | Title | Priority | Effort | Status |
|----|-------|----------|--------|--------|
| M9-01 | Lock Paper 1 preregistration and confirmatory config | P0 | S | 🔴 |
| M9-02 | Execute Paper 1 confirmatory experiments | P0 | XL | 🔴 |
| M9-03 | Execute Paper 2 confirmatory experiments | P0 | XL | 🔴 |
| M9-04 | Populate manuscript results (paper1/, paper2/) | P0 | L | 🔴 |
| M9-05 | Generate publication figures from CALIPER outputs | P0 | M | 🔴 |
| M9-06 | Tag caliper v1.0.0 and archive on OSF/Zenodo | P0 | M | 🔴 |
| M9-07 | Finalize citations and replace README placeholders | P1 | S | 🔴 |
| M9-08 | Submission checklist and deviation log | P1 | S | 🔴 |

---

## Effort scale

| Label | Typical scope |
|-------|----------------|
| **S** | ≤ 4 hours — single file, tests included |
| **M** | 1–2 days — one module or CLI command |
| **L** | 3–5 days — cross-cutting feature |
| **XL** | 1+ weeks — empirical campaign or multi-benchmark runs |

---

## Priority scale

| Label | Meaning |
|-------|---------|
| **P0** | Blocks papers or public release |
| **P1** | Important for reproducibility and reviewer response |
| **P2** | Nice-to-have; defer if schedule slips |

---

## How to use this roadmap

1. **Create GitHub milestones** M0–M9 and close M0–M7 (or leave open for audit).
2. **Copy issues** from [`issues.md`](issues.md) into GitHub Issues (one issue per section).
3. **Implement in dependency order** — each issue lists blocking issue IDs.
4. **Cursor workflow:** Paste a single issue (title + objective + acceptance criteria) as the task prompt; reference files in the acceptance criteria.
5. **Track deviations** in Paper 1 preregistration amendments and M9-08 checklist.

---

## Critical path to submission

```text
M8-01 → M8-02 → M8-04 → M9-01 → M9-02 → M9-04 → M9-06
                              ↘ M9-03 → M9-04 → M9-05 → M9-06
M5-05 (optional) ─────────────────────────────────────────────→ M9-08
```

---

## Related documents

- [`issues.md`](issues.md) — Full issue specifications
- [`architecture.md`](architecture.md) — Module map
- [`../../paper1/preregistration.md`](../../paper1/preregistration.md) — Paper 1 analysis lock
- [`../analyses/paper1/README.md`](../analyses/paper1/README.md) — Paper 1 scripts
