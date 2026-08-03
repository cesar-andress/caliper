"""Tests for executable code generation task and GLMM analysis."""

from __future__ import annotations

import pandas as pd

from caliper.statistics.glmm_analysis import fit_pass_fail_glmm
from caliper.tasks import create_task
from caliper.tasks.loader import TaskDataset
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "benchmarks" / "sample_tasks.jsonl"


def test_executable_task_scores_pass_at_1():
    task = create_task(
        "executable_code_generation",
        "fixture-task",
        FIXTURE_PATH,
        filter_task_id="he-fixture-001",
        metrics=["pass_at_1", "syntax_check", "normalized_code_match"],
    )
    example = task.load_examples()[0]
    scores = task.score(example, "```python\n    return a + b\n```")
    assert scores["pass_at_1"] == 1.0
    assert scores["syntax_check"] == 1.0


def test_glmm_fit_on_synthetic_pass_fail_data():
    rows = []
    for model in ("m1", "m2"):
        for prompt in ("minimal", "professional"):
            for task in ("t1", "t2"):
                for run in range(2):
                    base = 0.7 if model == "m1" else 0.3
                    prompt_bonus = 0.1 if prompt == "professional" else 0.0
                    value = 1.0 if (run + hash(task)) % 3 != 0 and base + prompt_bonus > 0.55 else 0.0
                    rows.append(
                        {
                            "model_id": model,
                            "prompt_variant_id": prompt,
                            "temperature": 0.0 if run % 2 == 0 else 0.2,
                            "task_id": task,
                            "run_index": run,
                            "pass_at_1": value,
                            "score": value,
                        }
                    )
    df = pd.DataFrame(rows)
    result = fit_pass_fail_glmm(df, metric="pass_at_1")
    assert result.primary.n_observations == len(rows)
    assert not result.coefficients.empty
    assert result.primary.valid_for_inference or result.primary.method.startswith("Binomial")
