"""Code summarization benchmark task."""

from __future__ import annotations

from caliper.tasks.base import BaseTask, exact_match_score, token_f1_score
from caliper.tasks.registry import register_task
from caliper.tasks.schema import TaskMetadata


@register_task("code_summarization")
class CodeSummarizationTask(BaseTask):
    """Evaluate natural-language summaries of code snippets."""

    domain = "code_summarization"

    def score(self, example: TaskMetadata, prediction: str) -> dict[str, float]:
        return {
            "exact_match": exact_match_score(example.expected_output, prediction),
            "rouge_l": token_f1_score(example.expected_output, prediction),
        }
