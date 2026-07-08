"""Controlled prompt family for Paper 1 confirmatory study.

All variants enforce the same output format so instruction style is the only
manipulated factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PromptStyle = Literal["minimal", "explicit_reasoning", "testing_oriented", "professional"]

CONTROLLED_OUTPUT_SUFFIX = (
    "\n\nRespond with exactly one Python code block using triple backticks "
    "(` ```python `). Do not include explanations, commentary, or text "
    "outside the code block."
)


@dataclass(frozen=True)
class ConfirmatoryPromptFamily:
    """Four controlled prompt styles with identical output constraints."""

    style: PromptStyle
    template: str

    def render(self, task_input: str) -> str:
        body = self.template.format(input=task_input)
        return f"{body}{CONTROLLED_OUTPUT_SUFFIX}"


_STYLE_PREFIXES: dict[PromptStyle, str] = {
    "minimal": "Complete the following Python programming task.\n\n{input}",
    "explicit_reasoning": (
        "Solve the following Python programming task. "
        "You may reason internally, but your response must contain only code "
        "in the required format.\n\n{input}"
    ),
    "testing_oriented": (
        "Write Python code for the following task so that it passes hidden unit tests.\n\n{input}"
    ),
    "professional": (
        "As a professional Python developer, implement the following task with "
        "clear, maintainable code.\n\n{input}"
    ),
}


def controlled_prompt_templates() -> list[ConfirmatoryPromptFamily]:
    """Return the four confirmatory prompt variants."""
    return [
        ConfirmatoryPromptFamily(style=style, template=prefix)
        for style, prefix in _STYLE_PREFIXES.items()
    ]


def prompt_variant_yaml(style: PromptStyle) -> dict[str, str]:
    """Serialize one prompt variant for experiment YAML configs."""
    family = next(p for p in controlled_prompt_templates() if p.style == style)
    return {"id": style, "template": family.template + CONTROLLED_OUTPUT_SUFFIX}
