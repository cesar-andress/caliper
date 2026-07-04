"""Load and render prompt templates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from caliper.config.schema import PromptVariantConfig


@dataclass
class PromptTemplate:
    """A resolved prompt template ready for rendering."""

    id: str
    template: str
    default_variables: dict[str, str] = field(default_factory=dict)

    def render(self, **variables: str) -> str:
        """Render the template with the given variables."""
        merged = {**self.default_variables, **variables}
        return render_prompt(self.template, merged)


def load_prompt(config: PromptVariantConfig) -> PromptTemplate:
    """Load a prompt template from inline text or a file path."""
    if config.template is not None:
        text = config.template
    elif config.path is not None:
        text = config.path.read_text(encoding="utf-8")
    else:
        msg = f"Prompt '{config.id}' must specify either 'template' or 'path'"
        raise ValueError(msg)

    return PromptTemplate(
        id=config.id,
        template=text,
        default_variables=config.variables,
    )


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def render_prompt(template: str, variables: dict[str, str]) -> str:
    """Substitute ``{variable}`` placeholders in a template string."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            msg = f"Missing template variable: {key}"
            raise KeyError(msg)
        return variables[key]

    return _PLACEHOLDER.sub(_replace, template)
