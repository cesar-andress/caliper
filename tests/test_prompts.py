"""Tests for prompt loading and rendering."""

import pytest

from caliper.config.schema import PromptVariantConfig
from caliper.prompts.loader import load_prompt, render_prompt


class TestRenderPrompt:
    def test_simple_substitution(self) -> None:
        result = render_prompt("Hello {name}!", {"name": "world"})
        assert result == "Hello world!"

    def test_missing_variable_raises(self) -> None:
        with pytest.raises(KeyError, match="Missing template variable"):
            render_prompt("Hello {name}!", {})


class TestLoadPrompt:
    def test_inline_template(self) -> None:
        config = PromptVariantConfig(id="test", template="Q: {question}\nA:")
        prompt = load_prompt(config)
        assert prompt.id == "test"
        assert prompt.render(question="2+2?") == "Q: 2+2?\nA:"

    def test_requires_template_or_path(self) -> None:
        with pytest.raises(ValueError, match="must specify"):
            PromptVariantConfig(id="empty")
