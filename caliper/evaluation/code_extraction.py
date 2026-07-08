"""Extract and normalize Python code from model predictions."""

from __future__ import annotations

import re

FENCED_PYTHON = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
FENCED_GENERIC = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str:
    """Extract Python code from a model prediction.

    Priority:
    1. ```python fenced block
    2. generic ``` fenced block
    3. raw text starting at the first ``def ``
    4. original prediction
    """
    stripped = text.strip()
    if not stripped:
        return stripped

    match = FENCED_PYTHON.search(stripped)
    if match:
        return match.group(1).strip()

    match = FENCED_GENERIC.search(stripped)
    if match:
        return match.group(1).strip()

    def_index = stripped.find("def ")
    if def_index >= 0:
        return stripped[def_index:].strip()

    return stripped


def normalize_code(text: str) -> str:
    """Normalize code for comparison: extract, trim lines, collapse blank lines."""
    code = extract_python_code(text)
    lines = [line.rstrip() for line in code.splitlines()]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank

    return "\n".join(normalized)
