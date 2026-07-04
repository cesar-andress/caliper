"""Shared text normalization helpers for metrics."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip outer space."""
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> list[str]:
    return normalize_text(text).lower().split()
