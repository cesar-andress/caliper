"""Configuration loading and validation errors."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError


class ConfigError(Exception):
    """Base exception for configuration errors."""


class ConfigValidationError(ConfigError):
    """Raised when a YAML config fails schema validation."""

    def __init__(self, path: Path | None, errors: list[str]) -> None:
        self.path = path
        self.errors = errors
        super().__init__(format_validation_errors(path, errors))


class ConfigParseError(ConfigError):
    """Raised when a YAML file cannot be parsed."""


def format_validation_errors(path: Path | None, errors: list[str]) -> str:
    """Format validation errors into a human-readable multi-line message."""
    header = f"Invalid experiment config: {path}" if path else "Invalid experiment config"
    body = "\n".join(f"  - {err}" for err in errors)
    return f"{header}\n{body}"


def pydantic_errors_to_messages(exc: ValidationError) -> list[str]:
    """Convert a Pydantic ValidationError into plain-language messages."""
    messages: list[str] = []
    for error in exc.errors():
        loc = _format_location(error["loc"])
        msg = error["msg"]
        if error["type"] == "missing":
            messages.append(f"{loc}: required field is missing")
        elif error["type"] == "value_error":
            messages.append(f"{loc}: {msg.removeprefix('Value error, ')}")
        else:
            messages.append(f"{loc}: {msg}")
    return messages


def _format_location(loc: tuple[str | int, ...]) -> str:
    if not loc:
        return "(root)"
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts[-1] = f"{parts[-1]}[{item}]"
        else:
            parts.append(str(item))
    return ".".join(parts)
