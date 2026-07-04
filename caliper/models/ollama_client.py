"""HTTP client for the local Ollama API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _request_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaHttpError(exc.code, detail, url=url) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise OllamaConnectionError(str(reason), url=url) from exc

    if not body.strip():
        return {}
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        msg = f"Ollama returned non-object JSON from {url}"
        raise OllamaResponseError(msg)
    return parsed


class OllamaClientError(Exception):
    """Base error for Ollama HTTP client failures."""


class OllamaConnectionError(OllamaClientError):
    """Raised when Ollama is unreachable."""

    def __init__(self, message: str, *, url: str) -> None:
        self.url = url
        super().__init__(message)


class OllamaHttpError(OllamaClientError):
    """Raised for non-success HTTP responses from Ollama."""

    def __init__(self, status_code: int, body: str, *, url: str) -> None:
        self.status_code = status_code
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {body}")


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns an unexpected payload."""


def list_models(*, base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
    """Return model metadata from ``GET /api/tags``."""
    payload = _request_json(
        method="GET",
        url=_join_url(base_url, "/api/tags"),
        timeout_seconds=timeout_seconds,
    )
    models = payload.get("models", [])
    if not isinstance(models, list):
        msg = "Ollama /api/tags response missing 'models' list"
        raise OllamaResponseError(msg)
    return [model for model in models if isinstance(model, dict)]


def generate(
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    stop: list[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Call ``POST /api/generate`` and return the parsed JSON response."""
    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed
    if stop:
        options["stop"] = stop

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    payload = _request_json(
        method="POST",
        url=_join_url(base_url, "/api/generate"),
        payload=body,
        timeout_seconds=timeout_seconds,
    )
    if payload.get("error"):
        raise OllamaResponseError(str(payload["error"]))
    return payload
