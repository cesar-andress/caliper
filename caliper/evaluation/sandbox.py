"""Isolated Python code execution for functional evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

HarnessType = Literal["humaneval", "mbpp", "generic"]


@dataclass(frozen=True)
class ExecutionLimits:
    """Resource limits for sandboxed execution."""

    timeout_seconds: float = 5.0
    memory_mb: int = 512


@dataclass
class ExecutionResult:
    """Outcome of running generated code against unit tests."""

    passed: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def build_program(
    *,
    harness: HarnessType,
    prompt: str | None,
    completion: str,
    tests: list[str],
    entry_point: str | None = None,
) -> str:
    """Assemble a runnable Python program for the given benchmark harness."""
    completion_body = completion.rstrip()
    if not completion_body.strip():
        return ""

    if harness == "humaneval":
        prefix = prompt or ""
        test_block = tests[0] if tests else ""
        ep = entry_point or "candidate"
        if completion_body.lstrip().startswith("def "):
            program_prefix = ""
            body = completion_body
        else:
            program_prefix = prefix
            body = completion_body
            if body and not body.startswith((" ", "\t")):
                body = f"    {body.lstrip()}"
        return (
            f"{program_prefix}{body}\n"
            f"{test_block}\n"
            f"if __name__ == '__main__':\n"
            f"    check({ep})\n"
        )

    if harness == "mbpp":
        body = completion_body
        if not body.lstrip().startswith("def ") and prompt:
            body = f"{completion_body}\n"
        test_lines = "\n".join(tests)
        return f"{body}\n{test_lines}\n"

    test_lines = "\n".join(tests)
    if prompt:
        return f"{prompt}\n{completion_body}\n{test_lines}\n"
    return f"{completion_body}\n{test_lines}\n"


def _apply_memory_limit(memory_mb: int) -> None:
    """Apply a soft memory limit in child processes (Linux only)."""
    if sys.platform != "linux":
        return
    try:
        import resource

        limit_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ImportError, OSError, ValueError):
        return


def execute_python_program(
    program: str,
    *,
    limits: ExecutionLimits | None = None,
) -> ExecutionResult:
    """Execute Python code in an isolated subprocess with timeout and limits."""
    import time

    if not program.strip():
        return ExecutionResult(
            passed=False,
            error_type="empty_program",
            error_message="No executable program was produced",
        )

    limits = limits or ExecutionLimits()
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", tempfile.gettempdir()),
        "LANG": "C.UTF-8",
    }

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="caliper_exec_") as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(program, encoding="utf-8")

        cmd = [sys.executable, "-I", "-S", str(script_path)]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=limits.timeout_seconds,
                env=env,
                cwd=tmpdir,
                preexec_fn=lambda: _apply_memory_limit(limits.memory_mb),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ExecutionResult(
                passed=False,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                timed_out=True,
                error_type="timeout",
                error_message=f"Execution exceeded {limits.timeout_seconds}s",
                latency_ms=latency_ms,
            )
        except OSError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ExecutionResult(
                passed=False,
                error_type="os_error",
                error_message=str(exc),
                latency_ms=latency_ms,
            )

    latency_ms = (time.perf_counter() - start) * 1000.0
    passed = completed.returncode == 0
    error_type = None if passed else "execution_failure"
    error_message = None if passed else (completed.stderr.strip() or "non-zero exit code")

    return ExecutionResult(
        passed=passed,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        timed_out=False,
        error_type=error_type,
        error_message=error_message,
        latency_ms=latency_ms,
        metadata={"script_bytes": len(program.encode("utf-8"))},
    )


def execute_sample(
    *,
    harness: HarnessType,
    prompt: str | None,
    completion: str,
    tests: list[str],
    entry_point: str | None = None,
    limits: ExecutionLimits | None = None,
) -> ExecutionResult:
    """Build and execute a candidate solution against benchmark tests."""
    program = build_program(
        harness=harness,
        prompt=prompt,
        completion=completion,
        tests=tests,
        entry_point=entry_point,
    )
    return execute_python_program(program, limits=limits)
