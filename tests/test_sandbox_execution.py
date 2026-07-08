"""Tests for sandbox execution and pass@1 metrics."""

from __future__ import annotations

from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.pass_at_k import pass_at_1
from caliper.evaluation.sandbox import build_program, execute_python_program


def test_build_humaneval_program_includes_check():
    program = build_program(
        harness="humaneval",
        prompt="def add(a, b):\n",
        completion="    return a + b\n",
        tests=["\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n"],
        entry_point="add",
    )
    assert "check(add)" in program
    assert "return a + b" in program


def test_execute_correct_humaneval_solution_passes():
    program = build_program(
        harness="humaneval",
        prompt="def add(a, b):\n",
        completion="    return a + b\n",
        tests=["\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n"],
        entry_point="add",
    )
    result = execute_python_program(program)
    assert result.passed is True
    assert result.timed_out is False


def test_execute_incorrect_solution_fails():
    program = build_program(
        harness="humaneval",
        prompt="def add(a, b):\n",
        completion="    return a - b\n",
        tests=["\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n"],
        entry_point="add",
    )
    result = execute_python_program(program)
    assert result.passed is False


def test_pass_at_1_metric_on_wrapped_code():
    sample = EvaluationInput(
        prediction="```python\n    return a + b\n```",
        expected_output="    return a + b\n",
        tests=["\n\ndef check(candidate):\n    assert candidate(1, 2) == 3\n"],
        domain="executable_code_generation",
        metadata={"harness": "humaneval", "entry_point": "add", "prompt_stub": "def add(a, b):\n"},
    )
    record = pass_at_1(sample)
    assert record.value == 1.0
    assert record.success is True


def test_pass_at_1_mbpp_harness():
    sample = EvaluationInput(
        prediction="```python\ndef abs_value(n):\n    return n if n >= 0 else -n\n```",
        tests=["assert abs_value(-5) == 5", "assert abs_value(3) == 3"],
        domain="executable_code_generation",
        metadata={"harness": "mbpp"},
    )
    record = pass_at_1(sample)
    assert record.value == 1.0
