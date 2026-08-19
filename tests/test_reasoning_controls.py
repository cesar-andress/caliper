"""Tests for Ollama reasoning-model controls and budget_exhausted status."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from caliper.config.schema import DecodingConfig, ExperimentConfig, ModelConfig, OutputConfig
from caliper.models.ollama_client import resolve_think_flag
from caliper.models.types import ModelRequest, ModelResponse
from caliper.runners.executor import resolve_think_mode, execute_cell
from caliper.runners.results import ExperimentResultRecord, FINISHED_STATUSES


def test_resolve_think_flag_mapping() -> None:
    assert resolve_think_flag("auto") is None
    assert resolve_think_flag(None) is None
    assert resolve_think_flag(True) is True
    assert resolve_think_flag("true") is True
    assert resolve_think_flag(False) is False
    assert resolve_think_flag("false") is False


def test_resolve_think_mode_prefers_model() -> None:
    model = ModelConfig(id="m", provider="p", model_id="qwen3:32b", think=False)
    decoding = DecodingConfig(think="true")
    config = ExperimentConfig.model_construct(
        experiment_id="t",
        models=[model],
        tasks=[],
        providers={},
        decoding=decoding,
        output=OutputConfig(),
    )
    assert resolve_think_mode(model, config) is False


def test_finished_statuses_include_budget_exhausted() -> None:
    assert "budget_exhausted" in FINISHED_STATUSES


def test_save_raw_response_uses_run_output_dir(tmp_path: Path) -> None:
    from caliper.runners.executor import _save_raw_response

    config = ExperimentConfig.model_construct(
        experiment_id="t",
        models=[],
        tasks=[],
        providers={},
        decoding=DecodingConfig(),
        output=OutputConfig(directory=str(tmp_path / "config_parent"), save_raw_responses=True),
    )
    run_dir = tmp_path / "actual_run"
    response = ModelResponse(
        text="print(1)",
        model_name="qwen3:32b",
        provider_name="ollama_local",
        prompt_id="minimal",
        task_id="t",
        run_id="r",
        temperature=0.0,
        latency_ms=1.0,
        done_reason="stop",
        completion_tokens=10,
        prompt_tokens=5,
        thinking_length=0,
        budget_exhausted=False,
        raw_metadata={"ollama": {"done_reason": "stop"}},
    )
    path = _save_raw_response(
        output_dir=run_dir,
        config=config,
        cell_id="cellabc",
        response=response,
    )
    assert path is not None
    assert path.parent == run_dir / "raw_responses"
    assert path.exists()
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert payload["done_reason"] == "stop"
    assert payload["raw_metadata"]["ollama"]["done_reason"] == "stop"
    assert not (tmp_path / "config_parent" / "raw_responses").exists()


def test_budget_exhausted_record_roundtrip(tmp_path: Path) -> None:
    record = ExperimentResultRecord(
        cell_id="c1",
        experiment_id="e",
        run_id="r",
        run_index=0,
        model_id="qwen3_32b",
        provider_name="ollama_local",
        provider_type="ollama",
        task_id="t",
        prompt_variant_id="minimal",
        temperature=0.0,
        metric="pass_at_1",
        score=0.0,
        status="budget_exhausted",
        error="Budget Exhausted",
        done_reason="length",
        eval_count=1024,
        thinking_length=100,
    )
    path = tmp_path / "results.jsonl"
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    loaded = ExperimentResultRecord.model_validate_json(path.read_text().strip())
    assert loaded.status == "budget_exhausted"
    assert loaded.eval_count == 1024
