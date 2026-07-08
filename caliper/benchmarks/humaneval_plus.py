"""HumanEval+ benchmark loader."""

from __future__ import annotations

import gzip
import io
import json
import re
import urllib.error
import urllib.request
from typing import Any

from caliper.benchmarks.base import BenchmarkInfo, BenchmarkRecord
from caliper.tasks.schema import TaskMetadata

HUMANEVAL_PLUS_VERSION = "humanevalplus_release-main"
HUMANEVAL_PLUS_URL = (
    "https://raw.githubusercontent.com/evalplus/humanevalplus_release/main/"
    "HumanEvalPlus.jsonl.gz"
)
HUMANEVAL_PLUS_INFO = BenchmarkInfo(
    name="humaneval_plus",
    version=HUMANEVAL_PLUS_VERSION,
    source_url=HUMANEVAL_PLUS_URL,
    num_tasks=164,
    license="Apache-2.0",
)


def _slug_task_id(original_id: str) -> str:
    """Convert HumanEval/0 style ids to CALIPER task_id format."""
    cleaned = original_id.lower().replace("/", "-").replace(" ", "-")
    cleaned = re.sub(r"[^a-z0-9_-]", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"he-{cleaned}"
    return cleaned


def _load_evalplus_dict() -> dict[str, Any]:
    try:
        from evalplus.data import get_human_eval_plus  # type: ignore[import-not-found]

        return dict(get_human_eval_plus())
    except ImportError:
        pass

    rows: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(HUMANEVAL_PLUS_URL, timeout=120) as response:
            raw_bytes = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        msg = (
            "Could not load HumanEval+ data. Install optional dependency "
            "'evalplus' or ensure network access to GitHub raw content."
        )
        raise RuntimeError(msg) from exc

    if HUMANEVAL_PLUS_URL.endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz_handle:
            payload_text = gz_handle.read().decode("utf-8")
    else:
        payload_text = raw_bytes.decode("utf-8")

    for line in payload_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if rows:
        keyed: dict[str, Any] = {}
        for row in rows:
            task_id = str(row.get("task_id", row.get("name", len(keyed))))
            keyed[task_id] = row
        return keyed

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        msg = "HumanEval+ payload is neither JSONL nor JSON object"
        raise RuntimeError(msg) from exc

    if not isinstance(payload, dict):
        msg = "HumanEval+ payload must be a JSON object keyed by task id"
        raise RuntimeError(msg)
    return payload


def load_humaneval_plus() -> tuple[BenchmarkInfo, list[BenchmarkRecord]]:
    """Load HumanEval+ tasks from evalplus or the official JSON snapshot."""
    raw = _load_evalplus_dict()
    records: list[BenchmarkRecord] = []

    for original_id in sorted(raw):
        item = raw[original_id]
        if isinstance(item, dict) and "prompt" in item:
            row = item
            original_id = str(row.get("task_id", original_id))
        else:
            row = item  # type: ignore[assignment]

        prompt = str(row["prompt"])
        canonical = str(row.get("canonical_solution", ""))
        test_block = str(row["test"])
        entry_point = str(row["entry_point"])
        slug = _slug_task_id(original_id)

        records.append(
            BenchmarkRecord(
                benchmark_id=slug,
                prompt=prompt,
                canonical_solution=canonical,
                tests=[test_block],
                language="python",
                source_benchmark=f"humaneval-plus:{HUMANEVAL_PLUS_VERSION}",
                harness="humaneval",
                tags=["humaneval-plus", "python", "function-completion"],
                metadata={
                    "original_id": original_id,
                    "entry_point": entry_point,
                    "dataset_version": HUMANEVAL_PLUS_VERSION,
                    "test_block": test_block,
                },
            )
        )

    info = BenchmarkInfo(
        name="humaneval_plus",
        version=HUMANEVAL_PLUS_VERSION,
        source_url=HUMANEVAL_PLUS_URL,
        num_tasks=len(records),
        license="Apache-2.0",
    )
    return info, records


def to_task_metadata(record: BenchmarkRecord) -> TaskMetadata:
    """Convert a HumanEval+ record to CALIPER TaskMetadata."""
    return TaskMetadata(
        task_id=record.benchmark_id,
        domain="executable_code_generation",
        input=record.prompt,
        expected_output=record.canonical_solution,
        tests=record.tests,
        language=record.language,
        source_benchmark=record.source_benchmark,
        tags=record.tags,
        difficulty=record.difficulty,  # type: ignore[arg-type]
        extra={
            "harness": record.harness,
            **record.metadata,
        },
    )
