"""MBPP benchmark loader (sanitized split, with MBPP+ fallback)."""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import urllib.request
from typing import Any

from caliper.benchmarks.base import BenchmarkInfo, BenchmarkRecord
from caliper.tasks.schema import TaskMetadata

MBPP_VERSION = "google-research-mbpp-sanitized-v1"
MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/"
    "mbpp.jsonl"
)
MBPP_PLUS_URL = (
    "https://raw.githubusercontent.com/evalplus/mbppplus_release/main/MBPPPlus.jsonl.gz"
)


def _slug_task_id(task_id: int | str) -> str:
    return f"mbpp-{int(task_id):04d}"


def _parse_mbpp_row(data: dict[str, Any]) -> BenchmarkRecord | None:
    """Parse one MBPP JSON object into a BenchmarkRecord."""
    task_id = data.get("task_id")
    if task_id is None:
        return None

    text = str(data.get("text", data.get("prompt", ""))).strip()
    code = str(data.get("code", data.get("canonical_solution", ""))).strip()
    raw_tests = data.get("test_list") or data.get("tests") or []
    test_list = [str(t).strip() for t in raw_tests if str(t).strip()]
    if not text or not test_list:
        return None

    difficulty_raw = data.get("difficulty")
    difficulty = None
    if isinstance(difficulty_raw, str) and difficulty_raw.lower() in {"easy", "medium", "hard"}:
        difficulty = difficulty_raw.lower()

    return BenchmarkRecord(
        benchmark_id=_slug_task_id(task_id),
        prompt=text,
        canonical_solution=code,
        tests=test_list,
        language="python",
        source_benchmark=f"mbpp:{MBPP_VERSION}",
        harness="mbpp",
        difficulty=difficulty,
        tags=["mbpp", "python", "program-synthesis"],
        metadata={
            "original_id": task_id,
            "dataset_version": MBPP_VERSION,
            "challenge_test_list": data.get("challenge_test_list", []),
        },
    )


def _load_jsonl_bytes(raw_bytes: bytes, *, gzipped: bool) -> list[dict[str, Any]]:
    if gzipped:
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz_handle:
            payload_text = gz_handle.read().decode("utf-8")
    else:
        payload_text = raw_bytes.decode("utf-8")

    rows: list[dict[str, Any]] = []
    for line in payload_text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_mbpp_jsonl() -> list[dict[str, Any]]:
    try:
        from evalplus.data import get_mbpp_plus  # type: ignore[import-not-found]

        return [dict(value) for value in get_mbpp_plus().values()]
    except ImportError:
        pass

    sources = [
        (MBPP_URL, False),
        (MBPP_PLUS_URL, True),
    ]
    errors: list[str] = []
    for url, gzipped in sources:
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return _load_jsonl_bytes(response.read(), gzipped=gzipped)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")

    msg = (
        "Could not load MBPP data. Install optional dependency 'evalplus' or ensure "
        f"network access. Attempts: {'; '.join(errors)}"
    )
    raise RuntimeError(msg)


def load_mbpp(*, split: str = "test") -> tuple[BenchmarkInfo, list[BenchmarkRecord]]:
    """Load MBPP tasks from EvalPlus, MBPP+, or the Google Research JSONL snapshot."""
    raw_rows = _load_mbpp_jsonl()
    records: list[BenchmarkRecord] = []

    for row in raw_rows:
        partition = row.get("partition")
        if split != "all" and partition not in {None, split}:
            continue
        parsed = _parse_mbpp_row(row)
        if parsed is not None:
            records.append(parsed)

    records.sort(key=lambda r: r.benchmark_id)
    info = BenchmarkInfo(
        name="mbpp",
        version=MBPP_VERSION,
        source_url=MBPP_URL,
        num_tasks=len(records),
        license="CC-BY-4.0",
    )
    return info, records


def mbpp_to_task_metadata(record: BenchmarkRecord) -> TaskMetadata:
    """Convert an MBPP record to CALIPER TaskMetadata."""
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
