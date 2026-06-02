from __future__ import annotations

"""Generic async-eval queue and result JSONL helpers."""

import json
import time
from pathlib import Path
from typing import Any

from serl_launcher.async_eval.runtime import AsyncEvalRuntime
from serl_launcher.utils.jsonl import append_jsonl


def append_async_eval_request(
    async_eval: AsyncEvalRuntime,
    payload: dict[str, Any],
) -> None:
    """Append an eval request to the async-eval queue."""

    if (not async_eval.enabled) or async_eval.queue_path is None:
        return
    record = dict(payload)
    record["type"] = "eval"
    record.setdefault(
        "timestamp",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )
    append_jsonl(async_eval.queue_path, record)
    async_eval.triggered_count += 1


def append_async_eval_stop(async_eval: AsyncEvalRuntime) -> None:
    """Append a stop marker to the async-eval queue."""

    if (not async_eval.enabled) or async_eval.queue_path is None:
        return
    append_jsonl(
        async_eval.queue_path,
        {
            "type": "stop",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )


def load_completed_async_eval_indices(summary_jsonl: Path) -> set[int]:
    """Load the set of eval indices already written to the summary JSONL."""

    completed: set[int] = set()
    for payload in _load_jsonl_dict_records(summary_jsonl):
        eval_index_raw = payload.get("eval_index", None)
        if eval_index_raw is None and isinstance(payload.get("request"), dict):
            eval_index_raw = payload["request"].get("eval_index", None)
        try:
            if eval_index_raw is not None:
                completed.add(int(eval_index_raw))
        except Exception:
            continue
    return completed


def load_async_eval_queue(queue_file: Path) -> tuple[list[dict[str, Any]], bool]:
    """Load eval queue records and whether a stop marker has been observed."""

    records: list[dict[str, Any]] = []
    stop_requested = False
    for payload in _load_jsonl_dict_records(queue_file):
        record_type = str(payload.get("type", "eval")).strip().lower()
        if record_type == "stop":
            stop_requested = True
            continue
        if record_type == "eval":
            records.append(payload)
    return records, stop_requested


def load_new_async_eval_results(async_eval: AsyncEvalRuntime) -> list[dict[str, Any]]:
    """Load newly appended async-eval result records since the last poll."""

    summary_jsonl_path = async_eval.summary_jsonl_path
    if summary_jsonl_path is None or (not summary_jsonl_path.exists()):
        return []
    with summary_jsonl_path.open("r", encoding="utf-8") as fp:
        lines = fp.readlines()
    processed_lines = int(async_eval.processed_summary_lines)
    if processed_lines < 0 or processed_lines > len(lines):
        processed_lines = 0

    records: list[dict[str, Any]] = []
    for line in lines[processed_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    async_eval.processed_summary_lines = int(len(lines))
    return records


def summarize_async_eval_results(summary_jsonl_path: Path | None) -> dict[str, int]:
    """Count ok/failed result records in an async-eval summary JSONL."""

    if summary_jsonl_path is None or (not summary_jsonl_path.exists()):
        return {"ok": 0, "failed": 0, "total": 0}

    counts = {"ok": 0, "failed": 0, "total": 0}
    for payload in _load_jsonl_dict_records(summary_jsonl_path):
        counts["total"] += 1
        status = str(payload.get("status", "")).strip().lower()
        if status == "ok":
            counts["ok"] += 1
        else:
            counts["failed"] += 1
    return counts


def _load_jsonl_dict_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or (not path.exists()):
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records
