from __future__ import annotations

"""Generic async-eval runtime helpers."""

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Sequence


@dataclass
class AsyncEvalRuntime:
    enabled: bool = False
    every_episodes: int = 0
    queue_path: Path | None = None
    summary_jsonl_path: Path | None = None
    worker_log_path: Path | None = None
    worker_proc: subprocess.Popen | None = None
    worker_log_fp: IO[str] | None = None
    eval_checkpoint_dir: Path | None = None
    eval_checkpoint_keep: int = 0
    processed_summary_lines: int = 0
    triggered_count: int = 0
    worker_dead_reported: bool = False


def resolve_async_eval_path(path_value: Any, *, run_dir: Path) -> Path:
    """Resolve an async-eval artifact path relative to the run directory."""

    path = Path(str(path_value))
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def count_jsonl_lines(path: Path | None) -> int:
    """Count newline-delimited records in a JSONL file if it exists."""

    if path is None or (not path.exists()):
        return 0
    with path.open("r", encoding="utf-8") as fp:
        return sum(1 for _ in fp)


def launch_async_eval_worker_process(
    *,
    cmd: Sequence[str],
    worker_log_path: Path,
) -> tuple[subprocess.Popen, IO[str]]:
    """Launch an async-eval worker process and stream its logs to disk."""

    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log_fp = worker_log_path.open("a", encoding="utf-8")
    worker_proc = subprocess.Popen(
        list(cmd),
        stdout=worker_log_fp,
        stderr=subprocess.STDOUT,
    )
    return worker_proc, worker_log_fp


def check_async_eval_worker(async_eval: AsyncEvalRuntime, *, logger: logging.Logger) -> None:
    """Fail the caller if the async-eval worker exits before training is done."""

    if (not async_eval.enabled) or async_eval.worker_proc is None:
        return
    if async_eval.worker_dead_reported:
        return
    return_code = async_eval.worker_proc.poll()
    if return_code is None:
        return
    async_eval.worker_dead_reported = True
    message = (
        "Async eval worker exited early with returncode="
        f"{return_code}; see {async_eval.worker_log_path}"
    )
    logger.error(message)
    raise RuntimeError(message)


def wait_for_async_eval_worker(
    async_eval: AsyncEvalRuntime,
    *,
    logger: logging.Logger,
    poll_interval_sec: float = 5.0,
) -> int | None:
    """Wait for the async-eval worker to exit while emitting periodic heartbeats."""

    if (not async_eval.enabled) or async_eval.worker_proc is None:
        return None

    worker_proc = async_eval.worker_proc
    last_log_time = 0.0
    logger.info("Waiting for async eval worker to drain queued evaluations")
    try:
        while True:
            return_code = worker_proc.poll()
            if return_code is not None:
                return return_code
            now = time.time()
            if now - last_log_time >= 30.0:
                logger.info(
                    "Async eval worker still running; queue=%s summary=%s",
                    async_eval.queue_path,
                    async_eval.summary_jsonl_path,
                )
                last_log_time = now
            time.sleep(max(0.1, float(poll_interval_sec)))
    finally:
        if async_eval.worker_log_fp is not None:
            async_eval.worker_log_fp.close()
            async_eval.worker_log_fp = None
