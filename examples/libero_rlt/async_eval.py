"""Async eval helpers for the RLT training loop."""

from __future__ import annotations

import logging
import json
import sys
from pathlib import Path

from serl_launcher.async_eval import (
    AsyncEvalRuntime,
    append_async_eval_request,
    append_async_eval_stop,
    check_async_eval_worker,
    count_jsonl_lines,
    launch_async_eval_worker_process,
    load_new_async_eval_results,
    resolve_async_eval_path,
    summarize_async_eval_results,
    wait_for_async_eval_worker,
)

from examples.libero_rlt.config import (
    AsyncEvalConfig,
    LiberoRLTTrainConfig,
    cfg_to_log_payload,
)

# Re-export for convenience
__all__ = [
    "append_async_eval_request",
    "append_async_eval_stop",
    "check_async_eval_worker",
    "load_new_async_eval_results",
    "start_async_eval_worker",
    "summarize_async_eval_results",
    "wait_for_async_eval_worker",
]


def start_async_eval_worker(
    cfg: LiberoRLTTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> AsyncEvalRuntime:
    async_eval_cfg = cfg.training.async_eval
    if not async_eval_cfg.enabled:
        return AsyncEvalRuntime()

    every_episodes = int(async_eval_cfg.every_episodes)
    if every_episodes <= 0:
        raise ValueError("async_eval.every_episodes must be > 0 when enabled")

    queue_path = resolve_async_eval_path(async_eval_cfg.queue_file, run_dir=run_dir)
    summary_jsonl_path = resolve_async_eval_path(async_eval_cfg.summary_jsonl, run_dir=run_dir)
    worker_log_path = resolve_async_eval_path(async_eval_cfg.worker_log_file, run_dir=run_dir)
    eval_checkpoint_dir = resolve_async_eval_path(async_eval_cfg.checkpoint.dir, run_dir=run_dir)
    config_snapshot_path = run_dir / "async_eval_train_config.json"

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    summary_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path.write_text(
        json.dumps(cfg_to_log_payload(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    worker_script = Path(__file__).resolve().parent / "scripts" / "process_eval_queue.py"
    if not worker_script.exists():
        raise FileNotFoundError(f"Missing async eval worker script: {worker_script}")

    poll_interval_sec = float(async_eval_cfg.poll_interval_sec)

    cmd = [
        sys.executable,
        str(worker_script),
        "--train-run-dir", str(run_dir),
        "--train-config", str(config_snapshot_path.resolve()),
        "--queue-file", str(queue_path),
        "--summary-jsonl", str(summary_jsonl_path),
        "--poll-interval-sec", str(poll_interval_sec),
    ]

    worker_proc, worker_log_fp = launch_async_eval_worker_process(
        cmd=cmd,
        worker_log_path=worker_log_path,
    )
    logger.info(
        "Async eval worker started: pid=%s queue=%s summary=%s",
        worker_proc.pid, queue_path, summary_jsonl_path,
    )
    return AsyncEvalRuntime(
        enabled=True,
        every_episodes=every_episodes,
        queue_path=queue_path,
        summary_jsonl_path=summary_jsonl_path,
        worker_log_path=worker_log_path,
        worker_proc=worker_proc,
        worker_log_fp=worker_log_fp,
        eval_checkpoint_dir=eval_checkpoint_dir,
        eval_checkpoint_keep=int(async_eval_cfg.checkpoint.keep),
        processed_summary_lines=count_jsonl_lines(summary_jsonl_path),
    )
