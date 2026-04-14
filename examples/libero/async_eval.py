from __future__ import annotations

"""Local async-eval helpers for the LIBERO reference training loop."""

import logging
import sys
from pathlib import Path

from serl_launcher.async_eval import append_async_eval_request
from serl_launcher.async_eval import append_async_eval_stop
from serl_launcher.async_eval import AsyncEvalRuntime
from serl_launcher.async_eval import check_async_eval_worker
from serl_launcher.async_eval import count_jsonl_lines
from serl_launcher.async_eval import launch_async_eval_worker_process
from serl_launcher.async_eval import load_new_async_eval_results
from serl_launcher.async_eval import resolve_async_eval_path
from serl_launcher.async_eval import summarize_async_eval_results
from serl_launcher.async_eval import wait_for_async_eval_worker

from serl_torch.examples.libero.config import AsyncEvalConfig
from serl_torch.examples.libero.config import LiberoTrainConfig


def _validate_async_eval_env(
    cfg: LiberoTrainConfig,
    *,
    async_eval_cfg: AsyncEvalConfig,
) -> None:
    if async_eval_cfg.env.backend != "remote":
        raise ValueError(
            "training.async_eval.env.backend must be 'remote' so eval runs on a dedicated env server"
        )

    if cfg.env.backend == "remote":
        train_host = str(cfg.env.remote.host).strip()
        train_port = int(cfg.env.remote.port)
        async_host = str(async_eval_cfg.env.remote.host).strip()
        async_port = int(async_eval_cfg.env.remote.port)
        if train_host == async_host and train_port == async_port:
            raise ValueError(
                "training.async_eval.env.remote must point to a dedicated env server; "
                "it currently matches env.remote.host/port"
            )


def start_async_eval_worker(
    cfg: LiberoTrainConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
) -> AsyncEvalRuntime:
    async_eval_cfg = cfg.training.async_eval
    if not async_eval_cfg.enabled:
        return AsyncEvalRuntime()

    every_episodes = int(async_eval_cfg.every_episodes)
    if every_episodes <= 0:
        raise ValueError(
            "training.async_eval.enabled=true requires training.async_eval.every_episodes > 0"
        )

    _validate_async_eval_env(cfg, async_eval_cfg=async_eval_cfg)

    poll_interval_sec = float(async_eval_cfg.poll_interval_sec)
    if poll_interval_sec <= 0.0:
        raise ValueError(
            "training.async_eval.poll_interval_sec must be positive, "
            f"got {poll_interval_sec}"
        )

    queue_path = resolve_async_eval_path(
        async_eval_cfg.queue_file,
        run_dir=run_dir,
    )
    summary_jsonl_path = resolve_async_eval_path(
        async_eval_cfg.summary_jsonl,
        run_dir=run_dir,
    )
    worker_log_path = resolve_async_eval_path(
        async_eval_cfg.worker_log_file,
        run_dir=run_dir,
    )
    eval_checkpoint_dir = resolve_async_eval_path(
        async_eval_cfg.checkpoint.dir,
        run_dir=run_dir,
    )
    eval_checkpoint_keep = int(async_eval_cfg.checkpoint.keep)
    if eval_checkpoint_keep < 0:
        raise ValueError(
            "training.async_eval.checkpoint.keep must be >= 0, "
            f"got {eval_checkpoint_keep}"
        )
    if eval_checkpoint_keep != 0:
        raise ValueError(
            "training.async_eval.checkpoint.keep must currently be 0 so queued "
            "eval checkpoints are not deleted before the worker processes them"
        )

    worker_script = Path(__file__).resolve().parent / "scripts" / "process_eval_queue.py"
    if not worker_script.exists():
        raise FileNotFoundError(f"Missing async eval worker script: {worker_script}")

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    summary_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(worker_script),
        "--train-run-dir",
        str(run_dir),
        "--train-config",
        str((run_dir / ".hydra" / "config.yaml").resolve()),
        "--queue-file",
        str(queue_path),
        "--summary-jsonl",
        str(summary_jsonl_path),
        "--poll-interval-sec",
        str(poll_interval_sec),
    ]

    worker_proc, worker_log_fp = launch_async_eval_worker_process(
        cmd=cmd,
        worker_log_path=worker_log_path,
    )
    logger.info(
        "Async eval worker started: pid=%s queue=%s summary=%s log=%s",
        worker_proc.pid,
        queue_path,
        summary_jsonl_path,
        worker_log_path,
    )
    return AsyncEvalRuntime(
        enabled=True,
        every_episodes=int(every_episodes),
        queue_path=queue_path,
        summary_jsonl_path=summary_jsonl_path,
        worker_log_path=worker_log_path,
        worker_proc=worker_proc,
        worker_log_fp=worker_log_fp,
        eval_checkpoint_dir=eval_checkpoint_dir,
        eval_checkpoint_keep=int(eval_checkpoint_keep),
        processed_summary_lines=count_jsonl_lines(summary_jsonl_path),
    )
