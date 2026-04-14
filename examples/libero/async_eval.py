from __future__ import annotations

"""Local async-eval helpers for the LIBERO reference training loop."""

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from serl_launcher.utils.jsonl import append_jsonl

from serl_torch.examples.libero.config import AsyncEvalConfig
from serl_torch.examples.libero.config import LiberoTrainConfig


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


def _resolve_path(path_value: Any, *, run_dir: Path) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _count_jsonl_lines(path: Path | None) -> int:
    if path is None or (not path.exists()):
        return 0
    with open(path, "r", encoding="utf-8") as fp:
        return sum(1 for _ in fp)


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

    queue_path = _resolve_path(
        async_eval_cfg.queue_file,
        run_dir=run_dir,
    )
    summary_jsonl_path = _resolve_path(
        async_eval_cfg.summary_jsonl,
        run_dir=run_dir,
    )
    worker_log_path = _resolve_path(
        async_eval_cfg.worker_log_file,
        run_dir=run_dir,
    )
    eval_checkpoint_dir = _resolve_path(
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

    worker_log_fp = open(worker_log_path, "a", encoding="utf-8")
    worker_proc = subprocess.Popen(
        cmd,
        stdout=worker_log_fp,
        stderr=subprocess.STDOUT,
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
        processed_summary_lines=_count_jsonl_lines(summary_jsonl_path),
    )


def check_async_eval_worker(async_eval: AsyncEvalRuntime, *, logger: logging.Logger) -> None:
    if (not async_eval.enabled) or async_eval.worker_proc is None:
        return
    if async_eval.worker_dead_reported:
        return
    return_code = async_eval.worker_proc.poll()
    if return_code is None:
        return
    async_eval.worker_dead_reported = True
    logger.warning(
        "Async eval worker exited early with returncode=%s; see %s",
        return_code,
        async_eval.worker_log_path,
    )


def append_async_eval_request(
    async_eval: AsyncEvalRuntime,
    payload: dict[str, Any],
) -> None:
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
    if (not async_eval.enabled) or async_eval.queue_path is None:
        return
    append_jsonl(
        async_eval.queue_path,
        {
            "type": "stop",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )


def load_new_async_eval_results(async_eval: AsyncEvalRuntime) -> list[dict[str, Any]]:
    summary_jsonl_path = async_eval.summary_jsonl_path
    if summary_jsonl_path is None or (not summary_jsonl_path.exists()):
        return []
    with open(summary_jsonl_path, "r", encoding="utf-8") as fp:
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


def wait_for_async_eval_worker(
    async_eval: AsyncEvalRuntime,
    *,
    logger: logging.Logger,
    poll_interval_sec: float = 5.0,
) -> int | None:
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


def summarize_async_eval_results(
    summary_jsonl_path: Path | None,
) -> dict[str, int]:
    if summary_jsonl_path is None or (not summary_jsonl_path.exists()):
        return {"ok": 0, "failed": 0, "total": 0}

    counts = {"ok": 0, "failed": 0, "total": 0}
    with open(summary_jsonl_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            counts["total"] += 1
            status = str(payload.get("status", "")).strip().lower()
            if status == "ok":
                counts["ok"] += 1
            else:
                counts["failed"] += 1
    return counts
