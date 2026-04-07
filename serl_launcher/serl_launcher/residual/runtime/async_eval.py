"""Async evaluation watcher orchestration helpers."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Any, Dict, Optional

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


def _start_async_eval_watcher(
    *,
    watcher_path: Path,
    cfg: DictConfig,
    run_dir: Path,
    checkpoint_dir: Path,
    logger: logging.Logger,
) -> tuple[
    Optional[subprocess.Popen],
    Optional[IO[str]],
    Optional[Path],
    Optional[Path],
    Optional[Path],
]:
    async_eval_cfg = cfg.training.get("async_eval", None)
    if async_eval_cfg is None or (not bool(async_eval_cfg.get("enabled", False))):
        return None, None, None, None, None

    if not watcher_path.exists():
        logger.warning(
            "training.async_eval.enabled=true but watcher script is missing: %s",
            watcher_path,
        )
        return None, None, None, None, None

    train_cfg_path = run_dir / ".hydra" / "config.yaml"
    summary_jsonl_path = Path(
        str(async_eval_cfg.get("summary_jsonl", "async_eval_results.jsonl"))
    )
    if not summary_jsonl_path.is_absolute():
        summary_jsonl_path = run_dir / summary_jsonl_path
    queue_path = Path(str(async_eval_cfg.get("queue_file", "async_eval_queue.jsonl")))
    if not queue_path.is_absolute():
        queue_path = run_dir / queue_path

    cmd = [
        sys.executable,
        str(watcher_path),
        "--train-run-dir",
        str(run_dir),
        "--train-config",
        str(train_cfg_path),
        "--checkpoints-dir",
        str(checkpoint_dir),
        "--summary-jsonl",
        str(summary_jsonl_path),
        "--queue-file",
        str(queue_path),
    ]

    log_path = run_dir / "async_eval_watch.log"
    log_fp = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )

    logger.info(
        "Async eval watcher started: pid=%s log=%s summary=%s",
        proc.pid,
        log_path,
        summary_jsonl_path,
    )
    return proc, log_fp, log_path, summary_jsonl_path, queue_path


def _append_async_eval_request(queue_path: Path, payload: Dict[str, Any]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with open(queue_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stop_async_eval_watcher(
    proc: Optional[subprocess.Popen],
    log_fp: Optional[IO[str]],
    *,
    logger: logging.Logger,
) -> Optional[int]:
    return_code: Optional[int] = None
    try:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Async eval watcher did not exit in time; killing it"
                    )
                    proc.kill()
                    proc.wait(timeout=5.0)
            return_code = proc.returncode
    finally:
        if log_fp is not None:
            log_fp.close()
    return return_code


def _init_async_eval_tb_sync_state(
    summary_jsonl_path: Optional[Path],
) -> Dict[str, Any]:
    processed_lines = 0
    if summary_jsonl_path is not None and summary_jsonl_path.exists():
        try:
            with open(summary_jsonl_path, "r", encoding="utf-8") as f:
                processed_lines = sum(1 for _ in f)
        except Exception:
            processed_lines = 0
    return {"processed_lines": int(processed_lines)}


def _sync_async_eval_results_to_tb(
    tb_writer: SummaryWriter,
    *,
    summary_jsonl_path: Optional[Path],
    sync_state: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    if summary_jsonl_path is None or (not summary_jsonl_path.exists()):
        return

    try:
        with open(summary_jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:
        logger.debug("Failed to read async eval summary jsonl: %s", exc)
        return

    processed_lines = int(sync_state.get("processed_lines", 0))
    if processed_lines < 0:
        processed_lines = 0
    if processed_lines > len(lines):
        processed_lines = 0

    new_lines = lines[processed_lines:]
    if not new_lines:
        return

    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        train_env_step_raw = payload.get("train_env_step", None)
        train_episode_id_raw = payload.get("train_episode_id", None)
        try:
            if train_env_step_raw is None:
                continue
            train_env_step = int(train_env_step_raw)
        except Exception:
            continue
        train_episode_id: Optional[int]
        try:
            train_episode_id = (
                int(train_episode_id_raw) if train_episode_id_raw is not None else None
            )
        except Exception:
            train_episode_id = None

        status = str(payload.get("status", "")).lower()
        duration_sec = payload.get("duration_sec", None)
        return_code = payload.get("return_code", None)

        if isinstance(duration_sec, (int, float)):
            tb_writer.add_scalar(
                "async_eval/duration_sec", float(duration_sec), train_env_step
            )
            if train_episode_id is not None:
                tb_writer.add_scalar(
                    "async_eval_episode/duration_sec",
                    float(duration_sec),
                    train_episode_id,
                )
        if isinstance(return_code, (int, float)):
            tb_writer.add_scalar(
                "async_eval/return_code", float(return_code), train_env_step
            )
            if train_episode_id is not None:
                tb_writer.add_scalar(
                    "async_eval_episode/return_code",
                    float(return_code),
                    train_episode_id,
                )

        if status == "ok":
            tb_writer.add_scalar("async_eval/status_ok", 1.0, train_env_step)
            tb_writer.add_scalar("async_eval/status_failed", 0.0, train_env_step)
            if train_episode_id is not None:
                tb_writer.add_scalar(
                    "async_eval_episode/status_ok", 1.0, train_episode_id
                )
                tb_writer.add_scalar(
                    "async_eval_episode/status_failed", 0.0, train_episode_id
                )
            summary = payload.get("summary", None)
            if isinstance(summary, dict):
                success_rate = summary.get("success_rate", None)
                total_success = summary.get("total_success", None)
                episodes = summary.get("episodes", None)
                if isinstance(success_rate, (int, float)):
                    tb_writer.add_scalar(
                        "async_eval/success_rate", float(success_rate), train_env_step
                    )
                    if train_episode_id is not None:
                        tb_writer.add_scalar(
                            "async_eval_episode/success_rate",
                            float(success_rate),
                            train_episode_id,
                        )
                if isinstance(total_success, (int, float)):
                    tb_writer.add_scalar(
                        "async_eval/total_success", float(total_success), train_env_step
                    )
                    if train_episode_id is not None:
                        tb_writer.add_scalar(
                            "async_eval_episode/total_success",
                            float(total_success),
                            train_episode_id,
                        )
                if isinstance(episodes, (int, float)):
                    tb_writer.add_scalar(
                        "async_eval/eval_episodes", float(episodes), train_env_step
                    )
                    if train_episode_id is not None:
                        tb_writer.add_scalar(
                            "async_eval_episode/eval_episodes",
                            float(episodes),
                            train_episode_id,
                        )
        elif status == "failed":
            tb_writer.add_scalar("async_eval/status_ok", 0.0, train_env_step)
            tb_writer.add_scalar("async_eval/status_failed", 1.0, train_env_step)
            if train_episode_id is not None:
                tb_writer.add_scalar(
                    "async_eval_episode/status_ok", 0.0, train_episode_id
                )
                tb_writer.add_scalar(
                    "async_eval_episode/status_failed", 1.0, train_episode_id
                )
        elif status == "aborted":
            tb_writer.add_scalar("async_eval/status_ok", 0.0, train_env_step)
            tb_writer.add_scalar("async_eval/status_failed", 1.0, train_env_step)
            if train_episode_id is not None:
                tb_writer.add_scalar(
                    "async_eval_episode/status_ok", 0.0, train_episode_id
                )
                tb_writer.add_scalar(
                    "async_eval_episode/status_failed", 1.0, train_episode_id
                )

    sync_state["processed_lines"] = int(len(lines))
    tb_writer.flush()
