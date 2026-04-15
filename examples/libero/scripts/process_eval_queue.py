from __future__ import annotations

"""Process queued LIBERO async-eval requests."""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
SERL_LAUNCHER_ROOT = REPO_PARENT / "serl_torch" / "serl_launcher"
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_launcher.utils.jsonl import append_jsonl
from serl_launcher.async_eval import format_async_eval_run_dir_name
from serl_launcher.async_eval import load_async_eval_queue
from serl_launcher.async_eval import load_completed_async_eval_indices
from serl_torch.examples.libero.config import EvalConfig
from serl_torch.examples.libero.config import LiberoEvalConfig
from serl_torch.examples.libero.config import LiberoTrainConfig
from serl_torch.examples.libero.config import LoggingConfig
from serl_torch.examples.libero.config import parse_train_cfg
from serl_torch.examples.libero.config import train_cfg_to_eval_cfg
from serl_torch.examples.libero.eval_runner import run_eval


LOGGER = logging.getLogger("libero_async_eval_worker")


def _build_eval_cfg(
    train_cfg: LiberoTrainConfig,
    request: dict[str, Any],
) -> LiberoEvalConfig:
    async_eval_cfg = train_cfg.training.async_eval
    return train_cfg_to_eval_cfg(
        train_cfg,
        eval_cfg=EvalConfig(
            episodes=int(async_eval_cfg.episodes),
            start_episode_idx=int(async_eval_cfg.start_episode_idx),
            max_env_steps_per_episode=async_eval_cfg.max_env_steps_per_episode,
            deterministic=bool(async_eval_cfg.deterministic),
            checkpoint_path=str(request["checkpoint_path"]),
            checkpoint_step=int(request["checkpoint_step"]),
        ),
        env_override=async_eval_cfg.env,
        logging=LoggingConfig(
            summary_file="summary.json",
            episode_log_file="episode_logs.jsonl",
        ),
    )


def _process_one_request(
    *,
    train_cfg: LiberoTrainConfig,
    train_run_dir: Path,
    summary_jsonl: Path,
    request: dict[str, Any],
) -> None:
    eval_index = int(request["eval_index"])
    train_episode_id_raw = request.get("train_episode_id", None)
    train_episode_id = (
        None if train_episode_id_raw is None else int(train_episode_id_raw)
    )
    checkpoint_step = int(request["checkpoint_step"])
    train_update_step = int(request["train_update_step"])
    train_env_step = int(request["train_env_step"])
    checkpoint_path = str(request["checkpoint_path"])
    eval_dir_name = format_async_eval_run_dir_name(
        eval_index=int(eval_index),
        checkpoint_step=int(checkpoint_step),
        train_episode_id=train_episode_id,
    )
    eval_run_dir = (train_run_dir / "async_eval_runs" / eval_dir_name).resolve()
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    status = "ok"
    summary: dict[str, Any] | None = None
    error_payload: dict[str, Any] = {}

    try:
        eval_cfg = _build_eval_cfg(train_cfg, request)
        eval_logger = logging.getLogger(f"libero_async_eval_worker.eval_{eval_index}")
        summary = run_eval(
            eval_cfg,
            run_dir=eval_run_dir,
            logger=eval_logger,
            original_cwd=train_run_dir,
        )
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    completed_at = time.time()
    record: dict[str, Any] = {
        "status": status,
        "eval_index": int(eval_index),
        "train_update_step": int(train_update_step),
        "train_env_step": int(train_env_step),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_path": str(checkpoint_path),
        "eval_run_dir": str(eval_run_dir),
        "duration_sec": float(completed_at - started_at),
        "completed_timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(completed_at),
        ),
    }
    if train_episode_id is not None:
        record["train_episode_id"] = int(train_episode_id)
    if summary is not None:
        record["summary"] = summary
    record.update(error_payload)
    append_jsonl(summary_jsonl, record)
    LOGGER.info(
        "Processed eval_index=%s episode=%s checkpoint_step=%s status=%s duration=%.2fs",
        eval_index,
        train_episode_id,
        checkpoint_step,
        status,
        float(completed_at - started_at),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run-dir", type=str, required=True)
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--queue-file", type=str, required=True)
    parser.add_argument("--summary-jsonl", type=str, required=True)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )

    train_run_dir = Path(args.train_run_dir).expanduser().resolve()
    train_cfg_path = Path(args.train_config).expanduser().resolve()
    queue_file = Path(args.queue_file).expanduser().resolve()
    summary_jsonl = Path(args.summary_jsonl).expanduser().resolve()
    poll_interval_sec = max(0.1, float(args.poll_interval_sec))

    train_cfg = parse_train_cfg(OmegaConf.load(train_cfg_path))
    LOGGER.info(
        "Eval worker started: train_run_dir=%s queue=%s summary=%s",
        train_run_dir,
        queue_file,
        summary_jsonl,
    )

    while True:
        completed_eval_indices = load_completed_async_eval_indices(summary_jsonl)
        queue_records, stop_requested = load_async_eval_queue(queue_file)
        pending_records = []
        for record in queue_records:
            eval_index_raw = record.get("eval_index", None)
            try:
                eval_index = int(eval_index_raw)
            except Exception:
                continue
            if eval_index not in completed_eval_indices:
                pending_records.append(record)

        if pending_records:
            _process_one_request(
                train_cfg=train_cfg,
                train_run_dir=train_run_dir,
                summary_jsonl=summary_jsonl,
                request=pending_records[0],
            )
            continue

        if stop_requested:
            LOGGER.info("Eval worker received stop signal and queue is drained")
            break

        time.sleep(poll_interval_sec)


if __name__ == "__main__":
    main()
