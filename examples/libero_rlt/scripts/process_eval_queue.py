#!/usr/bin/env python3
"""Async eval worker for RLT training.

Polls a queue file for eval requests, loads checkpoints, runs evaluation
episodes, and writes results to a summary JSONL file.

Launched automatically by the learner when async_eval is enabled.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for _path in (REPO_ROOT, SERL_LAUNCHER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
REPO_PARENT = REPO_ROOT.parent

from omegaconf import OmegaConf

from examples.libero_rlt.config import parse_train_cfg
from examples.libero_rlt.eval_runner import run_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RLT async eval worker")
    parser.add_argument("--train-run-dir", type=str, required=True)
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--queue-file", type=str, required=True)
    parser.add_argument("--summary-jsonl", type=str, required=True)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    return parser.parse_args()


def read_queue_lines(queue_path: Path, processed: int) -> list[dict]:
    """Read new lines from the queue file."""
    if not queue_path.exists():
        return []
    lines = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < processed:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return lines


def main():
    args = parse_args()
    queue_path = Path(args.queue_file)
    summary_path = Path(args.summary_jsonl)
    config_path = Path(args.train_config)

    # Load training config
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    if config_path.suffix.lower() == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = OmegaConf.create(json.load(f))
    else:
        cfg_dict = OmegaConf.load(str(config_path))
    cfg = parse_train_cfg(cfg_dict)
    async_eval_cfg = cfg.training.async_eval

    processed_lines = 0
    logger.info("Eval worker started. Polling %s every %.1fs", queue_path, args.poll_interval_sec)

    while True:
        new_requests = read_queue_lines(queue_path, processed_lines)

        if not new_requests:
            time.sleep(args.poll_interval_sec)
            continue

        for request in new_requests:
            processed_lines += 1

            # Check for stop signal
            if request.get("type") == "stop":
                logger.info("Received stop signal. Exiting.")
                return

            checkpoint_path = request.get("checkpoint_path")
            if not checkpoint_path or not Path(checkpoint_path).exists():
                logger.warning("Checkpoint not found: %s", checkpoint_path)
                result = {
                    "status": "failed",
                    "eval_index": request.get("eval_index"),
                    "error": f"checkpoint not found: {checkpoint_path}",
                    "request": request,
                }
                with open(summary_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                continue

            # Load checkpoint
            try:
                checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
            except Exception as e:
                logger.error("Failed to load checkpoint %s: %s", checkpoint_path, e)
                result = {
                    "status": "failed",
                    "eval_index": request.get("eval_index"),
                    "error": str(e),
                    "request": request,
                }
                with open(summary_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                continue

            # Run evaluation
            try:
                eval_metrics = run_eval(
                    cfg,
                    checkpoint_payload=checkpoint_payload,
                    episodes=int(async_eval_cfg.episodes),
                    deterministic=bool(async_eval_cfg.deterministic),
                    start_episode_idx=int(async_eval_cfg.start_episode_idx),
                    max_env_steps_per_episode=async_eval_cfg.max_env_steps_per_episode,
                    eval_logger=logger,
                )
                result = {
                    "status": "ok",
                    "eval_index": request.get("eval_index"),
                    "request": request,
                    "eval/success_rate": eval_metrics["success_rate"],
                    "eval/mean_return": eval_metrics["mean_return"],
                    "eval/mean_steps": eval_metrics["mean_steps"],
                    "eval/episodes_run": eval_metrics["episodes_run"],
                }
            except Exception as e:
                logger.error("Eval failed for checkpoint %s: %s", checkpoint_path, e, exc_info=True)
                result = {
                    "status": "failed",
                    "eval_index": request.get("eval_index"),
                    "error": str(e),
                    "request": request,
                }

            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

            logger.info(
                "Eval result: episode=%s success_rate=%.3f",
                request.get("train_episode_id", "?"),
                result.get("eval/success_rate", 0),
            )


if __name__ == "__main__":
    main()
