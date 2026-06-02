#!/usr/bin/env python3
"""Evaluate a saved LIBERO RLT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from examples.libero_rlt.config import parse_train_cfg
from examples.libero_rlt.eval_runner import run_eval


def _load_config(config_path: Path):
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as fp:
            return OmegaConf.create(json.load(fp))
    return OmegaConf.load(str(config_path))


def _default_config_path(run_dir: Path) -> Path:
    snapshot_path = run_dir / "async_eval_train_config.json"
    if snapshot_path.exists():
        return snapshot_path
    hydra_config_path = run_dir / ".hydra" / "config.yaml"
    if hydra_config_path.exists():
        return hydra_config_path
    raise FileNotFoundError(
        "Could not find async_eval_train_config.json or .hydra/config.yaml "
        f"under run dir: {run_dir}"
    )


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    return checkpoint_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LIBERO RLT checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-episode-idx", type=int, default=0)
    parser.add_argument("--max-env-steps-per-episode", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else _default_config_path(run_dir)
    checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = parse_train_cfg(_load_config(config_path))
    payload = torch.load(checkpoint_path, map_location="cpu")
    result = run_eval(
        cfg,
        checkpoint_payload=payload,
        episodes=int(args.episodes),
        deterministic=not bool(args.stochastic),
        start_episode_idx=int(args.start_episode_idx),
        max_env_steps_per_episode=args.max_env_steps_per_episode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
