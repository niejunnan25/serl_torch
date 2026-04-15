"""Environment factory helpers for LIBERO training/eval."""
from __future__ import annotations

import logging

from serl_torch.examples.libero.config import LiberoRunConfig

from .remote_task_env import RemoteLiberoTaskEnv
from .task_env import LiberoTaskEnv


def create_env(cfg: LiberoRunConfig, logger: logging.Logger):
    common_kwargs = dict(
        suite_name=cfg.task.suite_name,
        task_id=cfg.task.task_id,
        action_dim=cfg.env.action_dim,
        resolution=cfg.env.resolution,
        num_steps_wait=cfg.env.num_steps_wait,
        max_episode_steps=cfg.env.max_episode_steps,
        libero_root=cfg.libero_root,
        libero_config_dir=cfg.libero_config_dir,
        libero_datasets_root=cfg.libero_datasets_root,
        env_seed=cfg.env.seed,
        logger=logger,
    )
    if cfg.env.backend == "local":
        env = LiberoTaskEnv(**common_kwargs)
    elif cfg.env.backend == "remote":
        env = RemoteLiberoTaskEnv(
            host=cfg.env.remote.host,
            port=cfg.env.remote.port,
            timeout_sec=cfg.env.remote.timeout_sec,
            **common_kwargs,
        )
    else:
        raise ValueError(
            f"env.backend must be 'local' or 'remote', got {cfg.env.backend}"
        )

    if int(env.action_dim) <= 0:
        raise ValueError(f"LIBERO env action_dim must be positive, got {env.action_dim}")
    if int(env.step_limit) <= 0:
        raise ValueError(f"LIBERO env step_limit must be positive, got {env.step_limit}")
    return env
