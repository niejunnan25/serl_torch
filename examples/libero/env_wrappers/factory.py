"""Environment factory helpers for LIBERO training/eval."""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from .remote_task_env import RemoteLiberoTaskEnv
from .task_env import LiberoTaskEnv


def _create_env(cfg: DictConfig, logger: logging.Logger):
    env_backend = str(cfg.get("env", {}).get("backend", "remote")).lower()
    common_kwargs = dict(
        suite_name=str(cfg.task.suite_name),
        task_id=int(cfg.task.task_id),
        action_dim=cfg.get("env", {}).get("action_dim", None),
        resolution=int(cfg.task.resolution),
        num_steps_wait=int(cfg.task.num_steps_wait),
        max_episode_steps=(
            int(cfg.task.max_episode_steps) if cfg.task.max_episode_steps is not None else None
        ),
        libero_root=cfg.get("libero_root", None),
        openpi_root=cfg.get("openpi_root", None),
        libero_config_dir=cfg.get("libero_config_dir", None),
        libero_datasets_root=cfg.get("libero_datasets_root", None),
        env_seed_mode=str(cfg.task.get("env_seed_mode", "per_episode")),
        fixed_env_seed=cfg.task.get("fixed_env_seed", None),
        init_state_index_mode=str(cfg.task.get("init_state_index_mode", "seed")),
        logger=logger,
    )
    if env_backend == "local":
        return LiberoTaskEnv(**common_kwargs)
    if env_backend == "remote":
        remote_cfg = cfg.get("env", {}).get("remote", {})
        return RemoteLiberoTaskEnv(
            host=str(remote_cfg.get("host", "127.0.0.1")),
            port=int(remote_cfg.get("port", 30000)),
            timeout_sec=float(remote_cfg.get("timeout_sec", 120.0)),
            **common_kwargs,
        )
    raise ValueError(f"env.backend must be 'local' or 'remote', got {env_backend}")
