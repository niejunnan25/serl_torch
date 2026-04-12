from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig


def set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def resolve_runtime_role(value: Any) -> str:
    role = str(value).strip().lower()
    if role not in {"actor", "learner"}:
        raise ValueError(f"runtime.role must be 'actor' or 'learner', got {value!r}")
    return role


def validate_residual_cfg(cfg: DictConfig) -> str:
    role = resolve_runtime_role(cfg.runtime.role)

    if int(cfg.env.action_dim) <= 0:
        raise ValueError(f"env.action_dim must be positive, got {cfg.env.action_dim}")
    env_seed = cfg.env.get("seed", None)
    if env_seed is None:
        raise ValueError("env.seed must be explicitly set")
    if int(cfg.training.training_starts) < 0:
        raise ValueError(
            f"training.training_starts must be >= 0, got {cfg.training.training_starts}"
        )
    if int(cfg.training.steps_per_update) <= 0:
        raise ValueError(
            f"training.steps_per_update must be positive, got {cfg.training.steps_per_update}"
        )
    if int(cfg.training.critic_actor_ratio) <= 0:
        raise ValueError(
            "training.critic_actor_ratio must be positive, "
            f"got {cfg.training.critic_actor_ratio}"
        )
    if int(cfg.training.log_period) <= 0:
        raise ValueError(
            f"training.log_period must be positive, got {cfg.training.log_period}"
        )
    if int(cfg.training.max_env_steps) <= 0:
        raise ValueError(
            f"training.max_env_steps must be positive, got {cfg.training.max_env_steps}"
        )
    if int(cfg.training.max_update_steps) <= 0:
        raise ValueError(
            "training.max_update_steps must be positive, "
            f"got {cfg.training.max_update_steps}"
        )
    if int(cfg.obs.stack_horizon) != 1:
        raise ValueError(
            "LIBERO residual DRQ currently supports only obs.stack_horizon=1"
        )

    residual_alpha = cfg.residual.get("alpha", None)
    if residual_alpha is None:
        raise ValueError("residual.alpha must be explicitly set")
    residual_alpha = float(residual_alpha)
    if (not math.isfinite(residual_alpha)) or residual_alpha < 0.0:
        raise ValueError(
            f"residual.alpha must be finite and >= 0.0, got {cfg.residual.get('alpha')!r}"
        )

    chunk_horizon = int(cfg.residual.get("chunk_horizon", 1))
    if chunk_horizon <= 0:
        raise ValueError(
            f"residual.chunk_horizon must be positive, got {cfg.residual.get('chunk_horizon')!r}"
        )

    action_mask_cfg = cfg.residual.get("action_mask", None)
    if action_mask_cfg is not None:
        action_mask = np.asarray([bool(v) for v in action_mask_cfg], dtype=bool)
        if action_mask.size != int(cfg.env.action_dim):
            raise ValueError(
                "residual.action_mask length mismatch: "
                f"got {int(action_mask.size)}, expected env.action_dim={int(cfg.env.action_dim)}"
            )
        if not np.any(action_mask):
            raise ValueError("residual.action_mask must enable at least one action dim")

    action_limits_cfg = cfg.residual.get("action_limits", None)
    if action_limits_cfg is None:
        raise ValueError("residual.action_limits must be explicitly set")
    action_limits = np.asarray(action_limits_cfg, dtype=np.float32).reshape(-1)
    if action_limits.size != int(cfg.env.action_dim):
        raise ValueError(
            "residual.action_limits length mismatch: "
            f"got {int(action_limits.size)}, expected env.action_dim={int(cfg.env.action_dim)}"
        )
    if np.any(~np.isfinite(action_limits)) or np.any(action_limits < 0.0):
        raise ValueError(
            f"residual.action_limits must be finite and >= 0.0, got {action_limits.tolist()}"
        )

    return role
