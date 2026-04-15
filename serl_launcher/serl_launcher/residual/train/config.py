"""Residual-train-specific configuration helpers."""
from __future__ import annotations

import random

import numpy as np
from omegaconf import DictConfig

from serl_launcher.residual.action import ResidualActionTransform
from serl_launcher.residual.action_spec import resolve_action_mask
from serl_launcher.residual.action_spec import resolve_control_indices


def resolve_action_mask_from_cfg(
    cfg: DictConfig,
    *,
    full_action_dim: int,
) -> np.ndarray:
    action_mask_cfg = cfg.residual.get("action_mask", None)
    action_mask = (
        [bool(v) for v in action_mask_cfg] if action_mask_cfg is not None else None
    )
    return resolve_action_mask(
        full_action_dim=int(full_action_dim),
        action_mask=action_mask,
    )


def resolve_control_indices_from_cfg(
    cfg: DictConfig,
    *,
    full_action_dim: int,
) -> np.ndarray:
    action_mask = resolve_action_mask_from_cfg(
        cfg,
        full_action_dim=int(full_action_dim),
    )
    return resolve_control_indices(
        full_action_dim=int(full_action_dim),
        action_mask=action_mask,
    )


def build_residual_action_transform(
    *,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    full_action_dim: int,
    chunk_horizon: int,
    chunk_step_enabled: bool,
    clip_gripper: bool,
) -> ResidualActionTransform:
    return ResidualActionTransform(
        control_indices=np.asarray(control_indices, dtype=np.int64),
        limits=np.asarray(residual_limits, dtype=np.float32),
        full_action_dim=int(full_action_dim),
        chunk_horizon=int(chunk_horizon),
        chunk_step_enabled=bool(chunk_step_enabled),
        clip_gripper=bool(clip_gripper),
    )


def sample_probing_steps(probing_cfg: DictConfig, *, episode_horizon: int) -> int:
    if not bool(probing_cfg.get("enable_base_probing", False)):
        return 0

    alpha_cfg = probing_cfg.get("probing_alpha", None)
    if alpha_cfg is not None:
        alpha = float(np.clip(float(alpha_cfg), 0.0, 1.0))
        max_steps = int(max(0, round(alpha * float(max(0, int(episode_horizon))))))
        if max_steps <= 0:
            return 0
        return int(random.randint(0, max_steps))

    min_steps = int(probing_cfg.get("probing_min_steps", 0))
    max_steps = int(probing_cfg.get("probing_max_steps", 0))
    min_steps = max(0, min_steps)
    max_steps = max(0, max_steps)
    if max_steps < min_steps:
        min_steps, max_steps = max_steps, min_steps
    if max_steps == 0:
        return 0
    return int(random.randint(min_steps, max_steps))
