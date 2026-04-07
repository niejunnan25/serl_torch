"""Replay transition builders shared by residual pipelines."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from serl_launcher.data.normalizer import StateActionNormalizer


def normalize_step_action(
    action: np.ndarray,
    *,
    normalizer: Optional[StateActionNormalizer] = None,
) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if normalizer is None:
        return action_arr.astype(np.float32)
    return np.asarray(normalizer.normalize_action(action_arr), dtype=np.float32)


def build_stepchunk_transition(
    *,
    obs_core: Dict[str, Any],
    base_action: np.ndarray,
    actions: np.ndarray,
    reward: float,
    done: bool,
    alpha: float,
    episode_id: int,
    episode_step: int,
    normalizer: Optional[StateActionNormalizer] = None,
    base_action_norm: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if base_action_norm is None:
        base_action_norm = normalize_step_action(base_action_arr, normalizer=normalizer)

    return {
        "obs_core": deepcopy(obs_core),
        "base_action": base_action_arr,
        "base_action_norm": np.asarray(base_action_norm, dtype=np.float32).reshape(-1),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1),
        "rewards": np.float32(reward),
        "dones": bool(done),
        "alpha": np.float32(alpha),
        "episode_id": int(episode_id),
        "episode_step": int(episode_step),
    }


def build_step_transition(
    *,
    observations: Dict[str, Any],
    actions: np.ndarray,
    next_observations: Dict[str, Any],
    reward: float,
    done: bool,
    mask: float,
) -> Dict[str, Any]:
    return {
        "observations": deepcopy(observations),
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1),
        "next_observations": deepcopy(next_observations),
        "rewards": np.float32(reward),
        "masks": np.float32(mask),
        "dones": bool(done),
    }
