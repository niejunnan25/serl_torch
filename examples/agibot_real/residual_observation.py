"""Observation-schema helpers for AgiBot residual training."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from .env.observation import AGIBOT_STATE_DIM
from .env.observation import RESIDUAL_IMAGE_HEIGHT
from .env.observation import RESIDUAL_IMAGE_WIDTH
from .env.observation import build_agibot_state
from .env.observation import extract_agibot_residual_images
from .schema import resolve_agibot_image_keys


def prepare_base_actions_chunk(
    *,
    base_actions: np.ndarray,
    chunk_horizon: int,
    source: str = "base policy",
) -> np.ndarray:
    base_actions_array = np.asarray(base_actions, dtype=np.float32)
    if base_actions_array.ndim != 2:
        raise ValueError(
            f"{source} must return a 2D action chunk, got shape {base_actions_array.shape}"
        )
    if int(base_actions_array.shape[0]) < int(chunk_horizon):
        raise ValueError(
            f"{source} returned only {int(base_actions_array.shape[0])} actions, "
            f"expected at least chunk_horizon={int(chunk_horizon)}"
        )
    return np.asarray(base_actions_array[: int(chunk_horizon)], dtype=np.float32)


def build_chunk_residual_obs(
    *,
    obs: dict[str, Any],
    base_actions: np.ndarray,
    image_keys: tuple[str, ...],
    residual_alpha: float,
) -> dict[str, np.ndarray]:
    base_actions_array = np.asarray(base_actions, dtype=np.float32)
    resolved_image_keys = resolve_agibot_image_keys(image_keys)
    images = extract_agibot_residual_images(obs, image_keys=resolved_image_keys)
    residual_obs: dict[str, np.ndarray] = {
        "robot_proprio": np.expand_dims(build_agibot_state(obs), axis=0).astype(
            np.float32
        ),
        "base_action": np.expand_dims(base_actions_array[0], axis=0).astype(np.float32),
        "base_action_chunk": np.expand_dims(base_actions_array, axis=0).astype(
            np.float32
        ),
        "alpha": np.asarray([[residual_alpha]], dtype=np.float32),
    }
    for key in resolved_image_keys:
        residual_obs[key] = np.expand_dims(images[key].copy(), axis=0)
    return residual_obs


def build_chunk_residual_sample_obs(
    *,
    action_dim: int,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if int(action_dim) <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if int(chunk_horizon) <= 0:
        raise ValueError(f"chunk_horizon must be positive, got {chunk_horizon}")

    sample_obs: dict[str, np.ndarray] = {
        "robot_proprio": np.zeros((1, AGIBOT_STATE_DIM), dtype=np.float32),
        "base_action": np.zeros((1, int(action_dim)), dtype=np.float32),
        "base_action_chunk": np.zeros(
            (1, int(chunk_horizon), int(action_dim)),
            dtype=np.float32,
        ),
        "alpha": np.zeros((1, 1), dtype=np.float32),
    }
    for key in resolve_agibot_image_keys(image_keys):
        sample_obs[key] = np.zeros(
            (1, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH, 3),
            dtype=np.uint8,
        )
    return sample_obs


def build_chunk_residual_observation_space(
    *,
    sample_obs: dict[str, np.ndarray],
    image_keys: tuple[str, ...],
) -> gym.spaces.Dict:
    resolved_image_keys = set(resolve_agibot_image_keys(image_keys))
    spaces: dict[str, gym.Space] = {}
    for key, value in sample_obs.items():
        value_array = np.asarray(value)
        if key in resolved_image_keys:
            spaces[key] = gym.spaces.Box(
                low=0,
                high=255,
                shape=value_array.shape,
                dtype=np.uint8,
            )
        else:
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=value_array.shape,
                dtype=np.float32,
            )
    return gym.spaces.Dict(spaces)


__all__ = [
    "build_chunk_residual_obs",
    "build_chunk_residual_observation_space",
    "build_chunk_residual_sample_obs",
    "prepare_base_actions_chunk",
]
