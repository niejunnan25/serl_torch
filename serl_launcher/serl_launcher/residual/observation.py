"""Environment-agnostic residual-observation schema helpers."""
from __future__ import annotations

from typing import Iterable
from typing import Mapping

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym


def _normalize_image_keys(image_keys: Iterable[str]) -> tuple[str, ...]:
    normalized_keys: list[str] = []
    seen: set[str] = set()
    for image_key in image_keys:
        key = str(image_key)
        if key in seen:
            continue
        seen.add(key)
        normalized_keys.append(key)
    if not normalized_keys:
        raise ValueError("At least one image key is required")
    return tuple(normalized_keys)


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
    robot_state: np.ndarray,
    images: Mapping[str, np.ndarray],
    image_keys: Iterable[str],
    base_actions: np.ndarray,
    residual_alpha: float,
) -> dict[str, np.ndarray]:
    base_actions_array = np.asarray(base_actions, dtype=np.float32)
    if base_actions_array.ndim != 2 or int(base_actions_array.shape[0]) <= 0:
        raise ValueError(
            "base_actions must be a non-empty 2D array, "
            f"got shape {base_actions_array.shape}"
        )

    residual_obs: dict[str, np.ndarray] = {
        "robot_proprio": np.expand_dims(
            np.asarray(robot_state, dtype=np.float32).reshape(-1),
            axis=0,
        ).astype(np.float32),
        "base_action": np.expand_dims(base_actions_array[0], axis=0).astype(np.float32),
        "base_action_chunk": np.expand_dims(base_actions_array, axis=0).astype(
            np.float32
        ),
        "alpha": np.asarray([[residual_alpha]], dtype=np.float32),
    }
    for key in _normalize_image_keys(image_keys):
        if key not in images:
            raise KeyError(f"Missing image key {key!r}; available keys={sorted(images)}")
        residual_obs[key] = np.expand_dims(
            np.asarray(images[key], dtype=np.uint8).copy(),
            axis=0,
        )
    return residual_obs


def build_chunk_residual_sample_obs(
    *,
    state_dim: int,
    action_dim: int,
    chunk_horizon: int,
    image_keys: Iterable[str],
    image_height: int,
    image_width: int,
) -> dict[str, np.ndarray]:
    if int(state_dim) <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}")
    if int(action_dim) <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if int(chunk_horizon) <= 0:
        raise ValueError(f"chunk_horizon must be positive, got {chunk_horizon}")
    if int(image_height) <= 0:
        raise ValueError(f"image_height must be positive, got {image_height}")
    if int(image_width) <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}")

    sample_obs: dict[str, np.ndarray] = {
        "robot_proprio": np.zeros((1, int(state_dim)), dtype=np.float32),
        "base_action": np.zeros((1, int(action_dim)), dtype=np.float32),
        "base_action_chunk": np.zeros(
            (1, int(chunk_horizon), int(action_dim)),
            dtype=np.float32,
        ),
        "alpha": np.zeros((1, 1), dtype=np.float32),
    }
    for key in _normalize_image_keys(image_keys):
        sample_obs[key] = np.zeros(
            (1, int(image_height), int(image_width), 3),
            dtype=np.uint8,
        )
    return sample_obs


def build_chunk_residual_observation_space(
    *,
    sample_obs: dict[str, np.ndarray],
    image_keys: Iterable[str],
) -> gym.spaces.Dict:
    resolved_image_keys = set(_normalize_image_keys(image_keys))
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
