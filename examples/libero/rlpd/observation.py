"""Observation-schema helpers for LIBERO direct-action RLPD training."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from ..env.observation import LIBERO_STATE_DIM
from ..env.observation import RESIDUAL_IMAGE_HEIGHT
from ..env.observation import RESIDUAL_IMAGE_WIDTH
from ..env.observation import build_libero_state
from ..env.observation import extract_libero_images
from ..env.observation import resolve_libero_image_keys


def build_rlpd_obs(
    *,
    obs: dict[str, Any],
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    resolved_image_keys = resolve_libero_image_keys(image_keys)
    images = extract_libero_images(obs)
    rlpd_obs: dict[str, np.ndarray] = {
        "robot_proprio": np.expand_dims(build_libero_state(obs), axis=0).astype(
            np.float32
        ),
    }
    for key in resolved_image_keys:
        rlpd_obs[key] = np.expand_dims(images[key].copy(), axis=0)
    return rlpd_obs


def build_rlpd_sample_obs(
    *,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    sample_obs: dict[str, np.ndarray] = {
        "robot_proprio": np.zeros((1, LIBERO_STATE_DIM), dtype=np.float32),
    }
    for key in resolve_libero_image_keys(image_keys):
        sample_obs[key] = np.zeros(
            (1, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH, 3),
            dtype=np.uint8,
        )
    return sample_obs


def build_rlpd_observation_space(
    *,
    sample_obs: dict[str, np.ndarray],
    image_keys: tuple[str, ...],
) -> gym.spaces.Dict:
    resolved_image_keys = set(resolve_libero_image_keys(image_keys))
    spaces: dict[str, gym.Space] = {}
    for key, value in sample_obs.items():
        value_arr = np.asarray(value)
        if key in resolved_image_keys:
            spaces[key] = gym.spaces.Box(
                low=0,
                high=255,
                shape=value_arr.shape,
                dtype=np.uint8,
            )
        else:
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=value_arr.shape,
                dtype=np.float32,
            )
    return gym.spaces.Dict(spaces)


__all__ = [
    "build_rlpd_obs",
    "build_rlpd_observation_space",
    "build_rlpd_sample_obs",
]
