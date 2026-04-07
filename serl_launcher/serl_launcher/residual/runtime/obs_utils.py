"""Observation dictionary helper utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym


def _obs_space_from_sample(sample_obs: Dict[str, np.ndarray]) -> gym.spaces.Dict:
    spaces: Dict[str, gym.spaces.Space] = {}
    for key, value in sample_obs.items():
        arr = np.asarray(value)
        if key == "state":
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=np.float32,
            )
        elif np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            spaces[key] = gym.spaces.Box(
                low=info.min,
                high=info.max,
                shape=arr.shape,
                dtype=arr.dtype,
            )
        else:
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=np.float32,
            )
    return gym.spaces.Dict(spaces)


def _clone_obs_dict(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in obs_dict.items()}


def _zero_obs_like(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in obs_dict.items()}
