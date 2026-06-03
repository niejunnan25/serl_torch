"""RLT observation construction and observation space definition."""

from __future__ import annotations

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym
import numpy as np


def build_rlt_sample_obs(
    z_rl_dim: int = 2048,
    proprio_dim: int = 8,
    chunk_size: int = 10,
    action_dim: int = 7,
) -> dict[str, np.ndarray]:
    """Build a sample observation dict for agent initialization.

    All values have a leading batch dimension of 1.
    """
    return {
        "z_rl": np.zeros((1, z_rl_dim), dtype=np.float32),
        "proprio": np.zeros((1, proprio_dim), dtype=np.float32),
        "reference_action": np.zeros((1, chunk_size * action_dim), dtype=np.float32),
    }


def build_rlt_obs(
    z_rl: np.ndarray,
    proprio: np.ndarray,
    reference_action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build observation dict for a single timestep (no batch dim)."""
    return {
        "z_rl": np.asarray(z_rl, dtype=np.float32),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "reference_action": np.asarray(reference_action, dtype=np.float32),
    }


def build_rlt_observation_space(
    z_rl_dim: int = 2048,
    proprio_dim: int = 8,
    chunk_size: int = 10,
    action_dim: int = 7,
) -> gym.spaces.Dict:
    """Build gym observation space for the RLT replay buffer."""
    return gym.spaces.Dict(
        {
            "z_rl": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(z_rl_dim,), dtype=np.float32
            ),
            "proprio": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32
            ),
            "reference_action": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(chunk_size * action_dim,), dtype=np.float32
            ),
        }
    )
