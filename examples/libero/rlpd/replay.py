from __future__ import annotations

"""Direct replay helpers for LIBERO RLPD pipelines."""

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

if TYPE_CHECKING:
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore


def create_rlpd_replay_buffer(
    *,
    observation_space: gym.spaces.Dict,
    action_dim: int,
    image_keys: tuple[str, ...],
    capacity: int,
) -> MemoryEfficientReplayBufferDataStore:
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

    return MemoryEfficientReplayBufferDataStore(
        observation_space=observation_space,
        action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        ),
        capacity=int(capacity),
        image_keys=image_keys,
    )


def concat_batch_trees(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {
            key: concat_batch_trees([value[key] for value in values])
            for key in first
        }
    arrays = [np.asarray(value) for value in values]
    return np.concatenate(arrays, axis=0)


def sample_mixed_training_batch(
    *,
    online_replay_buffer: Any,
    offline_replay_buffer: Any | None,
    batch_size: int,
    offline_ratio: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    offline_size = 0 if offline_replay_buffer is None else int(len(offline_replay_buffer))
    online_size = int(len(online_replay_buffer))

    if offline_replay_buffer is None or offline_size <= 0 or offline_ratio <= 0.0:
        return (
            online_replay_buffer.sample(int(batch_size)),
            {
                "online_batch_size": int(batch_size),
                "offline_batch_size": 0,
            },
        )
    if online_size <= 0:
        return (
            offline_replay_buffer.sample(int(batch_size)),
            {
                "online_batch_size": 0,
                "offline_batch_size": int(batch_size),
            },
        )

    offline_batch_size = int(round(float(batch_size) * float(offline_ratio)))
    offline_batch_size = min(int(batch_size), max(0, int(offline_batch_size)))
    online_batch_size = int(batch_size) - int(offline_batch_size)

    if offline_batch_size <= 0:
        return (
            online_replay_buffer.sample(int(batch_size)),
            {
                "online_batch_size": int(batch_size),
                "offline_batch_size": 0,
            },
        )
    if online_batch_size <= 0:
        return (
            offline_replay_buffer.sample(int(batch_size)),
            {
                "online_batch_size": 0,
                "offline_batch_size": int(batch_size),
            },
        )

    online_batch = online_replay_buffer.sample(int(online_batch_size))
    offline_batch = offline_replay_buffer.sample(int(offline_batch_size))
    return (
        concat_batch_trees([online_batch, offline_batch]),
        {
            "online_batch_size": int(online_batch_size),
            "offline_batch_size": int(offline_batch_size),
        },
    )


__all__ = [
    "create_rlpd_replay_buffer",
    "sample_mixed_training_batch",
]
