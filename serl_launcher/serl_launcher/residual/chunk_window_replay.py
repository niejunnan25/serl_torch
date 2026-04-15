from __future__ import annotations

"""Chunk-window replay helpers for residual training pipelines."""

from typing import Any

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore


def create_chunk_replay_buffer(
    *,
    observation_space: gym.spaces.Dict,
    action_dim: int,
    chunk_horizon: int,
    discount: float,
    image_keys: tuple[str, ...],
    capacity: int,
) -> MemoryEfficientStepWindowReplayBufferDataStore:
    """Create a replay buffer configured for chunked residual training."""

    return MemoryEfficientStepWindowReplayBufferDataStore(
        observation_space=observation_space,
        action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        ),
        capacity=int(capacity),
        window_size=int(chunk_horizon),
        discount=float(discount),
        sample_stride=1,
        require_full_window=False,
        image_keys=image_keys,
    )


def reshape_chunk_batch_for_training(batch: dict[str, Any]) -> dict[str, Any]:
    """Flatten chunk action tensors into the shape expected by learner updates."""

    batch_out = dict(batch)
    if "actions" in batch_out:
        actions = np.asarray(batch_out["actions"])
        batch_out["actions"] = actions.reshape(int(actions.shape[0]), -1)
    if "action_mask" in batch_out:
        action_mask = np.asarray(batch_out["action_mask"])
        batch_out["action_mask"] = action_mask.reshape(int(action_mask.shape[0]), -1)
    return batch_out


def concat_batch_trees(values: list[Any]) -> Any:
    """Concatenate nested batch trees along the batch axis."""

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
    online_replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore,
    offline_replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore | None,
    batch_size: int,
    offline_ratio: float,
    reshape_batch: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Sample a mixed online/offline training batch for chunked residual updates."""

    offline_size = (
        0 if offline_replay_buffer is None else int(len(offline_replay_buffer))
    )
    online_size = int(len(online_replay_buffer))

    if offline_replay_buffer is None or offline_size <= 0 or offline_ratio <= 0.0:
        batch = online_replay_buffer.sample(int(batch_size))
        batch_mix = {
            "online_batch_size": int(batch_size),
            "offline_batch_size": 0,
        }
        if reshape_batch:
            batch = reshape_chunk_batch_for_training(batch)
        return batch, batch_mix
    if online_size <= 0:
        batch = offline_replay_buffer.sample(int(batch_size))
        batch_mix = {
            "online_batch_size": 0,
            "offline_batch_size": int(batch_size),
        }
        if reshape_batch:
            batch = reshape_chunk_batch_for_training(batch)
        return batch, batch_mix

    offline_batch_size = int(round(float(batch_size) * float(offline_ratio)))
    offline_batch_size = min(int(batch_size), max(0, int(offline_batch_size)))
    online_batch_size = int(batch_size) - int(offline_batch_size)

    if offline_batch_size <= 0:
        batch = online_replay_buffer.sample(int(batch_size))
        batch_mix = {
            "online_batch_size": int(batch_size),
            "offline_batch_size": 0,
        }
        if reshape_batch:
            batch = reshape_chunk_batch_for_training(batch)
        return batch, batch_mix
    if online_batch_size <= 0:
        batch = offline_replay_buffer.sample(int(batch_size))
        batch_mix = {
            "online_batch_size": 0,
            "offline_batch_size": int(batch_size),
        }
        if reshape_batch:
            batch = reshape_chunk_batch_for_training(batch)
        return batch, batch_mix

    online_batch = online_replay_buffer.sample(int(online_batch_size))
    offline_batch = offline_replay_buffer.sample(int(offline_batch_size))
    batch = concat_batch_trees([online_batch, offline_batch])
    batch_mix = {
        "online_batch_size": int(online_batch_size),
        "offline_batch_size": int(offline_batch_size),
    }
    if reshape_batch:
        batch = reshape_chunk_batch_for_training(batch)
    return batch, batch_mix
