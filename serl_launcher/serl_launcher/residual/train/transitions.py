"""Residual-train transition and chunk packing helpers."""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import numpy as np


def _insert_online_transition(
    replay_buffer: Any,
    transition_payload: Dict[str, Any],
    *,
    chunk_step_enabled: bool,
) -> None:
    if chunk_step_enabled:
        replay_buffer.insert(transition_payload)
        return
    replay_buffer.insert(
        {
            "observations": transition_payload["observations"],
            "actions": transition_payload["actions"],
            "next_observations": transition_payload["next_observations"],
            "rewards": transition_payload["rewards"],
            "masks": transition_payload["masks"],
            "dones": transition_payload["dones"],
        }
    )


def _discounted_chunk_reward_sum(
    rewards: List[float], *, discount: float
) -> np.float32:
    total = 0.0
    for offset, reward in enumerate(rewards):
        total += (float(discount) ** int(offset)) * float(reward)
    return np.float32(total)


def _pack_chunk_actions(
    action_chunk: Any,
    *,
    chunk_horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    action_chunk_arr = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk_arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D chunk action array, got shape={action_chunk_arr.shape}"
        )
    if action_chunk_arr.shape[1] != int(action_dim):
        raise ValueError(
            f"Unexpected chunk action dim: {action_chunk_arr.shape[1]} != {int(action_dim)}"
        )

    max_steps = int(min(int(chunk_horizon), int(action_chunk_arr.shape[0])))
    packed_actions = np.zeros((int(chunk_horizon), int(action_dim)), dtype=np.float32)
    action_mask = np.zeros((int(chunk_horizon), int(action_dim)), dtype=np.float32)
    if max_steps > 0:
        packed_actions[:max_steps] = action_chunk_arr[:max_steps]
        action_mask[:max_steps] = 1.0
    return packed_actions.reshape(-1), action_mask.reshape(-1), int(max_steps)


def _chunk_bootstrap_mask(
    *, chunk_steps: int, discount: float, done: bool
) -> np.float32:
    if bool(done):
        return np.float32(0.0)
    return np.float32(float(discount) ** max(0, int(chunk_steps) - 1))
