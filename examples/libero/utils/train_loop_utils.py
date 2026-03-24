from __future__ import annotations

"""Shared helpers for residual trainer env-step scheduling and replay insertion."""

from typing import Any, Dict, List, Optional

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


def _discounted_chunk_reward_sum(rewards: List[float], *, discount: float) -> np.float32:
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
        raise ValueError(f"Expected 2-D chunk action array, got shape={action_chunk_arr.shape}")
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


def _chunk_bootstrap_mask(*, chunk_steps: int, discount: float, done: bool) -> np.float32:
    if bool(done):
        return np.float32(0.0)
    return np.float32(float(discount) ** max(0, int(chunk_steps) - 1))


def _remaining_train_budget_steps(
    *,
    max_train_env_steps: int,
    train_env_step: int,
) -> Optional[int]:
    if max_train_env_steps <= 0:
        return None
    return max(0, int(max_train_env_steps - train_env_step))


def _count_env_step_update_triggers(
    *,
    train_step_before: int,
    train_step_after: int,
    replay_size_before: int,
    replay_size_after: int,
    training_starts: int,
    update_every: int,
) -> int:
    if int(train_step_after) <= int(train_step_before):
        return 0
    if int(update_every) <= 0:
        raise ValueError(f"update_every must be positive, got {update_every}")
    replay_size = int(replay_size_before)
    replay_size_after = int(replay_size_after)
    trigger_count = 0
    for step_before in range(int(train_step_before), int(train_step_after)):
        if replay_size < replay_size_after:
            replay_size += 1
        if replay_size < int(training_starts):
            continue
        if step_before % int(update_every) == 0:
            trigger_count += 1
    return int(trigger_count)


def _iter_period_hits(
    *,
    step_before: int,
    step_after: int,
    period: int,
) -> List[int]:
    if int(period) <= 0 or int(step_after) <= int(step_before):
        return []
    first_hit = ((int(step_before) // int(period)) + 1) * int(period)
    if first_hit > int(step_after):
        return []
    return list(range(first_hit, int(step_after) + 1, int(period)))
