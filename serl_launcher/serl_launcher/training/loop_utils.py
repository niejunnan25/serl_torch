"""Generic training loop scheduling helpers."""
from __future__ import annotations

from typing import List
from typing import Optional


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
