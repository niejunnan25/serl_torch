from __future__ import annotations

"""Runtime helpers for LIBERO direct-action RLPD rollouts and eval."""

from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np


def should_sample_random_action(*, env_steps: int, random_steps: int) -> bool:
    return int(env_steps) < max(0, int(random_steps))


def sample_uniform_random_action(
    *,
    action_dim: int,
    rng: Any = np.random,
) -> np.ndarray:
    resolved_action_dim = int(action_dim)
    if resolved_action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {resolved_action_dim}")
    return np.asarray(
        rng.uniform(-1.0, 1.0, size=(resolved_action_dim,)),
        dtype=np.float32,
    ).reshape(-1)


def sample_actor_action(
    *,
    policy_action_fn: Callable[[], Any],
    env_steps: int,
    random_steps: int,
    action_dim: int,
    rng: Any = np.random,
) -> tuple[np.ndarray, bool]:
    if should_sample_random_action(
        env_steps=int(env_steps),
        random_steps=int(random_steps),
    ):
        return (
            sample_uniform_random_action(
                action_dim=int(action_dim),
                rng=rng,
            ),
            True,
        )
    return np.asarray(policy_action_fn(), dtype=np.float32).reshape(-1), False


def require_eval_checkpoint(
    *,
    checkpoint_file: Path | None,
    allow_random_policy: bool,
) -> None:
    if checkpoint_file is not None or bool(allow_random_policy):
        return
    raise ValueError(
        "direct-action RLPD eval requires eval.checkpoint_path unless "
        "eval.allow_random_policy=true"
    )


__all__ = [
    "require_eval_checkpoint",
    "sample_actor_action",
    "sample_uniform_random_action",
    "should_sample_random_action",
]
