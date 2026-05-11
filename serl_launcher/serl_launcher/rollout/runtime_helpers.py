from __future__ import annotations

"""Lightweight helpers for rollout episode boundaries."""

from typing import Any
from typing import Callable


def commit_finished_episode_chunks(
    *,
    transition_assembler: Any,
    commit_assembled_chunks: Callable[[list[Any]], None],
    wait_for_episode_commit: bool,
    require_last_transition_ready: bool = False,
) -> None:
    """Drain finished-episode chunks before boundary-side bookkeeping.

    This lets callers materialize the previous real step before attaching any
    synthetic terminal reward or final flags.
    """

    assembled_chunks = transition_assembler.finish_episode(
        block=bool(wait_for_episode_commit or require_last_transition_ready),
    )
    commit_assembled_chunks(assembled_chunks)
