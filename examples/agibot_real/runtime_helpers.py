from __future__ import annotations

"""Lightweight runtime helpers for AgiBot actor-side episode boundaries."""

from typing import Any
from typing import Callable


def commit_finished_episode_chunks(
    *,
    transition_assembler: Any,
    commit_assembled_chunks: Callable[[list[Any]], None],
    wait_for_episode_commit: bool,
    require_last_transition_ready: bool = False,
) -> None:
    """Drain finished-episode chunks before applying boundary-side bookkeeping.

    AgiBot's ``executed_steps == 0`` terminal path needs the previous real step to
    be materialized first, otherwise synthetic terminal reward/final flags have no
    transition to attach to.
    """

    assembled_chunks = transition_assembler.finish_episode(
        block=bool(wait_for_episode_commit or require_last_transition_ready),
    )
    commit_assembled_chunks(assembled_chunks)
