from __future__ import annotations

"""In-process AgiBot rollout processor.

The real-robot actor keeps responsibility for robot control and safety-critical
stepping. This processor owns the data path after a chunk has executed:
transition assembly, pending terminal bookkeeping, and replay insertion.
"""

from dataclasses import dataclass
from typing import Any
from typing import Callable

import numpy as np

from serl_launcher.rollout.runtime_helpers import commit_finished_episode_chunks

from .transition_assembly import AgiBotTransitionAssembler
from .transition_assembly import AssemblyResult
from .transition_assembly import RawChunkRecord


@dataclass(frozen=True, slots=True)
class AgiBotProcessedChunk:
    raw: RawChunkRecord
    assembled_chunks: tuple[AssemblyResult, ...]


class AgiBotRolloutProcessor:
    """Process executed rollout chunks into replay transitions."""

    def __init__(
        self,
        *,
        transition_assembler: AgiBotTransitionAssembler,
        data_store: Any,
        trainer_update_fn: Callable[[str], None],
        steps_per_update: int,
    ) -> None:
        self.transition_assembler = transition_assembler
        self._data_store = data_store
        self._trainer_update_fn = trainer_update_fn
        self._steps_per_update = max(1, int(steps_per_update))
        self._pending_last_transition: dict[str, Any] | None = None
        self._committed_env_steps = 0

    @property
    def committed_env_steps(self) -> int:
        return int(self._committed_env_steps)

    def drain_ready(self) -> tuple[AssemblyResult, ...]:
        assembled_chunks = tuple(self.transition_assembler.drain_ready())
        self.commit_assembled_chunks(assembled_chunks)
        return assembled_chunks

    def process_step_chunk(
        self,
        *,
        episode_id: int,
        episode_step_start: int,
        residual_obs_before_chunk: dict[str, np.ndarray],
        action_chunk: np.ndarray,
        chunk_result: dict[str, Any],
        task_prompt: str,
    ) -> AgiBotProcessedChunk:
        raw = RawChunkRecord.from_step_chunk_result(
            episode_id=int(episode_id),
            episode_step_start=int(episode_step_start),
            residual_obs_before_chunk=residual_obs_before_chunk,
            action_chunk=action_chunk,
            chunk_result=chunk_result,
        )
        assembled_chunks = tuple(
            self.transition_assembler.handle_chunk(
                raw=raw,
                task_prompt=str(task_prompt),
            )
        )
        self.commit_assembled_chunks(assembled_chunks)
        return AgiBotProcessedChunk(
            raw=raw,
            assembled_chunks=assembled_chunks,
        )

    def finalize_zero_step_terminal(
        self,
        *,
        terminal_reward: float,
        boundary_flag: bool,
        wait_for_episode_commit: bool,
    ) -> None:
        self._commit_finished_episode_chunks(
            wait_for_episode_commit=bool(wait_for_episode_commit),
            require_last_transition_ready=True,
        )
        self._finalize_pending_last_transition(
            terminal_reward=float(terminal_reward),
            boundary_flag=bool(boundary_flag),
        )

    def finish_episode(
        self,
        *,
        wait_for_episode_commit: bool,
        require_last_transition_ready: bool = False,
    ) -> None:
        self._commit_finished_episode_chunks(
            wait_for_episode_commit=bool(wait_for_episode_commit),
            require_last_transition_ready=bool(require_last_transition_ready),
        )
        self._flush_pending_last_transition()

    def _commit_finished_episode_chunks(
        self,
        *,
        wait_for_episode_commit: bool,
        require_last_transition_ready: bool,
    ) -> None:
        commit_finished_episode_chunks(
            transition_assembler=self.transition_assembler,
            commit_assembled_chunks=self.commit_assembled_chunks,
            wait_for_episode_commit=bool(wait_for_episode_commit),
            require_last_transition_ready=bool(require_last_transition_ready),
        )

    def commit_assembled_chunks(self, assembled_chunks: list[Any] | tuple[Any, ...]) -> None:
        for assembled_chunk in assembled_chunks:
            self._flush_pending_last_transition()
            if bool(assembled_chunk.episode_done):
                transitions_to_insert = assembled_chunk.transitions
            else:
                transitions_to_insert = assembled_chunk.transitions[:-1]
                self._pending_last_transition = assembled_chunk.transitions[-1]

            for transition in transitions_to_insert:
                self._data_store.insert(transition)

            for step_offset in range(1, int(assembled_chunk.env_steps_delta) + 1):
                next_committed_env_step = int(self._committed_env_steps + step_offset)
                if next_committed_env_step % self._steps_per_update == 0:
                    self._trainer_update_fn(
                        f"commit_step_{int(next_committed_env_step)}"
                    )
            self._committed_env_steps += int(assembled_chunk.env_steps_delta)

    def _flush_pending_last_transition(self) -> None:
        if self._pending_last_transition is None:
            return
        self._data_store.insert(self._pending_last_transition)
        self._pending_last_transition = None

    def _finalize_pending_last_transition(
        self,
        *,
        terminal_reward: float,
        boundary_flag: bool,
    ) -> None:
        if self._pending_last_transition is None:
            return
        self._pending_last_transition["rewards"] = float(
            self._pending_last_transition["rewards"]
        ) + float(terminal_reward)
        self._pending_last_transition["dones"] = bool(boundary_flag)
        self._pending_last_transition["masks"] = 0.0
        self._data_store.insert(self._pending_last_transition)
        self._pending_last_transition = None

    def close(self) -> None:
        self.transition_assembler.close()


__all__ = [
    "AgiBotProcessedChunk",
    "AgiBotRolloutProcessor",
]
