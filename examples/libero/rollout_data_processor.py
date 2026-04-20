from __future__ import annotations

"""LIBERO rollout data processor for chunk post-processing and replay commit."""

from typing import Any
from typing import Callable

from serl_torch.examples.libero.transition_assembly import (
    AssemblyResult,
    ChunkExecutionRecord,
    LiberoActorTransitionAssembler,
)


class RolloutDataProcessor:
    """Owns chunk post-processing and replay commit details for LIBERO rollout."""

    def __init__(
        self,
        *,
        cfg: Any,
        policy_client: Any,
        data_store: Any,
        update_trainer_transport: Callable[..., bool],
        logger: Any,
    ) -> None:
        self._data_store = data_store
        self._update_trainer_transport = update_trainer_transport
        self._steps_per_update = int(cfg.training.steps_per_update)
        self._committed_env_steps = 0
        self._transition_processor = LiberoActorTransitionAssembler(
            cfg=cfg,
            policy_client=policy_client,
            logger=logger,
        )

    def observe_chunk(
        self,
        *,
        raw_chunk: ChunkExecutionRecord,
        task_prompt: str,
    ) -> None:
        if self._transition_processor.async_transition_assembly_enabled:
            self._commit_assembled_chunks(self._transition_processor.drain_ready())
        assembled_chunks = self._transition_processor.handle_chunk(
            raw=raw_chunk,
            task_prompt=task_prompt,
        )
        self._commit_assembled_chunks(assembled_chunks)

    def finish_episode(
        self,
        *,
        wait_for_episode_commit: bool,
    ) -> None:
        self._commit_assembled_chunks(
            self._transition_processor.finish_episode(
                block=bool(wait_for_episode_commit),
            )
        )

    def close(self) -> None:
        try:
            self._commit_assembled_chunks(
                self._transition_processor.finish_episode(block=True)
            )
        finally:
            self._transition_processor.close()

    def _commit_assembled_chunks(self, assembled_chunks: list[AssemblyResult]) -> None:
        for assembled_chunk in assembled_chunks:
            for transition in assembled_chunk.transitions:
                self._data_store.insert(transition)
            for step_offset in range(1, assembled_chunk.env_steps_delta + 1):
                next_committed_env_step = int(self._committed_env_steps + step_offset)
                if next_committed_env_step % self._steps_per_update == 0:
                    self._update_trainer_transport(
                        context=f"commit_step_{int(next_committed_env_step)}"
                    )
            self._committed_env_steps += int(assembled_chunk.env_steps_delta)


__all__ = ["RolloutDataProcessor"]
