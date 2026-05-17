from __future__ import annotations

"""Generic async transition-assembly coordination for rollout runtimes."""

from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from typing import Callable
from typing import Generic
from typing import TypeVar


RawT = TypeVar("RawT")
ObservationT = TypeVar("ObservationT")
BackfillT = TypeVar("BackfillT")
ResultT = TypeVar("ResultT")


@dataclass
class _PendingTransitionAssembly(Generic[RawT, BackfillT]):
    chunk_seq: int
    raw: RawT
    backfill_future: Future[list[BackfillT]]


class AsyncTransitionAssemblyCoordinator(
    Generic[RawT, ObservationT, BackfillT, ResultT]
):
    def __init__(
        self,
        *,
        backfill_fn: Callable[[list[ObservationT], str], list[BackfillT]],
        build_result_fn: Callable[[RawT, list[BackfillT]], ResultT],
        thread_name_prefix: str,
        logger: logging.Logger | None = None,
        close_fn: Callable[[], None] | None = None,
        close_error_message: str = "ignored async transition assembly close error",
    ) -> None:
        self._backfill_fn = backfill_fn
        self._build_result_fn = build_result_fn
        self._logger = logger
        self._close_fn = close_fn
        self._close_error_message = str(close_error_message)
        self._assembly_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=str(thread_name_prefix),
        )
        self._pending: dict[int, _PendingTransitionAssembly[RawT, BackfillT]] = {}
        self._next_submit_chunk_seq = 0
        self._next_commit_chunk_seq = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def next_commit_chunk_seq(self) -> int:
        return int(self._next_commit_chunk_seq)

    @property
    def latest_submitted_chunk_seq(self) -> int | None:
        if int(self._next_submit_chunk_seq) <= 0:
            return None
        return int(self._next_submit_chunk_seq - 1)

    def submit_chunk(
        self,
        *,
        raw: RawT,
        observations: list[ObservationT],
        task_prompt: str,
    ) -> int:
        chunk_seq = int(self._next_submit_chunk_seq)
        self._next_submit_chunk_seq += 1
        backfill_future = self._assembly_executor.submit(
            self._backfill_fn,
            list(observations),
            task_prompt,
        )
        self._pending[chunk_seq] = _PendingTransitionAssembly(
            chunk_seq=chunk_seq,
            raw=raw,
            backfill_future=backfill_future,
        )
        return chunk_seq

    def pop_committable(
        self,
        *,
        block_until_seq: int | None = None,
    ) -> list[ResultT]:
        assembled_chunks: list[ResultT] = []
        while int(self._next_commit_chunk_seq) in self._pending:
            next_seq = int(self._next_commit_chunk_seq)
            pending = self._pending[next_seq]
            if block_until_seq is not None and next_seq <= int(block_until_seq):
                backfilled_values = pending.backfill_future.result()
            elif pending.backfill_future.done():
                backfilled_values = pending.backfill_future.result()
            else:
                break
            assembled_chunks.append(
                self._build_result_fn(pending.raw, backfilled_values)
            )
            self._pending.pop(next_seq, None)
            self._next_commit_chunk_seq += 1
        return assembled_chunks

    def close(self) -> None:
        self._assembly_executor.shutdown(wait=True, cancel_futures=False)
        if self._close_fn is None:
            return
        try:
            self._close_fn()
        except Exception:  # noqa: BLE001
            if self._logger is None:
                raise
            self._logger.debug(self._close_error_message, exc_info=True)


__all__ = ["AsyncTransitionAssemblyCoordinator"]
