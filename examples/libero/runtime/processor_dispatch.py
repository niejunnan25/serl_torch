from __future__ import annotations

from dataclasses import dataclass
import queue
from threading import Lock
from threading import Thread
from typing import Any
from typing import Literal


@dataclass(frozen=True, slots=True)
class _ProcessorCommand:
    kind: Literal["submit", "finish", "mark_episode_end", "shutdown", "close"]
    payload: dict[str, Any]
    context: str


class QueuedProcessorSubmitter:
    def __init__(
        self,
        *,
        processor_client: Any,
        logger: Any,
        queue_maxsize: int = 0,
        thread_name: str = "libero-processor-submit",
    ) -> None:
        self._processor_client = processor_client
        self._logger = logger
        self._command_queue: queue.Queue[_ProcessorCommand] = queue.Queue(
            maxsize=max(0, int(queue_maxsize))
        )
        self._failure_lock = Lock()
        self._failure: BaseException | None = None
        self._close_requested = False
        self._sender_thread = Thread(
            target=self._sender_loop,
            name=str(thread_name),
            daemon=True,
        )
        self._sender_thread.start()

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 30.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.raise_if_failed()
        self._processor_client.wait_until_ready(
            timeout_s=float(timeout_s),
            poll_interval_s=float(poll_interval_s),
        )

    def submit_chunk(self, *, payload: dict[str, Any], context: str) -> None:
        self._enqueue(
            _ProcessorCommand(
                kind="submit",
                payload=dict(payload),
                context=str(context),
            )
        )

    def finish_episode(
        self,
        *,
        episode_id: int,
        last_chunk_seq: int | None,
    ) -> None:
        target_chunk_seq = -1 if last_chunk_seq is None else int(last_chunk_seq)
        self._enqueue(
            _ProcessorCommand(
                kind="finish",
                payload={
                    "episode_id": int(episode_id),
                    "last_chunk_seq": int(target_chunk_seq),
                },
                context=f"episode_{int(episode_id)}_finish",
            )
        )

    def mark_episode_end(
        self,
        *,
        episode_id: int,
        last_chunk_seq: int | None,
        actor_progress: dict[str, Any] | None = None,
        rollout_stats: dict[str, Any] | None = None,
    ) -> None:
        target_chunk_seq = -1 if last_chunk_seq is None else int(last_chunk_seq)
        payload = {
            "episode_id": int(episode_id),
            "last_chunk_seq": int(target_chunk_seq),
        }
        if actor_progress is not None:
            payload["actor_progress"] = dict(actor_progress)
        if rollout_stats is not None:
            payload["rollout_stats"] = dict(rollout_stats)
        self._enqueue(
            _ProcessorCommand(
                kind="mark_episode_end",
                payload=payload,
                context=f"episode_{int(episode_id)}_mark",
            )
        )

    def shutdown(self, *, last_chunk_seq: int | None) -> None:
        target_chunk_seq = -1 if last_chunk_seq is None else int(last_chunk_seq)
        self._enqueue(
            _ProcessorCommand(
                kind="shutdown",
                payload={"last_chunk_seq": int(target_chunk_seq)},
                context="shutdown",
            )
        )

    def last_status(self) -> dict[str, Any]:
        try:
            return dict(self._processor_client.last_status())
        except Exception:  # noqa: BLE001
            return {}

    def status_snapshot(self) -> dict[str, Any]:
        failure = self._get_failure()
        return {
            "outbox_depth": int(self._command_queue.qsize()),
            "sender_failed": bool(failure is not None),
            "processor": self.last_status(),
        }

    def raise_if_failed(self) -> None:
        failure = self._get_failure()
        if failure is not None:
            raise RuntimeError(
                "processor sender thread failed; actor can no longer hand off rollout chunks"
            ) from failure

    def close(self, *, wait: bool = True) -> None:
        if not bool(self._close_requested):
            self._close_requested = True
            self._command_queue.put(
                _ProcessorCommand(
                    kind="close",
                    payload={},
                    context="close",
                )
            )
        if bool(wait):
            self._sender_thread.join()
        self.raise_if_failed()

    def _enqueue(self, command: _ProcessorCommand) -> None:
        self.raise_if_failed()
        self._command_queue.put(command)

    def _get_failure(self) -> BaseException | None:
        with self._failure_lock:
            return self._failure

    def _set_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = exc

    def _sender_loop(self) -> None:
        try:
            while True:
                command = self._command_queue.get()
                try:
                    if command.kind == "close":
                        return
                    if command.kind == "submit":
                        self._processor_client.submit(
                            payload=dict(command.payload),
                            context=str(command.context),
                        )
                        continue
                    if command.kind == "finish":
                        self._processor_client.finish(
                            episode_id=int(command.payload["episode_id"]),
                            last_chunk_seq=int(command.payload["last_chunk_seq"]),
                        )
                        continue
                    if command.kind == "mark_episode_end":
                        self._processor_client.mark_episode_end(
                            episode_id=int(command.payload["episode_id"]),
                            last_chunk_seq=int(command.payload["last_chunk_seq"]),
                            actor_progress=command.payload.get("actor_progress"),
                            rollout_stats=command.payload.get("rollout_stats"),
                        )
                        continue
                    if command.kind == "shutdown":
                        self._processor_client.shutdown(
                            last_chunk_seq=int(command.payload["last_chunk_seq"]),
                        )
                        continue
                    raise ValueError(
                        f"unsupported queued processor command: {command.kind}"
                    )
                finally:
                    self._command_queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            self._set_failure(exc)
            self._logger.exception("processor sender thread failed")
        finally:
            try:
                self._processor_client.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["QueuedProcessorSubmitter"]
