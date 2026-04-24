from __future__ import annotations

"""Stateful rollout processor control runtime built on top of control transport."""

from collections import deque
import queue
import time
from dataclasses import dataclass
from threading import Condition
from threading import Lock
from typing import Any
from typing import Callable

from serl_launcher.rollout.processor_transport import RolloutProcessorControlClient
from serl_launcher.rollout.processor_transport import RolloutProcessorControlServer

TransportStatusFn = Callable[[], dict[str, Any]]
FlushTransportFn = Callable[[str, bool], None]


@dataclass(frozen=True, slots=True)
class ProcessorTransportConfig:
    host: str
    port: int
    timeout_ms: int
    queue_capacity: int


class ProcessorClient:
    def __init__(
        self,
        *,
        transport_config: ProcessorTransportConfig,
        logger: Any,
    ) -> None:
        self.transport_config = transport_config
        self._logger = logger
        self._client = RolloutProcessorControlClient(
            server_ip=str(transport_config.host),
            port=int(transport_config.port),
            timeout_ms=int(transport_config.timeout_ms),
            wait_for_server=False,
        )
        self._logger.info(
            "Processor endpoint: host=%s port=%s timeout_ms=%s",
            str(transport_config.host),
            int(transport_config.port),
            int(transport_config.timeout_ms),
        )
        self._last_status: dict[str, Any] = {}
        self._consecutive_failures = 0
        self._long_request_retry_limit = max(
            5,
            int(
                (30_000.0 + float(max(1, transport_config.timeout_ms)) - 1.0)
                / float(max(1, transport_config.timeout_ms))
            ),
        )
        self._submit_retry_limit = max(
            int(self._long_request_retry_limit),
            10,
        )

    def last_status(self) -> dict[str, Any]:
        return dict(self._last_status)

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 30.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        deadline = time.monotonic() + float(timeout_s)
        self._consecutive_failures = 0
        while True:
            response = self._client.get_status()
            if response is not None:
                self._update_last_status(response)
                self._consecutive_failures = 0
                return
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError(
                    "Timed out waiting for processor server: "
                    f"host={str(self.transport_config.host)} "
                    f"port={int(self.transport_config.port)} "
                    f"timeout_s={float(timeout_s):.1f}"
                )
            self._logger.info(
                "waiting for processor server: host=%s port=%s",
                str(self.transport_config.host),
                int(self.transport_config.port),
            )
            time.sleep(min(float(poll_interval_s), remaining_s))

    def submit(
        self,
        *,
        payload: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        return self._request(
            request_type="submit-chunk",
            payload=payload,
            context=context,
            retry_limit=None,
        )

    def finish(
        self,
        *,
        episode_id: int,
        last_chunk_seq: int | None,
    ) -> dict[str, Any]:
        target_chunk_seq = -1 if last_chunk_seq is None else int(last_chunk_seq)
        return self._request(
            request_type="finish-episode",
            payload={
                "episode_id": int(episode_id),
                "last_chunk_seq": int(target_chunk_seq),
            },
            context=f"episode_{int(episode_id)}_finish",
            retry_limit=int(self._long_request_retry_limit),
        )

    def mark_episode_end(
        self,
        *,
        episode_id: int,
        last_chunk_seq: int | None,
        actor_progress: dict[str, Any] | None = None,
        rollout_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target_chunk_seq = -1 if last_chunk_seq is None else int(last_chunk_seq)
        payload = {
            "episode_id": int(episode_id),
            "last_chunk_seq": int(target_chunk_seq),
        }
        if actor_progress is not None:
            payload["actor_progress"] = dict(actor_progress)
        if rollout_stats is not None:
            payload["rollout_stats"] = dict(rollout_stats)
        return self._request(
            request_type="mark-episode-end",
            payload=payload,
            context=f"episode_{int(episode_id)}_mark",
            retry_limit=5,
        )

    def shutdown(
        self,
        *,
        last_chunk_seq: int | None,
    ) -> dict[str, Any]:
        target_chunk_seq = -1 if last_chunk_seq is None else int(last_chunk_seq)
        return self._request(
            request_type="shutdown",
            payload={"last_chunk_seq": int(target_chunk_seq)},
            context="shutdown",
            retry_limit=int(self._long_request_retry_limit),
        )

    def close(self) -> None:
        self._client.close()

    def _update_last_status(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            self._last_status.clear()
            self._last_status.update(payload)
            return dict(payload)
        return {}

    def _request(
        self,
        *,
        request_type: str,
        payload: dict[str, Any],
        context: str,
        retry_limit: int | None,
    ) -> dict[str, Any]:
        next_log_time = 0.0
        while True:
            response = self._client.request(
                str(request_type),
                payload,
            )
            if response is not None:
                if int(self._consecutive_failures) > 0:
                    self._logger.info(
                        "processor request recovered: type=%s context=%s "
                        "consecutive_failures=%s processor_status=%s",
                        str(request_type),
                        str(context),
                        int(self._consecutive_failures),
                        dict(self._last_status),
                    )
                self._consecutive_failures = 0
                return self._update_last_status(response)
            self._consecutive_failures += 1
            now = time.monotonic()
            if int(self._consecutive_failures) == 1 or now >= next_log_time:
                self._logger.warning(
                    "processor request waiting: type=%s context=%s "
                    "consecutive_failures=%s processor_status=%s",
                    str(request_type),
                    str(context),
                    int(self._consecutive_failures),
                    dict(self._last_status),
                )
                next_log_time = now + 30.0
            if retry_limit is not None and int(self._consecutive_failures) >= int(
                retry_limit
            ):
                raise RuntimeError(
                    f"processor request {str(request_type)!r} failed repeatedly; "
                    "aborting actor run"
                )
            time.sleep(0.1)


class ProcessorServer:
    def __init__(
        self,
        *,
        transport_config: ProcessorTransportConfig,
        transport_status_fn: TransportStatusFn,
        flush_transport_fn: FlushTransportFn,
        wait_committed_on_episode_end: bool,
        wait_committed_on_shutdown: bool,
        logger: Any | None = None,
    ) -> None:
        self.transport_config = transport_config
        self._transport_status_fn = transport_status_fn
        self._flush_transport_fn = flush_transport_fn
        self._wait_committed_on_episode_end = bool(wait_committed_on_episode_end)
        self._wait_committed_on_shutdown = bool(wait_committed_on_shutdown)
        self._logger = logger
        self._raw_chunk_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=int(transport_config.queue_capacity)
        )
        self._flush_lock = Lock()
        self._progress_lock = Lock()
        self._progress_cond = Condition(self._progress_lock)
        self._accepted_chunk_seq = -1
        self._processed_chunk_seq = -1
        self._last_marked_episode_chunk_seq = -1
        self._accepting_submissions = True
        self._stop_requested = False
        self._pending_episode_flush_markers: deque[dict[str, Any]] = deque()
        self._flushed_episode_markers: deque[dict[str, Any]] = deque()
        self._control_server = RolloutProcessorControlServer(
            port=int(transport_config.port),
            callback=self._control_callback,
        )

    def start(self) -> None:
        self._control_server.start(threaded=True)
        if self._logger is not None:
            self._logger.info(
                "Processor control server listening on port=%s queue_capacity=%s",
                int(self.transport_config.port),
                int(self.transport_config.queue_capacity),
            )

    def request_stop(self) -> None:
        with self._progress_cond:
            self._accepting_submissions = False
            self._stop_requested = True
            self._progress_cond.notify_all()

    def stop(self) -> None:
        self.request_stop()
        self._control_server.stop()

    def should_stop(self) -> bool:
        with self._progress_lock:
            return bool(self._stop_requested)

    def get_chunk(self, *, timeout_s: float = 0.1) -> dict[str, Any] | None:
        try:
            return self._raw_chunk_queue.get(timeout=float(timeout_s))
        except queue.Empty:
            return None

    def task_done(self) -> None:
        self._raw_chunk_queue.task_done()

    def mark_chunk_committed(self, *, chunk_seq: int) -> None:
        with self._progress_cond:
            self._processed_chunk_seq = max(
                int(self._processed_chunk_seq),
                int(chunk_seq),
            )
            self._progress_cond.notify_all()

    def status_snapshot(self) -> dict[str, Any]:
        with self._progress_lock:
            processed_chunk_seq = int(self._processed_chunk_seq)
            return {
                "accepted_chunk_seq": int(self._accepted_chunk_seq),
                "processed_chunk_seq": processed_chunk_seq,
                "committed_chunk_seq": processed_chunk_seq,
                "last_marked_episode_chunk_seq": int(
                    self._last_marked_episode_chunk_seq
                ),
                "pending_episode_flushes": int(
                    len(self._pending_episode_flush_markers)
                ),
                "accepting_submissions": bool(self._accepting_submissions),
                "stop_requested": bool(self._stop_requested),
                "queue_depth": int(self._raw_chunk_queue.qsize()),
            }

    def consume_flushed_episode_markers(self) -> list[dict[str, Any]]:
        with self._progress_cond:
            flushed_markers = list(self._flushed_episode_markers)
            self._flushed_episode_markers.clear()
            return flushed_markers

    def flush_ready_episode_markers(self) -> list[dict[str, Any]]:
        flushed_markers: list[dict[str, Any]] = []
        with self._flush_lock:
            while True:
                with self._progress_cond:
                    if not self._pending_episode_flush_markers:
                        return flushed_markers
                    marker = dict(self._pending_episode_flush_markers[0])
                    last_chunk_seq = int(marker["last_chunk_seq"])
                    if int(self._processed_chunk_seq) < int(last_chunk_seq):
                        return flushed_markers
                self._flush_transport_fn(
                    f"episode_{int(marker['episode_id'])}_end",
                    bool(self._wait_committed_on_episode_end),
                )
                with self._progress_cond:
                    if not self._pending_episode_flush_markers:
                        raise RuntimeError(
                            "processor pending episode marker disappeared during flush"
                        )
                    head_marker = self._pending_episode_flush_markers[0]
                    if int(head_marker["episode_id"]) != int(marker["episode_id"]) or int(
                        head_marker["last_chunk_seq"]
                    ) != int(marker["last_chunk_seq"]):
                        raise RuntimeError(
                            "processor pending episode marker changed during flush"
                        )
                    self._pending_episode_flush_markers.popleft()
                    self._flushed_episode_markers.append(dict(marker))
                flushed_markers.append(marker)

    def _wait_until_chunk_processed(self, *, last_chunk_seq: int) -> None:
        target_chunk_seq = int(last_chunk_seq)
        if target_chunk_seq < 0:
            return
        with self._progress_cond:
            while int(self._processed_chunk_seq) < int(target_chunk_seq):
                if bool(self._stop_requested):
                    raise RuntimeError(
                        "processor stopped before target chunk was processed: "
                        f"target={int(target_chunk_seq)} processed={int(self._processed_chunk_seq)}"
                    )
                self._progress_cond.wait(timeout=0.1)

    def _control_callback(
        self,
        request_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if request_type in ("status", "get-status"):
            return {
                **self.status_snapshot(),
                "transport": self._transport_status_fn(),
            }
        if request_type == "finish-episode":
            episode_id = int(payload.get("episode_id", -1))
            last_chunk_seq = int(payload.get("last_chunk_seq", -1))
            self._wait_until_chunk_processed(last_chunk_seq=last_chunk_seq)
            self._flush_transport_fn(
                f"episode_{int(episode_id)}_end",
                bool(self._wait_committed_on_episode_end),
            )
            return {
                **self.status_snapshot(),
                "transport": self._transport_status_fn(),
            }
        if request_type == "mark-episode-end":
            episode_id = int(payload.get("episode_id", -1))
            last_chunk_seq = int(payload.get("last_chunk_seq", -1))
            actor_progress = payload.get("actor_progress", None)
            rollout_stats = payload.get("rollout_stats", None)
            marker_payload = {
                "episode_id": int(episode_id),
                "last_chunk_seq": int(last_chunk_seq),
            }
            if isinstance(actor_progress, dict):
                marker_payload["actor_progress"] = dict(actor_progress)
            if isinstance(rollout_stats, dict):
                marker_payload["rollout_stats"] = dict(rollout_stats)
            with self._progress_cond:
                accepted_snapshot = int(self._accepted_chunk_seq)
                processed_snapshot = int(self._processed_chunk_seq)
                accepting_snapshot = bool(self._accepting_submissions)
                stop_snapshot = bool(self._stop_requested)
                last_marked_snapshot = int(self._last_marked_episode_chunk_seq)
                pending_snapshot = int(len(self._pending_episode_flush_markers))
                if int(last_chunk_seq) <= int(self._last_marked_episode_chunk_seq):
                    return {
                        "accepted_chunk_seq": int(accepted_snapshot),
                        "processed_chunk_seq": int(processed_snapshot),
                        "committed_chunk_seq": int(processed_snapshot),
                        "last_marked_episode_chunk_seq": int(last_marked_snapshot),
                        "pending_episode_flushes": int(pending_snapshot),
                        "accepting_submissions": bool(accepting_snapshot),
                        "stop_requested": bool(stop_snapshot),
                        "queue_depth": int(self._raw_chunk_queue.qsize()),
                        "transport": self._transport_status_fn(),
                        "deduped": True,
                    }
                if int(last_chunk_seq) >= 0:
                    self._pending_episode_flush_markers.append(marker_payload)
                self._last_marked_episode_chunk_seq = max(
                    int(self._last_marked_episode_chunk_seq),
                    int(last_chunk_seq),
                )
                accepted_snapshot = int(self._accepted_chunk_seq)
                processed_snapshot = int(self._processed_chunk_seq)
                accepting_snapshot = bool(self._accepting_submissions)
                stop_snapshot = bool(self._stop_requested)
                last_marked_snapshot = int(self._last_marked_episode_chunk_seq)
                pending_snapshot = int(len(self._pending_episode_flush_markers))
                self._progress_cond.notify_all()
            return {
                "accepted_chunk_seq": int(accepted_snapshot),
                "processed_chunk_seq": int(processed_snapshot),
                "committed_chunk_seq": int(processed_snapshot),
                "last_marked_episode_chunk_seq": int(last_marked_snapshot),
                "pending_episode_flushes": int(pending_snapshot),
                "accepting_submissions": bool(accepting_snapshot),
                "stop_requested": bool(stop_snapshot),
                "queue_depth": int(self._raw_chunk_queue.qsize()),
                "transport": self._transport_status_fn(),
                "deduped": False,
            }
        if request_type == "shutdown":
            last_chunk_seq = int(payload.get("last_chunk_seq", -1))
            with self._progress_cond:
                self._accepting_submissions = False
                self._progress_cond.notify_all()
            self._wait_until_chunk_processed(last_chunk_seq=last_chunk_seq)
            self.flush_ready_episode_markers()
            self._flush_transport_fn(
                "shutdown",
                bool(self._wait_committed_on_shutdown),
            )
            with self._progress_cond:
                self._stop_requested = True
                self._progress_cond.notify_all()
            return {
                **self.status_snapshot(),
                "transport": self._transport_status_fn(),
            }
        if request_type != "submit-chunk":
            raise ValueError(f"unsupported processor request: {request_type}")
        chunk_seq_value = int(payload.get("chunk_seq", -1))
        with self._progress_lock:
            accepted_snapshot = int(self._accepted_chunk_seq)
            processed_snapshot = int(self._processed_chunk_seq)
            accepting_snapshot = bool(self._accepting_submissions)
            stop_snapshot = bool(self._stop_requested)
            if int(chunk_seq_value) <= int(self._accepted_chunk_seq):
                return {
                    "accepted_chunk_seq": int(accepted_snapshot),
                    "processed_chunk_seq": int(processed_snapshot),
                    "committed_chunk_seq": int(processed_snapshot),
                    "accepting_submissions": bool(accepting_snapshot),
                    "stop_requested": bool(stop_snapshot),
                    "queue_depth": int(self._raw_chunk_queue.qsize()),
                    "deduped": True,
                }
        while True:
            with self._progress_lock:
                if (not bool(self._accepting_submissions)) or bool(self._stop_requested):
                    raise RuntimeError("processor stopping")
            if self.should_stop():
                raise RuntimeError("processor stopping")
            try:
                self._raw_chunk_queue.put(dict(payload), timeout=0.1)
                with self._progress_cond:
                    self._accepted_chunk_seq = max(
                        int(self._accepted_chunk_seq),
                        int(chunk_seq_value),
                    )
                    accepted_snapshot = int(self._accepted_chunk_seq)
                    processed_snapshot = int(self._processed_chunk_seq)
                    accepting_snapshot = bool(self._accepting_submissions)
                    stop_snapshot = bool(self._stop_requested)
                    self._progress_cond.notify_all()
                return {
                    "accepted_chunk_seq": int(accepted_snapshot),
                    "processed_chunk_seq": int(processed_snapshot),
                    "committed_chunk_seq": int(processed_snapshot),
                    "accepting_submissions": bool(accepting_snapshot),
                    "stop_requested": bool(stop_snapshot),
                    "queue_depth": int(self._raw_chunk_queue.qsize()),
                    "deduped": False,
                }
            except queue.Full:
                continue
