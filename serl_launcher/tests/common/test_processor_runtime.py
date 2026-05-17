from __future__ import annotations

import logging
import socket
import threading
import time

import pytest

from serl_launcher.rollout import ProcessorClient
from serl_launcher.rollout import ProcessorServer
from serl_launcher.rollout import ProcessorTransportConfig


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_processor_client_wait_until_ready_times_out() -> None:
    client = ProcessorClient(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=_find_free_port(),
            timeout_ms=50,
            queue_capacity=2,
        ),
        logger=logging.getLogger(__name__),
    )
    try:
        with pytest.raises(RuntimeError, match="Timed out waiting for processor server"):
            client.wait_until_ready(timeout_s=0.2, poll_interval_s=0.01)
    finally:
        client.close()


def test_processor_submit_fails_after_bounded_retries() -> None:
    client = ProcessorClient(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=_find_free_port(),
            timeout_ms=5,
            queue_capacity=2,
        ),
        logger=logging.getLogger(__name__),
    )
    client._submit_retry_limit = 2
    try:
        with pytest.raises(
            RuntimeError,
            match=r"processor request 'submit-chunk' failed repeatedly",
        ):
            client.submit(payload={"chunk_seq": 1}, context="missing_server")
    finally:
        client.close()


def test_processor_roundtrip_and_flush() -> None:
    port = _find_free_port()
    flush_calls: list[tuple[str, bool]] = []
    server = ProcessorServer(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=200,
            queue_capacity=2,
        ),
        transport_status_fn=lambda: {"ready": True},
        flush_transport_fn=lambda context, wait_until_committed: flush_calls.append(
            (str(context), bool(wait_until_committed))
        ),
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )
    server.start()
    client = ProcessorClient(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=200,
            queue_capacity=2,
        ),
        logger=logging.getLogger(__name__),
    )
    try:
        client.wait_until_ready(timeout_s=1.0, poll_interval_s=0.01)

        submit_response = client.submit(
            payload={"chunk_seq": 3, "episode_id": 1},
            context="submit",
        )
        assert submit_response["accepted_chunk_seq"] == 3
        assert submit_response["processed_chunk_seq"] == -1
        assert submit_response["committed_chunk_seq"] == -1
        assert submit_response["deduped"] is False

        queued_payload = server.get_chunk(timeout_s=1.0)
        assert queued_payload == {"chunk_seq": 3, "episode_id": 1}
        server.mark_chunk_committed(chunk_seq=3)
        server.task_done()

        finish_response = client.finish(episode_id=1, last_chunk_seq=3)
        assert finish_response["processed_chunk_seq"] == 3
        assert finish_response["committed_chunk_seq"] == 3
        assert finish_response["transport"] == {"ready": True}
        assert flush_calls == [("episode_1_end", False)]

        duplicate_response = client.submit(
            payload={"chunk_seq": 3, "episode_id": 1},
            context="duplicate",
        )
        assert duplicate_response["deduped"] is True

        shutdown_response = client.shutdown(last_chunk_seq=3)
        assert shutdown_response["stop_requested"] is True
        assert flush_calls == [("episode_1_end", False), ("shutdown", True)]
    finally:
        client.close()
        server.stop()


def test_processor_submit_tolerates_brief_queue_backpressure() -> None:
    port = _find_free_port()
    server = ProcessorServer(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=50,
            queue_capacity=1,
        ),
        transport_status_fn=lambda: {"ready": True},
        flush_transport_fn=lambda context, wait_until_committed: None,
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )
    server.start()
    client = ProcessorClient(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=50,
            queue_capacity=1,
        ),
        logger=logging.getLogger(__name__),
    )
    try:
        client.wait_until_ready(timeout_s=1.0, poll_interval_s=0.01)

        first_response = client.submit(
            payload={"chunk_seq": 0, "episode_id": 1},
            context="first",
        )
        assert first_response["accepted_chunk_seq"] == 0

        release_done = threading.Event()
        released_payload: list[dict[str, object]] = []

        def _release_queue_slot() -> None:
            time.sleep(0.7)
            first_payload = server.get_chunk(timeout_s=1.0)
            released_payload.append(dict(first_payload or {}))
            server.mark_chunk_committed(chunk_seq=0)
            server.task_done()
            release_done.set()

        releaser = threading.Thread(target=_release_queue_slot, daemon=True)
        releaser.start()

        second_response = client.submit(
            payload={"chunk_seq": 1, "episode_id": 1},
            context="second",
        )
        releaser.join(timeout=1.0)
        assert release_done.is_set()
        assert released_payload == [{"chunk_seq": 0, "episode_id": 1}]
        assert second_response["accepted_chunk_seq"] == 1
        assert second_response["deduped"] is True

        second_payload = server.get_chunk(timeout_s=1.0)
        assert second_payload == {"chunk_seq": 1, "episode_id": 1}
        server.mark_chunk_committed(chunk_seq=1)
        server.task_done()
    finally:
        client.close()
        server.stop()


def test_processor_finish_fails_when_flush_transport_raises() -> None:
    port = _find_free_port()

    def _failing_flush(context: str, wait_until_committed: bool) -> None:
        del wait_until_committed
        raise RuntimeError(f"flush failed: {str(context)}")

    server = ProcessorServer(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=20_000,
            queue_capacity=2,
        ),
        transport_status_fn=lambda: {"ready": True},
        flush_transport_fn=_failing_flush,
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )
    server.start()
    client = ProcessorClient(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=20_000,
            queue_capacity=2,
        ),
        logger=logging.getLogger(__name__),
    )
    try:
        client.wait_until_ready(timeout_s=1.0, poll_interval_s=0.01)
        submit_response = client.submit(
            payload={"chunk_seq": 7, "episode_id": 2},
            context="submit",
        )
        assert submit_response["accepted_chunk_seq"] == 7

        queued_payload = server.get_chunk(timeout_s=1.0)
        assert queued_payload == {"chunk_seq": 7, "episode_id": 2}
        server.mark_chunk_committed(chunk_seq=7)
        server.task_done()

        with pytest.raises(
            RuntimeError,
            match=r"processor request 'finish-episode' failed repeatedly",
        ):
            client.finish(episode_id=2, last_chunk_seq=7)
    finally:
        client.close()
        server.stop()


def test_processor_mark_episode_end_defers_flush_until_chunk_committed() -> None:
    port = _find_free_port()
    flush_calls: list[tuple[str, bool]] = []
    server = ProcessorServer(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=200,
            queue_capacity=2,
        ),
        transport_status_fn=lambda: {"ready": True},
        flush_transport_fn=lambda context, wait_until_committed: flush_calls.append(
            (str(context), bool(wait_until_committed))
        ),
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )
    server.start()
    client = ProcessorClient(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=200,
            queue_capacity=2,
        ),
        logger=logging.getLogger(__name__),
    )
    try:
        client.wait_until_ready(timeout_s=1.0, poll_interval_s=0.01)

        submit_response = client.submit(
            payload={"chunk_seq": 4, "episode_id": 9},
            context="submit",
        )
        assert submit_response["accepted_chunk_seq"] == 4

        mark_response = client.mark_episode_end(episode_id=9, last_chunk_seq=4)
        assert mark_response["pending_episode_flushes"] == 1
        assert mark_response["deduped"] is False
        assert flush_calls == []

        queued_payload = server.get_chunk(timeout_s=1.0)
        assert queued_payload == {"chunk_seq": 4, "episode_id": 9}
        server.mark_chunk_committed(chunk_seq=4)
        server.flush_ready_episode_markers()
        server.task_done()

        assert flush_calls == [("episode_9_end", False)]
        assert server.consume_flushed_episode_markers() == [
            {"episode_id": 9, "last_chunk_seq": 4}
        ]

        duplicate_mark_response = client.mark_episode_end(episode_id=9, last_chunk_seq=4)
        assert duplicate_mark_response["deduped"] is True
    finally:
        client.close()
        server.stop()


def test_processor_flush_failure_keeps_pending_episode_marker() -> None:
    port = _find_free_port()
    flush_calls: list[tuple[str, bool]] = []

    def _flaky_flush(context: str, wait_until_committed: bool) -> None:
        flush_calls.append((str(context), bool(wait_until_committed)))
        if len(flush_calls) == 1:
            raise RuntimeError("flush failed once")

    server = ProcessorServer(
        transport_config=ProcessorTransportConfig(
            host="127.0.0.1",
            port=port,
            timeout_ms=200,
            queue_capacity=2,
        ),
        transport_status_fn=lambda: {"ready": True},
        flush_transport_fn=_flaky_flush,
        wait_committed_on_episode_end=False,
        wait_committed_on_shutdown=True,
    )
    server._accepted_chunk_seq = 6
    server._processed_chunk_seq = 6
    server._last_marked_episode_chunk_seq = 6
    server._pending_episode_flush_markers.append(
        {"episode_id": 3, "last_chunk_seq": 6}
    )

    try:
        with pytest.raises(RuntimeError, match="flush failed once"):
            server.flush_ready_episode_markers()
        assert server.status_snapshot()["pending_episode_flushes"] == 1
        assert server.consume_flushed_episode_markers() == []

        server.flush_ready_episode_markers()
        assert flush_calls == [("episode_3_end", False), ("episode_3_end", False)]
        assert server.status_snapshot()["pending_episode_flushes"] == 0
        assert server.consume_flushed_episode_markers() == [
            {"episode_id": 3, "last_chunk_seq": 6}
        ]
    finally:
        server.stop()
