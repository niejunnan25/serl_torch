from __future__ import annotations

import socket
import time

from serl_launcher.rollout.processor_transport import RolloutProcessorControlClient
from serl_launcher.rollout.processor_transport import RolloutProcessorControlServer
from serl_launcher.rollout.processor_transport import RolloutProcessorDataClient
from serl_launcher.rollout.processor_transport import RolloutProcessorDataServer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_rollout_processor_transport_roundtrip() -> None:
    port = _find_free_port()
    seen_requests: list[tuple[str, dict[str, object]]] = []

    def _callback(
        request_type: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        seen_requests.append((request_type, dict(payload)))
        if request_type == "status":
            return {"ready": True, "count": len(seen_requests)}
        return {"echo": dict(payload), "type": str(request_type)}

    server = RolloutProcessorControlServer(port=port, callback=_callback)
    server.start(threaded=True)
    client = RolloutProcessorControlClient(
        server_ip="127.0.0.1",
        port=port,
        timeout_ms=1000,
        wait_for_server=True,
    )
    try:
        status = client.get_status()
        assert status is not None
        assert status["ready"] is True

        response = client.request("observe_chunk", {"chunk_seq": 3})
        assert response == {
            "echo": {"chunk_seq": 3},
            "type": "observe_chunk",
        }
        assert seen_requests == [
            ("status", {}),
            ("observe_chunk", {"chunk_seq": 3}),
        ]
    finally:
        client.close()
        server.stop()


def test_rollout_processor_data_transport_roundtrip() -> None:
    port = _find_free_port()
    seen_messages: list[dict[str, object]] = []

    def _callback(payload: dict[str, object]) -> None:
        seen_messages.append(dict(payload))

    server = RolloutProcessorDataServer(port=port, callback=_callback)
    server.start()
    client = RolloutProcessorDataClient(
        server_ip="127.0.0.1",
        port=port,
        timeout_ms=1000,
        hwm=4,
    )
    try:
        assert client.send({"type": "observe_chunk", "chunk_seq": 7}) is True
        deadline = time.time() + 2.0
        while time.time() < deadline and not seen_messages:
            time.sleep(0.01)
        assert seen_messages == [{"type": "observe_chunk", "chunk_seq": 7}]
    finally:
        client.close()
        server.stop()


def test_rollout_processor_data_transport_reports_callback_error() -> None:
    port = _find_free_port()
    seen_errors: list[str] = []

    def _callback(payload: dict[str, object]) -> None:
        raise RuntimeError(f"boom: {payload['chunk_seq']}")

    def _error_callback(exc: Exception) -> None:
        seen_errors.append(str(exc))

    server = RolloutProcessorDataServer(
        port=port,
        callback=_callback,
        error_callback=_error_callback,
    )
    server.start()
    client = RolloutProcessorDataClient(
        server_ip="127.0.0.1",
        port=port,
        timeout_ms=1000,
        hwm=4,
    )
    try:
        assert client.send({"type": "observe_chunk", "chunk_seq": 9}) is True
        deadline = time.time() + 2.0
        while time.time() < deadline and not seen_errors:
            time.sleep(0.01)
        assert seen_errors == ["boom: 9"]
    finally:
        client.close()
        server.stop()
