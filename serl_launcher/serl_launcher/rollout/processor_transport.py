from __future__ import annotations

"""Thin rollout processor transport with split control/data planes."""

import pickle
import threading
import time
import zlib
from typing import Any
from typing import Callable

import zmq


def _serialize_message(message: Any) -> bytes:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(payload)


def _deserialize_message(payload: bytes) -> Any:
    return pickle.loads(zlib.decompress(payload))


def _close_zmq_socket(socket_obj: Any) -> None:
    if socket_obj is None:
        return
    try:
        socket_obj.close(linger=0)
    except Exception:
        try:
            socket_obj.close()
        except Exception:
            pass


class RolloutProcessorControlServer:
    def __init__(
        self,
        *,
        port: int,
        callback: Callable[[str, dict[str, Any]], dict[str, Any] | None],
    ) -> None:
        self._port = int(port)
        self._callback = callback
        self._stop_event = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{self._port}")
        self._thread: threading.Thread | None = None

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break
                continue
            try:
                request = dict(_deserialize_message(payload))
                request_type = str(request.get("type", ""))
                request_payload = dict(request.get("payload", {}) or {})
                if request_type == "ping":
                    response: dict[str, Any] = {
                        "success": True,
                        "payload": {"port": int(self._port)},
                    }
                else:
                    raw_response = self._callback(request_type, request_payload)
                    response = {
                        "success": True,
                        "payload": {} if raw_response is None else raw_response,
                    }
            except Exception as exc:  # noqa: BLE001
                response = {"success": False, "message": str(exc)}
            try:
                self._socket.send(_serialize_message(response))
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break

    def start(self, *, threaded: bool = True) -> None:
        if threaded:
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            return
        self._serve()

    def stop(self) -> None:
        self._stop_event.set()
        _close_zmq_socket(self._socket)
        try:
            self._context.term()
        except Exception:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


class RolloutProcessorControlClient:
    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        timeout_ms: int,
        wait_for_server: bool = False,
    ) -> None:
        self._server_ip = str(server_ip)
        self._port = int(port)
        self._timeout_ms = int(timeout_ms)
        self._lock = threading.Lock()
        self._context: zmq.Context | None = None
        self._socket = None
        self._connect()
        self._wait_for_server(wait_for_server=bool(wait_for_server))

    def _connect(self) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(self._timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(self._timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._server_ip}:{self._port}")

    def _reset(self) -> None:
        _close_zmq_socket(self._socket)
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
        self._connect()

    def _request_locked(
        self,
        request_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        try:
            self._socket.send(
                _serialize_message(
                    {
                        "type": str(request_type),
                        "payload": {} if payload is None else dict(payload),
                    }
                )
            )
            return dict(_deserialize_message(self._socket.recv()))
        except zmq.Again:
            self._reset()
            return None
        except zmq.ZMQError:
            self._reset()
            return None

    def _wait_for_server(self, *, wait_for_server: bool) -> None:
        response = self.request("ping", {})
        while wait_for_server and response is None:
            time.sleep(1.0)
            response = self.request("ping", {})
        if wait_for_server and response is None:
            raise RuntimeError("Failed to connect to rollout processor server")

    def request(
        self,
        request_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            response = self._request_locked(request_type, payload)
        if response is None or not bool(response.get("success", False)):
            return None
        raw_payload = response.get("payload", {})
        return raw_payload if isinstance(raw_payload, dict) else {"payload": raw_payload}

    def get_status(self) -> dict[str, Any] | None:
        return self.request("status", {})

    def close(self) -> None:
        with self._lock:
            _close_zmq_socket(self._socket)
            if self._context is not None:
                try:
                    self._context.term()
                except Exception:
                    pass


class RolloutProcessorDataClient:
    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        timeout_ms: int,
        hwm: int = 8,
    ) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, int(hwm))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{server_ip}:{int(port)}")
        self._lock = threading.Lock()

    def send(self, payload: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self._socket.send(_serialize_message(dict(payload)))
                return True
            except zmq.Again:
                return False
            except zmq.ZMQError:
                return False

    def close(self) -> None:
        with self._lock:
            _close_zmq_socket(self._socket)
            try:
                self._context.term()
            except Exception:
                pass


class RolloutProcessorDataServer:
    def __init__(
        self,
        *,
        port: int,
        callback: Callable[[dict[str, Any]], None],
        hwm: int = 8,
        error_callback: Callable[[Exception], None] | None = None,
    ) -> None:
        self._callback = callback
        self._error_callback = error_callback
        self._stop_event = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, int(hwm))
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{int(port)}")
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break
                continue
            try:
                request = dict(_deserialize_message(payload))
                self._callback(request)
            except Exception as exc:
                if self._error_callback is not None:
                    try:
                        self._error_callback(exc)
                    except Exception:
                        pass
                self._stop_event.set()
                break

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        _close_zmq_socket(self._socket)
        try:
            self._context.term()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


RolloutProcessorServer = RolloutProcessorControlServer
RolloutProcessorClient = RolloutProcessorControlClient

__all__ = [
    "RolloutProcessorClient",
    "RolloutProcessorServer",
    "RolloutProcessorControlClient",
    "RolloutProcessorControlServer",
    "RolloutProcessorDataClient",
    "RolloutProcessorDataServer",
]
