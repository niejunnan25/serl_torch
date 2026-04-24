from __future__ import annotations

import hashlib
import json
import logging
import pickle
import queue
import threading
import time
import zlib
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Protocol

import zmq
from agentlace.data.data_store import DataStoreBase
from agentlace.trainer import TrainerClient as AgentlaceTrainerClient
from agentlace.trainer import TrainerConfig as AgentlaceTrainerConfig
from agentlace.trainer import TrainerServer as AgentlaceTrainerServer
from agentlace.zmq_wrapper.broadcast import BroadcastClient
from agentlace.zmq_wrapper.broadcast import BroadcastServer

from serl_launcher.data.batch_ops import pack_transition_batch

try:
    import lz4.frame as _lz4_frame
except ImportError:  # pragma: no cover - zlib fallback remains tested.
    _lz4_frame = None


TransportMode = str

SYNC_COMMIT_MODE = "sync_commit"
ASYNC_COMMIT_MODE = "async_commit"
SUPPORTED_TRANSPORT_MODES = (SYNC_COMMIT_MODE, ASYNC_COMMIT_MODE)


@dataclass(frozen=True, slots=True)
class TrainerTransportConfig:
    mode: TransportMode
    data_port: int
    control_timeout_ms: int
    data_queue_capacity: int
    data_socket_hwm: int
    commit_poll_ms: int
    wait_committed_on_episode_end: bool
    wait_committed_on_shutdown: bool


def validate_transport_mode(mode: Any) -> TransportMode:
    raw_mode = str(mode)
    if raw_mode not in SUPPORTED_TRANSPORT_MODES:
        allowed = ", ".join(repr(name) for name in SUPPORTED_TRANSPORT_MODES)
        raise ValueError(
            f"Unsupported trainer transport mode: {raw_mode!r}. Allowed values: {allowed}"
        )
    return raw_mode


class ActorTrainerTransport(Protocol):
    def recv_network_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        ...

    def update(self) -> bool:
        ...

    def request(self, type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        ...

    def wait_until_committed(self, timeout_ms: int | None = None) -> bool:
        ...

    def get_transport_status(self, store_name: str | None = None) -> dict[str, Any]:
        ...

    def stop(self) -> None:
        ...


class LearnerTrainerTransport(Protocol):
    def register_data_store(self, name: str, data_store: DataStoreBase) -> None:
        ...

    def publish_network(self, payload: dict[str, Any]) -> None:
        ...

    def start(self, threaded: bool = False) -> None:
        ...

    def get_transport_status(self, store_name: str) -> dict[str, Any]:
        ...

    def stop(self) -> None:
        ...


def build_actor_trainer_transport(
    *,
    store_name: str,
    server_ip: str,
    trainer_port: int,
    broadcast_port: int,
    transport_cfg: Any,
    data_store: DataStoreBase,
    request_types: Iterable[str] = (),
    wait_for_server: bool = False,
    log_level: int = logging.INFO,
) -> ActorTrainerTransport:
    mode = validate_transport_mode(getattr(transport_cfg, "mode", SYNC_COMMIT_MODE))
    if mode == SYNC_COMMIT_MODE:
        return _SyncCommitActorTransport(
            store_name=store_name,
            server_ip=server_ip,
            trainer_port=int(trainer_port),
            broadcast_port=int(broadcast_port),
            data_store=data_store,
            request_types=tuple(request_types),
            wait_for_server=bool(wait_for_server),
            timeout_ms=int(getattr(transport_cfg, "control_timeout_ms", 800)),
            log_level=int(log_level),
        )
    if mode == ASYNC_COMMIT_MODE:
        return _AsyncCommitActorTransport(
            store_name=store_name,
            server_ip=server_ip,
            trainer_port=int(trainer_port),
            broadcast_port=int(broadcast_port),
            data_port=int(getattr(transport_cfg, "data_port")),
            data_store=data_store,
            request_types=tuple(request_types),
            wait_for_server=bool(wait_for_server),
            timeout_ms=int(getattr(transport_cfg, "control_timeout_ms", 800)),
            commit_poll_ms=int(getattr(transport_cfg, "commit_poll_ms", 20)),
            data_socket_hwm=int(getattr(transport_cfg, "data_socket_hwm", 8)),
            log_level=int(log_level),
        )
    raise ValueError(f"Unsupported trainer transport mode: {mode!r}")


def build_learner_trainer_transport(
    *,
    trainer_port: int,
    broadcast_port: int,
    transport_cfg: Any,
    request_callback: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    request_types: Iterable[str] = (),
    log_level: int = logging.INFO,
) -> LearnerTrainerTransport:
    mode = validate_transport_mode(getattr(transport_cfg, "mode", SYNC_COMMIT_MODE))
    if mode == SYNC_COMMIT_MODE:
        return _SyncCommitLearnerTransport(
            trainer_port=int(trainer_port),
            broadcast_port=int(broadcast_port),
            request_callback=request_callback,
            request_types=tuple(request_types),
            log_level=int(log_level),
        )
    if mode == ASYNC_COMMIT_MODE:
        return _AsyncCommitLearnerTransport(
            trainer_port=int(trainer_port),
            broadcast_port=int(broadcast_port),
            data_port=int(getattr(transport_cfg, "data_port")),
            request_callback=request_callback,
            request_types=tuple(request_types),
            data_queue_capacity=int(getattr(transport_cfg, "data_queue_capacity", 8)),
            data_socket_hwm=int(getattr(transport_cfg, "data_socket_hwm", 8)),
            log_level=int(log_level),
        )
    raise ValueError(f"Unsupported trainer transport mode: {mode!r}")


def _serialize_message(message: Any) -> bytes:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if _lz4_frame is not None:
        return _lz4_frame.compress(payload)
    return zlib.compress(payload)


def _deserialize_message(payload: bytes) -> Any:
    if _lz4_frame is not None:
        raw = _lz4_frame.decompress(payload)
    else:
        raw = zlib.decompress(payload)
    return pickle.loads(raw)


def _transport_config_hash(config: Mapping[str, Any]) -> str:
    config_json = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _close_zmq_socket(socket_obj) -> None:
    if socket_obj is None:
        return
    try:
        socket_obj.close(linger=0)
    except Exception:
        try:
            socket_obj.close()
        except Exception:
            pass


def _close_broadcast_endpoint(endpoint: Any) -> None:
    socket_obj = getattr(endpoint, "socket", None)
    context_obj = getattr(endpoint, "context", None)
    _close_zmq_socket(socket_obj)
    if context_obj is not None:
        try:
            context_obj.term()
        except Exception:
            pass


def _close_reqrep_client_endpoint(endpoint: Any) -> None:
    socket_obj = getattr(endpoint, "socket", None)
    context_obj = getattr(endpoint, "context", None)
    _close_zmq_socket(socket_obj)
    if context_obj is not None:
        try:
            context_obj.term()
        except Exception:
            pass


class _ReqRepServer:
    def __init__(
        self,
        *,
        port: int,
        callback: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._callback = callback
        self._stop_event = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{int(port)}")
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
                request = _deserialize_message(payload)
                response = self._callback(dict(request))
            except Exception as exc:  # noqa: BLE001
                response = {"success": False, "message": str(exc)}
            try:
                self._socket.send(_serialize_message(response))
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break

    def start(self, *, threaded: bool) -> None:
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


class _ReqRepClient:
    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        timeout_ms: int,
    ) -> None:
        self._server_ip = str(server_ip)
        self._port = int(port)
        self._timeout_ms = int(timeout_ms)
        self._lock = threading.Lock()
        self._context: zmq.Context | None = None
        self._socket = None
        self._connect()

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

    def send_msg(self, message: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            try:
                self._socket.send(_serialize_message(message))
                return _deserialize_message(self._socket.recv())
            except zmq.Again:
                self._reset()
                return None
            except zmq.ZMQError:
                self._reset()
                return None

    def close(self) -> None:
        with self._lock:
            _close_zmq_socket(self._socket)
            if self._context is not None:
                try:
                    self._context.term()
                except Exception:
                    pass


class _PushClient:
    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        hwm: int,
        timeout_ms: int,
    ) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, int(hwm))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{server_ip}:{int(port)}")
        self._lock = threading.Lock()

    def send_msg(self, message: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self._socket.send(_serialize_message(message))
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


class _PullServer:
    def __init__(
        self,
        *,
        port: int,
        callback: Callable[[dict[str, Any]], None],
        hwm: int,
    ) -> None:
        self._callback = callback
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
                request = _deserialize_message(payload)
                self._callback(dict(request))
            except Exception:
                continue

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


class _SyncCommitActorTransport:
    def __init__(
        self,
        *,
        store_name: str,
        server_ip: str,
        trainer_port: int,
        broadcast_port: int,
        data_store: DataStoreBase,
        request_types: tuple[str, ...],
        wait_for_server: bool,
        timeout_ms: int,
        log_level: int,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(int(log_level))
        self._store_name = str(store_name)
        self._data_store = data_store
        self._consecutive_update_misses = 0
        self._next_update_warning_time = 0.0
        self._client = AgentlaceTrainerClient(
            self._store_name,
            server_ip,
            AgentlaceTrainerConfig(
                port_number=int(trainer_port),
                broadcast_port=int(broadcast_port),
                request_types=list(request_types),
            ),
            data_store,
            wait_for_server=bool(wait_for_server),
            timeout_ms=int(timeout_ms),
            log_level=int(log_level),
        )

    def recv_network_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._client.recv_network_callback(callback)

    def update(self) -> bool:
        ok = bool(self._client.update())
        if ok:
            if int(self._consecutive_update_misses) > 0:
                self._logger.info(
                    "sync_commit update recovered: store=%s consecutive_misses=%s",
                    self._store_name,
                    int(self._consecutive_update_misses),
                )
            self._consecutive_update_misses = 0
            return True

        self._consecutive_update_misses += 1
        now = time.monotonic()
        if (
            int(self._consecutive_update_misses) == 1
            or now >= float(self._next_update_warning_time)
        ):
            self._logger.warning(
                "sync_commit update missed ack; continuing with SERL-style "
                "best-effort semantics: store=%s consecutive_misses=%s "
                "local_latest_data_id=%s",
                self._store_name,
                int(self._consecutive_update_misses),
                int(self._data_store.latest_data_id()),
            )
            self._next_update_warning_time = now + 30.0

        # Original SERL actors call TrainerClient.update() as a best-effort
        # flush and do not abort when agentlace reports a transient missing ack.
        # The next update asks the learner for its last_update_id, so any
        # unacknowledged local range is retried instead of being dropped.
        return True

    def request(self, type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        return self._client.request(type, payload)

    def wait_until_committed(self, timeout_ms: int | None = None) -> bool:
        del timeout_ms
        return True

    def get_transport_status(self, store_name: str | None = None) -> dict[str, Any]:
        target_name = str(store_name or self._store_name)
        remote_last = self._client.get_server_last_update_id(target_name)
        accepted = -1 if remote_last is None else int(remote_last)
        return {
            "transport_mode": SYNC_COMMIT_MODE,
            "store_name": target_name,
            "accepted_update_id": int(accepted),
            "committed_update_id": int(accepted),
            "transport_backlog": 0,
            "data_queue_depth": 0,
            "local_latest_data_id": int(self._data_store.latest_data_id()),
        }

    def stop(self) -> None:
        self._client.stop()
        _close_reqrep_client_endpoint(getattr(self._client, "req_rep_client", None))


class _SyncCommitLearnerTransport:
    def __init__(
        self,
        *,
        trainer_port: int,
        broadcast_port: int,
        request_callback: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
        request_types: tuple[str, ...],
        log_level: int,
    ) -> None:
        self._server = AgentlaceTrainerServer(
            AgentlaceTrainerConfig(
                port_number=int(trainer_port),
                broadcast_port=int(broadcast_port),
                request_types=list(request_types),
            ),
            request_callback=request_callback,
            log_level=int(log_level),
        )

    def register_data_store(self, name: str, data_store: DataStoreBase) -> None:
        self._server.register_data_store(name, data_store)

    def publish_network(self, payload: dict[str, Any]) -> None:
        self._server.publish_network(payload)

    def start(self, threaded: bool = False) -> None:
        self._server.start(threaded=bool(threaded))

    def get_transport_status(self, store_name: str) -> dict[str, Any]:
        last_update_id = int(self._server.last_update_id_map.get(str(store_name), -1))
        return {
            "transport_mode": SYNC_COMMIT_MODE,
            "store_name": str(store_name),
            "accepted_update_id": int(last_update_id),
            "committed_update_id": int(last_update_id),
            "transport_backlog": 0,
            "data_queue_depth": 0,
        }

    def stop(self) -> None:
        self._server.stop()
        _close_broadcast_endpoint(getattr(self._server, "broadcast_server", None))


class _AsyncCommitLearnerTransport:
    def __init__(
        self,
        *,
        trainer_port: int,
        broadcast_port: int,
        data_port: int,
        request_callback: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
        request_types: tuple[str, ...],
        data_queue_capacity: int,
        data_socket_hwm: int,
        log_level: int,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(int(log_level))
        self._request_callback = request_callback
        self._request_types = set(request_types)
        self._data_stores: dict[str, DataStoreBase] = {}
        self._accepted_update_id_map: dict[str, int] = {}
        self._committed_update_id_map: dict[str, int] = {}
        self._progress_lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, int, Any]] = queue.Queue(
            maxsize=int(data_queue_capacity)
        )
        self._stop_event = threading.Event()
        self._transport_signature = {
            "mode": ASYNC_COMMIT_MODE,
            "trainer_port": int(trainer_port),
            "broadcast_port": int(broadcast_port),
            "data_port": int(data_port),
            "request_types": list(sorted(self._request_types)),
        }

        def _control_callback(message: dict[str, Any]) -> dict[str, Any]:
            msg_type = str(message.get("type", ""))
            payload = dict(message.get("payload", {}) or {})
            if msg_type == "hash":
                return {"success": True, "payload": _transport_config_hash(self._transport_signature)}
            if msg_type == "get_accepted_update_id":
                store_name = str(payload.get("store_name", ""))
                return {
                    "success": True,
                    "payload": int(self._accepted_update_id_map.get(store_name, -1)),
                }
            if msg_type == "get_committed_update_id":
                store_name = str(payload.get("store_name", ""))
                return {
                    "success": True,
                    "payload": int(self._committed_update_id_map.get(store_name, -1)),
                }
            if msg_type == "get_transport_status":
                store_name = str(payload.get("store_name", ""))
                return {
                    "success": True,
                    "payload": self.get_transport_status(store_name),
                }
            if msg_type in self._request_types:
                if self._request_callback is None:
                    return {"success": True, "payload": {}}
                return {"success": True, "payload": self._request_callback(msg_type, payload)}
            return {"success": False, "message": f"Unsupported request type: {msg_type}"}

        def _data_callback(message: dict[str, Any]) -> None:
            if str(message.get("type", "")) != "datastore":
                return
            store_name = str(message.get("store_name", ""))
            payload = dict(message.get("payload", {}) or {})
            last_id = int(payload.get("last_id", -1))
            batch_data = payload.get("data", [])
            if store_name not in self._data_stores:
                return
            with self._progress_lock:
                if last_id <= int(self._accepted_update_id_map.get(store_name, -1)):
                    return
            while not self._stop_event.is_set():
                try:
                    self._queue.put((store_name, last_id, batch_data), timeout=0.1)
                    with self._progress_lock:
                        self._accepted_update_id_map[store_name] = max(
                            int(self._accepted_update_id_map.get(store_name, -1)),
                            int(last_id),
                        )
                    return
                except queue.Full:
                    continue

        self._reqrep_server = _ReqRepServer(
            port=int(trainer_port),
            callback=_control_callback,
        )
        self._pull_server = _PullServer(
            port=int(data_port),
            callback=_data_callback,
            hwm=int(data_socket_hwm),
        )
        self._broadcast_server = BroadcastServer(
            int(broadcast_port),
            log_level=int(log_level),
        )
        self._worker_thread = threading.Thread(
            target=self._commit_worker,
            daemon=True,
            name="trainer-async-commit-worker",
        )

    def _commit_worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                store_name, last_id, batch_data = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._data_stores[store_name].batch_insert(batch_data)
                with self._progress_lock:
                    self._committed_update_id_map[store_name] = max(
                        int(self._committed_update_id_map.get(store_name, -1)),
                        int(last_id),
                    )
            finally:
                self._queue.task_done()

    def register_data_store(self, name: str, data_store: DataStoreBase) -> None:
        store_name = str(name)
        self._data_stores[store_name] = data_store
        with self._progress_lock:
            self._accepted_update_id_map[store_name] = -1
            self._committed_update_id_map[store_name] = -1

    def publish_network(self, payload: dict[str, Any]) -> None:
        self._broadcast_server.broadcast(payload)

    def start(self, threaded: bool = False) -> None:
        self._worker_thread.start()
        self._pull_server.start()
        self._reqrep_server.start(threaded=bool(threaded))

    def get_transport_status(self, store_name: str) -> dict[str, Any]:
        with self._progress_lock:
            accepted = int(self._accepted_update_id_map.get(str(store_name), -1))
            committed = int(self._committed_update_id_map.get(str(store_name), -1))
        return {
            "transport_mode": ASYNC_COMMIT_MODE,
            "store_name": str(store_name),
            "accepted_update_id": int(accepted),
            "committed_update_id": int(committed),
            "transport_backlog": int(max(0, accepted - committed)),
            "data_queue_depth": int(self._queue.qsize()),
        }

    def stop(self) -> None:
        self._stop_event.set()
        self._pull_server.stop()
        self._reqrep_server.stop()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        _close_broadcast_endpoint(self._broadcast_server)


class _AsyncCommitActorTransport:
    def __init__(
        self,
        *,
        store_name: str,
        server_ip: str,
        trainer_port: int,
        broadcast_port: int,
        data_port: int,
        data_store: DataStoreBase,
        request_types: tuple[str, ...],
        wait_for_server: bool,
        timeout_ms: int,
        commit_poll_ms: int,
        data_socket_hwm: int,
        log_level: int,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(int(log_level))
        self._store_name = str(store_name)
        self._data_store = data_store
        self._request_types = set(request_types)
        self._timeout_ms = int(timeout_ms)
        self._commit_poll_ms = max(1, int(commit_poll_ms))
        self._control_client = _ReqRepClient(
            server_ip=server_ip,
            port=int(trainer_port),
            timeout_ms=int(timeout_ms),
        )
        self._push_client = _PushClient(
            server_ip=server_ip,
            port=int(data_port),
            hwm=int(data_socket_hwm),
            timeout_ms=int(timeout_ms),
        )
        self._broadcast_client: BroadcastClient | None = None
        self._transport_signature = {
            "mode": ASYNC_COMMIT_MODE,
            "trainer_port": int(trainer_port),
            "broadcast_port": int(broadcast_port),
            "data_port": int(data_port),
            "request_types": list(sorted(self._request_types)),
        }
        self._broadcast_port = int(broadcast_port)
        self._server_ip = str(server_ip)
        self._acked_id = -1
        self._pending_message: dict[str, Any] | None = None
        self._pending_last_id: int | None = None
        self._pending_local_latest_id: int | None = None

        self._wait_for_server(wait_for_server=bool(wait_for_server))
        accepted = self.get_server_accepted_update_id(self._store_name)
        self._acked_id = -1 if accepted is None else int(accepted)
        self._pending_message = None
        self._pending_last_id = None
        self._pending_local_latest_id = None

    @property
    def _target_last_sent_id(self) -> int:
        if self._pending_last_id is not None:
            return int(self._pending_last_id)
        return int(self._acked_id)

    def _wait_for_server(self, *, wait_for_server: bool) -> None:
        response = self._control_client.send_msg({"type": "hash"})
        while wait_for_server and response is None:
            time.sleep(2.0)
            response = self._control_client.send_msg({"type": "hash"})
        if response is None or not response.get("success", False):
            raise RuntimeError("Failed to connect to async_commit trainer server")
        expected_hash = _transport_config_hash(self._transport_signature)
        if response.get("payload") != expected_hash:
            raise RuntimeError("Incompatible trainer transport config between actor and learner")

    def _send_pending_or_new(self) -> bool:
        if self._pending_message is not None:
            return self._push_client.send_msg(self._pending_message)
        local_latest_id = int(self._data_store.latest_data_id())
        if local_latest_id <= int(self._acked_id):
            return True
        batch_data = self._data_store.get_latest_data(int(self._acked_id))
        if len(batch_data) == 0:
            return True
        packed_batch = pack_transition_batch(batch_data)
        self._pending_last_id = int(local_latest_id)
        self._pending_local_latest_id = int(local_latest_id)
        self._pending_message = {
            "type": "datastore",
            "store_name": self._store_name,
            "payload": {
                "last_id": int(local_latest_id),
                "data": packed_batch,
            },
        }
        return self._push_client.send_msg(self._pending_message)

    def _refresh_accepted_id(self) -> int | None:
        remote = self.get_server_accepted_update_id(self._store_name)
        if remote is None:
            return None
        self._acked_id = max(int(self._acked_id), int(remote))
        if self._pending_last_id is not None and int(self._acked_id) >= int(self._pending_last_id):
            self._pending_message = None
            self._pending_last_id = None
            self._pending_local_latest_id = None
        return int(self._acked_id)

    def recv_network_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._broadcast_client = BroadcastClient(
            self._server_ip,
            self._broadcast_port,
            log_level=self._logger.level,
        )
        self._broadcast_client.async_start(callback)

    def get_server_accepted_update_id(self, name: str) -> int | None:
        response = self._control_client.send_msg(
            {
                "type": "get_accepted_update_id",
                "payload": {"store_name": str(name)},
            }
        )
        if response is None or not response.get("success", False):
            return None
        return int(response.get("payload", -1))

    def get_server_committed_update_id(self, name: str) -> int | None:
        response = self._control_client.send_msg(
            {
                "type": "get_committed_update_id",
                "payload": {"store_name": str(name)},
            }
        )
        if response is None or not response.get("success", False):
            return None
        return int(response.get("payload", -1))

    def update(self) -> bool:
        self._refresh_accepted_id()
        if self._pending_message is None:
            local_latest_id = int(self._data_store.latest_data_id())
            if local_latest_id <= int(self._acked_id):
                return True
        if not self._send_pending_or_new():
            return False
        deadline = time.monotonic() + (float(self._timeout_ms) / 1000.0)
        target_id = int(self._pending_last_id) if self._pending_last_id is not None else int(self._acked_id)
        while time.monotonic() <= deadline:
            accepted = self._refresh_accepted_id()
            if accepted is not None and int(accepted) >= int(target_id):
                return True
            time.sleep(float(self._commit_poll_ms) / 1000.0)
        return False

    def request(self, type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if type not in self._request_types:
            return None
        response = self._control_client.send_msg({"type": str(type), "payload": payload})
        if response is None or not response.get("success", False):
            return None
        raw_payload = response.get("payload", {})
        return raw_payload if isinstance(raw_payload, dict) else {"payload": raw_payload}

    def wait_until_committed(self, timeout_ms: int | None = None) -> bool:
        timeout_ms = int(self._timeout_ms if timeout_ms is None else timeout_ms)
        deadline = time.monotonic() + (float(timeout_ms) / 1000.0)
        target_id = int(self._target_last_sent_id)
        while time.monotonic() <= deadline:
            committed = self.get_server_committed_update_id(self._store_name)
            if committed is not None and int(committed) >= int(target_id):
                return True
            time.sleep(float(self._commit_poll_ms) / 1000.0)
        return False

    def get_transport_status(self, store_name: str | None = None) -> dict[str, Any]:
        target_name = str(store_name or self._store_name)
        response = self._control_client.send_msg(
            {
                "type": "get_transport_status",
                "payload": {"store_name": target_name},
            }
        )
        payload = (
            dict(response.get("payload", {}))
            if response is not None and response.get("success", False)
            else {}
        )
        payload.update(
            {
                "local_latest_data_id": int(self._data_store.latest_data_id()),
                "local_acked_update_id": int(self._acked_id),
                "local_pending_update_id": (
                    None if self._pending_last_id is None else int(self._pending_last_id)
                ),
                "last_sent_id": int(self._target_last_sent_id),
            }
        )
        return payload

    def stop(self) -> None:
        if self._broadcast_client is not None:
            self._broadcast_client.stop()
        self._control_client.close()
        self._push_client.close()
