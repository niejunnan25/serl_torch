#!/usr/bin/env python3
"""Synthetic trainer-ingest benchmark with simulated learner load.

This benchmark focuses on the actor -> learner datastore update path:

1. actor builds a batch of transitions
2. serializes / compresses the payload
3. sends it over localhost ZMQ
4. learner deserializes it
5. learner inserts into replay while a background sampler thread simulates
   training-side replay reads

It compares the current synchronous req/rep path against a set of candidate
optimizations:

- vectorized replay batch insert
- packed ndarray payloads instead of list[dict]
- req/rep enqueue-and-ack
- split data/control transport using push/pull

Example:

    conda run -n serl_torch python test/benchmark_trainer_datastore_variants.py \
      --iterations 10 \
      --warmup 2 \
      --json-out test/results/benchmark_trainer_datastore_variants.json
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zmq

from agentlace.internal.utils import make_compression_method
from agentlace.zmq_wrapper.req_rep import ReqRepClient


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _close_reqrep_client(client: ReqRepClient) -> None:
    socket_obj = getattr(client, "socket", None)
    if socket_obj is not None:
        try:
            socket_obj.close()
        except Exception:
            pass
    context_obj = getattr(client, "context", None)
    if context_obj is not None:
        try:
            context_obj.term()
        except Exception:
            pass


def _nested_buffer_like(example: Any, capacity: int) -> Any:
    if isinstance(example, dict):
        return {key: _nested_buffer_like(value, capacity) for key, value in example.items()}
    array = np.asarray(example)
    return np.empty((int(capacity), *array.shape), dtype=array.dtype)


def _nested_assign_single(dst: Any, src: Any, index: int) -> None:
    if isinstance(dst, dict):
        for key in dst:
            _nested_assign_single(dst[key], src[key], index)
        return
    dst[int(index)] = src


def _nested_assign_slice(
    dst: Any,
    src: Any,
    dst_start: int,
    dst_stop: int,
    src_start: int,
    src_stop: int,
) -> None:
    if isinstance(dst, dict):
        for key in dst:
            _nested_assign_slice(
                dst[key],
                src[key],
                dst_start=dst_start,
                dst_stop=dst_stop,
                src_start=src_start,
                src_stop=src_stop,
            )
        return
    dst[int(dst_start) : int(dst_stop)] = src[int(src_start) : int(src_stop)]


def _nested_stack(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {key: _nested_stack([item[key] for item in items]) for key in first}
    return np.stack(items, axis=0)


def _nested_take(data: Any, index: int, *, copy_leaf: bool) -> Any:
    if isinstance(data, dict):
        return {key: _nested_take(value, index, copy_leaf=copy_leaf) for key, value in data.items()}
    return np.array(data[int(index)], copy=copy_leaf)


def _nested_sample(data: Any, indices: np.ndarray) -> Any:
    if isinstance(data, dict):
        return {key: _nested_sample(value, indices) for key, value in data.items()}
    return np.array(data[indices], copy=True)


def _compress_bytes(obj: Any) -> int:
    compress, _ = make_compression_method("lz4")
    return int(len(compress(obj)))


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_s": 0.0,
            "median_s": 0.0,
            "p95_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
        }
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


@dataclass(frozen=True)
class BenchmarkConfig:
    iterations: int
    warmup: int
    batch_size: int
    replay_capacity: int
    image_size: int
    proprio_dim: int
    action_dim: int
    stats_offset_ms: float
    timeout_ms: int
    sampler_batch_size: int
    sampler_sleep_ms: float
    sampler_hold_ms: float
    sampler_threads: int
    control_poll_ms: float
    commit_timeout_s: float
    queue_capacity: int
    random_seed: int


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    payload_kind: str
    replay_mode: str
    transport_mode: str
    update_id_semantics: str


class SyntheticReplayStore:
    def __init__(
        self,
        *,
        example_transition: dict[str, Any],
        capacity: int,
        vectorized: bool,
    ) -> None:
        self._capacity = int(capacity)
        self._vectorized = bool(vectorized)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0)
        self._size = 0
        self._insert_index = 0
        self._insert_count = 0
        self._sample_calls = 0
        self._sample_digest = 0.0
        self._dataset = {
            key: _nested_buffer_like(value, int(capacity))
            for key, value in example_transition.items()
        }

    @property
    def insert_count(self) -> int:
        return int(self._insert_count)

    @property
    def sample_calls(self) -> int:
        return int(self._sample_calls)

    def latest_committed_id(self) -> int:
        return int(self._insert_count - 1)

    def insert(self, transition: dict[str, Any]) -> None:
        with self._lock:
            self._insert_one_nolock(transition)

    def _insert_one_nolock(self, transition: dict[str, Any]) -> None:
        _nested_assign_single(self._dataset, transition, int(self._insert_index))
        self._insert_index = (int(self._insert_index) + 1) % int(self._capacity)
        self._insert_count += 1
        self._size = min(int(self._size) + 1, int(self._capacity))

    def _write_packed_nolock(self, packed: dict[str, Any]) -> None:
        batch_count = int(np.asarray(packed["actions"]).shape[0])
        if batch_count <= 0:
            return
        if batch_count > int(self._capacity):
            keep_from = int(batch_count - self._capacity)
            packed = {
                key: value[keep_from:] if not isinstance(value, dict) else self._slice_nested(value, keep_from, batch_count)
                for key, value in packed.items()
            }
            batch_count = int(self._capacity)

        first = min(int(batch_count), int(self._capacity) - int(self._insert_index))
        _nested_assign_slice(
            self._dataset,
            packed,
            dst_start=int(self._insert_index),
            dst_stop=int(self._insert_index + first),
            src_start=0,
            src_stop=int(first),
        )
        remaining = int(batch_count - first)
        if remaining > 0:
            _nested_assign_slice(
                self._dataset,
                packed,
                dst_start=0,
                dst_stop=int(remaining),
                src_start=int(first),
                src_stop=int(batch_count),
            )

        self._insert_index = (int(self._insert_index) + int(batch_count)) % int(self._capacity)
        self._insert_count += int(batch_count)
        self._size = min(int(self._size) + int(batch_count), int(self._capacity))

    def _slice_nested(self, value: Any, start: int, stop: int) -> Any:
        if isinstance(value, dict):
            return {key: self._slice_nested(child, start, stop) for key, child in value.items()}
        return value[int(start) : int(stop)]

    def batch_insert(self, batch_payload: Any, *, payload_kind: str) -> None:
        if str(payload_kind) == "list":
            items = list(batch_payload)
            if not items:
                return
            if not self._vectorized:
                for item in items:
                    self.insert(item)
                return
            packed = _nested_stack(items)
        elif str(payload_kind) == "packed":
            packed = batch_payload
        else:
            raise ValueError(f"Unsupported payload_kind={payload_kind!r}")

        with self._lock:
            self._write_packed_nolock(packed)

    def sample(self, batch_size: int, *, hold_s: float = 0.0) -> float:
        with self._lock:
            if int(self._size) <= 0:
                return 0.0
            indices = self._rng.integers(0, int(self._size), size=int(batch_size))
            sample = {
                "observations": _nested_sample(self._dataset["observations"], indices),
                "next_observations": _nested_sample(self._dataset["next_observations"], indices),
                "actions": _nested_sample(self._dataset["actions"], indices),
                "rewards": _nested_sample(self._dataset["rewards"], indices),
                "masks": _nested_sample(self._dataset["masks"], indices),
                "dones": _nested_sample(self._dataset["dones"], indices),
            }
            if float(hold_s) > 0.0:
                time.sleep(float(hold_s))
        digest = float(
            np.asarray(sample["actions"], dtype=np.float32).mean()
            + np.asarray(sample["rewards"], dtype=np.float32).mean()
        )
        self._sample_calls += 1
        self._sample_digest += digest
        return digest


class SamplerLoad:
    def __init__(
        self,
        *,
        replay_store: SyntheticReplayStore,
        batch_size: int,
        sleep_s: float,
        hold_s: float,
        threads: int,
    ) -> None:
        self._replay_store = replay_store
        self._batch_size = int(batch_size)
        self._sleep_s = float(sleep_s)
        self._hold_s = float(hold_s)
        self._threads = int(threads)
        self._stop = threading.Event()
        self._thread_list: list[threading.Thread] = []
        self._iterations = 0
        self._iterations_lock = threading.Lock()

    @property
    def iterations(self) -> int:
        with self._iterations_lock:
            return int(self._iterations)

    def start(self) -> None:
        for index in range(int(self._threads)):
            thread = threading.Thread(
                target=self._loop,
                name=f"synthetic-sampler-{index}",
                daemon=True,
            )
            thread.start()
            self._thread_list.append(thread)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._replay_store.sample(int(self._batch_size), hold_s=float(self._hold_s))
            with self._iterations_lock:
                self._iterations += 1
            if self._sleep_s > 0.0:
                time.sleep(self._sleep_s)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._thread_list:
            thread.join(timeout=2.0)


class ThreadedReqRepServer:
    def __init__(self, *, port: int, callback: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._callback = callback
        self._stop = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._compress, self._decompress = make_compression_method("lz4")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        time.sleep(0.05)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop.is_set():
                    break
                raise
            request = self._decompress(message)
            response = self._callback(request)
            try:
                self._socket.send(self._compress(response))
            except zmq.ZMQError:
                if self._stop.is_set():
                    break
                raise

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._socket.close()
        self._context.term()


class PushClient:
    def __init__(self, *, host: str, port: int) -> None:
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 8)
        self._socket.connect(f"tcp://{host}:{port}")
        self._compress, _ = make_compression_method("lz4")

    def send(self, message: dict[str, Any]) -> None:
        self._socket.send(self._compress(message))

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class ThreadedPullServer:
    def __init__(self, *, port: int, callback: Callable[[dict[str, Any]], None]) -> None:
        self._callback = callback
        self._stop = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._socket.bind(f"tcp://*:{port}")
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        _, self._decompress = make_compression_method("lz4")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        time.sleep(0.05)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop.is_set():
                    break
                raise
            self._callback(self._decompress(message))

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._socket.close()
        self._context.term()


class QueueWorker:
    def __init__(
        self,
        *,
        replay_store: SyntheticReplayStore,
        payload_kind: str,
        capacity: int,
        on_committed: Callable[[int], None],
    ) -> None:
        self._replay_store = replay_store
        self._payload_kind = payload_kind
        self._on_committed = on_committed
        self._queue: queue.Queue[tuple[int, Any] | None] = queue.Queue(maxsize=int(capacity))
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def put(self, last_id: int, payload: Any) -> None:
        self._queue.put((int(last_id), payload))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            last_id, payload = item
            self._replay_store.batch_insert(payload, payload_kind=str(self._payload_kind))
            self._on_committed(int(last_id))

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)


class ScenarioServer:
    def __init__(self, *, scenario: Scenario, replay_store: SyntheticReplayStore, queue_capacity: int) -> None:
        self._scenario = scenario
        self._replay_store = replay_store
        self._accepted_id = -1
        self._committed_id = -1
        self._stats_count = 0
        self._lock = threading.Lock()
        self.control_port = _find_free_port()
        self.data_port: int | None = None
        self._reqrep: ThreadedReqRepServer | None = None
        self._pull: ThreadedPullServer | None = None
        self._worker: QueueWorker | None = None
        self._queue_capacity = int(queue_capacity)

        transport = str(scenario.transport_mode)
        if transport == "reqrep_sync":
            self._reqrep = ThreadedReqRepServer(port=self.control_port, callback=self._handle_control)
        elif transport == "reqrep_enqueue":
            self._worker = QueueWorker(
                replay_store=self._replay_store,
                payload_kind=str(scenario.payload_kind),
                capacity=int(queue_capacity),
                on_committed=self._set_committed_id,
            )
            self._reqrep = ThreadedReqRepServer(port=self.control_port, callback=self._handle_control)
        elif transport == "pipeline_direct":
            self.data_port = _find_free_port()
            self._reqrep = ThreadedReqRepServer(port=self.control_port, callback=self._handle_control)
            self._pull = ThreadedPullServer(port=int(self.data_port), callback=self._handle_data_direct)
        elif transport == "split_queue":
            self.data_port = _find_free_port()
            self._worker = QueueWorker(
                replay_store=self._replay_store,
                payload_kind=str(scenario.payload_kind),
                capacity=int(queue_capacity),
                on_committed=self._set_committed_id,
            )
            self._reqrep = ThreadedReqRepServer(port=self.control_port, callback=self._handle_control)
            self._pull = ThreadedPullServer(port=int(self.data_port), callback=self._handle_data_queue)
        else:
            raise ValueError(f"Unsupported transport_mode={transport!r}")

    def start(self) -> None:
        if self._worker is not None:
            self._worker.start()
        if self._reqrep is not None:
            self._reqrep.start()
        if self._pull is not None:
            self._pull.start()

    def stop(self) -> None:
        if self._pull is not None:
            self._pull.stop()
        if self._reqrep is not None:
            self._reqrep.stop()
        if self._worker is not None:
            self._worker.stop()

    def _set_committed_id(self, last_id: int) -> None:
        with self._lock:
            self._committed_id = max(int(self._committed_id), int(last_id))

    def _handle_control(self, request: dict[str, Any]) -> dict[str, Any]:
        req_type = str(request.get("type"))
        if req_type == "get_last_update_id":
            with self._lock:
                if str(self._scenario.update_id_semantics) == "accepted":
                    payload = int(self._accepted_id)
                else:
                    payload = int(self._committed_id)
            return {"success": True, "payload": payload}
        if req_type == "get_committed_update_id":
            with self._lock:
                payload = int(self._committed_id)
            return {"success": True, "payload": payload}
        if req_type == "send-stats":
            with self._lock:
                self._stats_count += 1
            return {"success": True, "payload": {"count": int(self._stats_count)}}
        if req_type == "datastore":
            last_id = int(request["last_id"])
            payload = request["payload"]
            with self._lock:
                self._accepted_id = max(int(self._accepted_id), int(last_id))

            if str(self._scenario.transport_mode) == "reqrep_sync":
                self._replay_store.batch_insert(payload, payload_kind=str(self._scenario.payload_kind))
                self._set_committed_id(int(last_id))
                return {"success": True}
            if str(self._scenario.transport_mode) == "reqrep_enqueue":
                assert self._worker is not None
                self._worker.put(int(last_id), payload)
                return {"success": True}
            return {"success": False, "message": f"unexpected reqrep datastore on {self._scenario.transport_mode}"}
        return {"success": False, "message": f"unknown request type={req_type!r}"}

    def _handle_data_direct(self, message: dict[str, Any]) -> None:
        last_id = int(message["last_id"])
        payload = message["payload"]
        with self._lock:
            self._accepted_id = max(int(self._accepted_id), int(last_id))
        self._replay_store.batch_insert(payload, payload_kind=str(message["payload_kind"]))
        self._set_committed_id(int(last_id))

    def _handle_data_queue(self, message: dict[str, Any]) -> None:
        last_id = int(message["last_id"])
        payload = message["payload"]
        with self._lock:
            self._accepted_id = max(int(self._accepted_id), int(last_id))
        assert self._worker is not None
        self._worker.put(int(last_id), payload)


class ScenarioClient:
    def __init__(self, *, control_port: int, data_port: int | None, timeout_ms: int) -> None:
        self._control = ReqRepClient("127.0.0.1", port=int(control_port), timeout_ms=int(timeout_ms), log_level=30)
        self._stats = ReqRepClient("127.0.0.1", port=int(control_port), timeout_ms=int(timeout_ms), log_level=30)
        self._data = PushClient(host="127.0.0.1", port=int(data_port)) if data_port is not None else None

    def control_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        return self._control.send_msg(message)

    def stats_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        return self._stats.send_msg(message)

    def send_data(self, message: dict[str, Any]) -> None:
        if self._data is None:
            raise RuntimeError("data transport is not configured")
        self._data.send(message)

    def close(self) -> None:
        if self._data is not None:
            self._data.close()
        _close_reqrep_client(self._control)
        _close_reqrep_client(self._stats)


def _make_packed_batch(
    *,
    rng: np.random.Generator,
    batch_size: int,
    image_size: int,
    proprio_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    def uint8_image() -> np.ndarray:
        return rng.integers(
            low=0,
            high=256,
            size=(int(batch_size), int(image_size), int(image_size), 3),
            dtype=np.uint8,
        )

    observations = {
        "agentview_rgb": uint8_image(),
        "wrist_rgb": uint8_image(),
        "proprio": rng.normal(size=(int(batch_size), int(proprio_dim))).astype(np.float32),
    }
    next_observations = {
        "agentview_rgb": uint8_image(),
        "wrist_rgb": uint8_image(),
        "proprio": rng.normal(size=(int(batch_size), int(proprio_dim))).astype(np.float32),
    }
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": rng.normal(size=(int(batch_size), int(action_dim))).astype(np.float32),
        "rewards": rng.normal(size=(int(batch_size),)).astype(np.float32),
        "masks": np.ones((int(batch_size),), dtype=np.float32),
        "dones": np.zeros((int(batch_size),), dtype=bool),
    }


def _packed_to_list(packed: dict[str, Any]) -> list[dict[str, Any]]:
    batch_size = int(np.asarray(packed["actions"]).shape[0])
    return [_nested_take(packed, index, copy_leaf=True) for index in range(batch_size)]


def _wait_for_committed_id(
    *,
    client: ScenarioClient,
    target_last_id: int,
    timeout_s: float,
    poll_s: float,
) -> tuple[bool, float]:
    start = time.perf_counter()
    deadline = start + float(timeout_s)
    while time.perf_counter() <= deadline:
        response = client.control_request({"type": "get_committed_update_id"})
        committed_id = int(response["payload"]) if response and response.get("success") else -1
        if committed_id >= int(target_last_id):
            return True, float(time.perf_counter() - start)
        time.sleep(float(poll_s))
    return False, float(time.perf_counter() - start)


def _run_iteration(
    *,
    scenario: Scenario,
    client: ScenarioClient,
    payload: Any,
    last_id: int,
    stats_offset_s: float,
    commit_timeout_s: float,
    control_poll_s: float,
) -> dict[str, Any]:
    update_result: dict[str, Any] = {}

    def _update() -> None:
        update_start = time.perf_counter()
        from_id_response = client.control_request({"type": "get_last_update_id"})
        from_id_ok = bool(from_id_response and from_id_response.get("success"))
        from_id = int(from_id_response["payload"]) if from_id_ok else -1
        if str(scenario.transport_mode).startswith("reqrep"):
            reply = client.control_request(
                {
                    "type": "datastore",
                    "payload_kind": str(scenario.payload_kind),
                    "payload": payload,
                    "last_id": int(last_id),
                }
            )
        else:
            client.send_data(
                {
                    "type": "datastore",
                    "payload_kind": str(scenario.payload_kind),
                    "payload": payload,
                    "last_id": int(last_id),
                }
            )
            reply = {"success": True}
        update_end = time.perf_counter()
        update_result.update(
            {
                "from_id_ok": bool(from_id_ok),
                "from_id": int(from_id),
                "reply_ok": bool(reply and reply.get("success", False)),
                "rpc_s": float(update_end - update_start),
                "start_t": float(update_start),
            }
        )

    thread = threading.Thread(target=_update, daemon=True)
    thread.start()
    time.sleep(float(stats_offset_s))

    stats_start = time.perf_counter()
    stats_reply = client.stats_request({"type": "send-stats", "payload": {"iteration": int(last_id)}})
    stats_latency = float(time.perf_counter() - stats_start)

    thread.join()
    committed_ok, committed_wait_s = _wait_for_committed_id(
        client=client,
        target_last_id=int(last_id),
        timeout_s=float(commit_timeout_s),
        poll_s=float(control_poll_s),
    )
    total_commit_s = float(update_result["rpc_s"] + committed_wait_s)

    return {
        "reply_ok": bool(update_result["reply_ok"]),
        "from_id_ok": bool(update_result["from_id_ok"]),
        "from_id": int(update_result["from_id"]),
        "update_rpc_s": float(update_result["rpc_s"]),
        "stats_s": float(stats_latency),
        "stats_ok": bool(stats_reply and stats_reply.get("success", False)),
        "commit_ok": bool(committed_ok),
        "commit_wait_after_rpc_s": float(committed_wait_s),
        "update_to_commit_s": float(total_commit_s),
    }


def _make_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="baseline_sync_reqrep_list_legacy",
            description="Current shape: req/rep datastore + list[dict] payload + per-transition insert",
            payload_kind="list",
            replay_mode="legacy",
            transport_mode="reqrep_sync",
            update_id_semantics="committed",
        ),
        Scenario(
            name="sync_reqrep_list_vectorized",
            description="Optimization 1: req/rep datastore + list[dict] payload + vectorized batch insert",
            payload_kind="list",
            replay_mode="vectorized",
            transport_mode="reqrep_sync",
            update_id_semantics="committed",
        ),
        Scenario(
            name="sync_reqrep_packed_vectorized",
            description="Optimization 1+3: req/rep datastore + packed ndarray payload + vectorized batch insert",
            payload_kind="packed",
            replay_mode="vectorized",
            transport_mode="reqrep_sync",
            update_id_semantics="committed",
        ),
        Scenario(
            name="plan_b_reqrep_enqueue_ack",
            description="Plan B: req/rep callback only enqueues and returns ack; worker thread inserts later",
            payload_kind="packed",
            replay_mode="vectorized",
            transport_mode="reqrep_enqueue",
            update_id_semantics="accepted",
        ),
        Scenario(
            name="plan_a_pipeline_direct",
            description="Plan A: split data to push/pull pipeline, control stays req/rep, data thread inserts directly",
            payload_kind="packed",
            replay_mode="vectorized",
            transport_mode="pipeline_direct",
            update_id_semantics="committed",
        ),
        Scenario(
            name="plan_c_split_control_data_queue",
            description="Plan C: control req/rep + data pipeline + worker queue + explicit accepted/committed separation",
            payload_kind="packed",
            replay_mode="vectorized",
            transport_mode="split_queue",
            update_id_semantics="accepted",
        ),
    ]


def _run_scenario(
    *,
    scenario: Scenario,
    cfg: BenchmarkConfig,
    packed_batches: list[dict[str, Any]],
    sample_transition: dict[str, Any],
) -> dict[str, Any]:
    replay_store = SyntheticReplayStore(
        example_transition=sample_transition,
        capacity=int(cfg.replay_capacity),
        vectorized=bool(str(scenario.replay_mode) == "vectorized"),
    )
    sampler = SamplerLoad(
        replay_store=replay_store,
        batch_size=int(cfg.sampler_batch_size),
        sleep_s=float(cfg.sampler_sleep_ms) / 1000.0,
        hold_s=float(cfg.sampler_hold_ms) / 1000.0,
        threads=int(cfg.sampler_threads),
    )
    server = ScenarioServer(
        scenario=scenario,
        replay_store=replay_store,
        queue_capacity=int(cfg.queue_capacity),
    )
    client = ScenarioClient(
        control_port=int(server.control_port),
        data_port=server.data_port,
        timeout_ms=int(cfg.timeout_ms),
    )

    total_iterations = int(cfg.warmup + cfg.iterations)
    update_rpc_samples: list[float] = []
    stats_samples: list[float] = []
    commit_samples: list[float] = []
    iteration_records: list[dict[str, Any]] = []

    payload_example = (
        packed_batches[0]
        if str(scenario.payload_kind) == "packed"
        else _packed_to_list(packed_batches[0])
    )
    update_message_bytes = (
        _compress_bytes(
            {
                "type": "datastore",
                "payload_kind": str(scenario.payload_kind),
                "payload": payload_example,
                "last_id": int(cfg.batch_size - 1),
            }
        )
        / 1024.0
        / 1024.0
    )

    server.start()
    sampler.start()
    try:
        for iteration in range(total_iterations):
            packed = packed_batches[int(iteration)]
            payload = packed if str(scenario.payload_kind) == "packed" else _packed_to_list(packed)
            last_id = int((iteration + 1) * cfg.batch_size - 1)
            result = _run_iteration(
                scenario=scenario,
                client=client,
                payload=payload,
                last_id=int(last_id),
                stats_offset_s=float(cfg.stats_offset_ms) / 1000.0,
                commit_timeout_s=float(cfg.commit_timeout_s),
                control_poll_s=float(cfg.control_poll_ms) / 1000.0,
            )
            if iteration >= int(cfg.warmup):
                update_rpc_samples.append(float(result["update_rpc_s"]))
                stats_samples.append(float(result["stats_s"]))
                commit_samples.append(float(result["update_to_commit_s"]))
                iteration_records.append(result)
    finally:
        client.close()
        sampler.stop()
        server.stop()

    update_summary = _summarize(update_rpc_samples)
    stats_summary = _summarize(stats_samples)
    commit_summary = _summarize(commit_samples)
    update_under_800 = float(np.mean(np.asarray(update_rpc_samples) <= 0.8)) if update_rpc_samples else 0.0
    stats_under_800 = float(np.mean(np.asarray(stats_samples) <= 0.8)) if stats_samples else 0.0
    commit_under_800 = float(np.mean(np.asarray(commit_samples) <= 0.8)) if commit_samples else 0.0
    return {
        "name": str(scenario.name),
        "description": str(scenario.description),
        "payload_kind": str(scenario.payload_kind),
        "replay_mode": str(scenario.replay_mode),
        "transport_mode": str(scenario.transport_mode),
        "update_id_semantics": str(scenario.update_id_semantics),
        "update_message_mb": float(update_message_bytes),
        "update_rpc": update_summary,
        "stats_during_update": stats_summary,
        "update_to_commit": commit_summary,
        "update_under_800ms_ratio": float(update_under_800),
        "stats_under_800ms_ratio": float(stats_under_800),
        "update_to_commit_under_800ms_ratio": float(commit_under_800),
        "all_updates_ok": bool(all(record["reply_ok"] and record["from_id_ok"] for record in iteration_records)),
        "all_stats_ok": bool(all(record["stats_ok"] for record in iteration_records)),
        "all_commits_ok": bool(all(record["commit_ok"] for record in iteration_records)),
        "sampler_iterations": int(sampler.iterations),
        "replay_insert_count": int(replay_store.insert_count),
        "replay_sample_calls": int(replay_store.sample_calls),
        "iteration_count": int(len(iteration_records)),
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("| scenario | update mean s | stats mean s | commit mean s | <=800ms update | <=800ms stats | <=800ms commit | payload MB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            "| {name} | {update:.4f} | {stats:.4f} | {commit:.4f} | {u800:.0%} | {s800:.0%} | {c800:.0%} | {payload:.2f} |".format(
                name=result["name"],
                update=result["update_rpc"]["mean_s"],
                stats=result["stats_during_update"]["mean_s"],
                commit=result["update_to_commit"]["mean_s"],
                u800=result["update_under_800ms_ratio"],
                s800=result["stats_under_800ms_ratio"],
                c800=result["update_to_commit_under_800ms_ratio"],
                payload=result["update_message_mb"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic trainer datastore benchmark")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--replay-capacity", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--proprio-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--stats-offset-ms", type=float, default=10.0)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--sampler-batch-size", type=int, default=128)
    parser.add_argument("--sampler-sleep-ms", type=float, default=1.0)
    parser.add_argument("--sampler-hold-ms", type=float, default=0.0)
    parser.add_argument("--sampler-threads", type=int, default=1)
    parser.add_argument("--control-poll-ms", type=float, default=5.0)
    parser.add_argument("--commit-timeout-s", type=float, default=10.0)
    parser.add_argument("--queue-capacity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        iterations=int(args.iterations),
        warmup=int(args.warmup),
        batch_size=int(args.batch_size),
        replay_capacity=int(args.replay_capacity),
        image_size=int(args.image_size),
        proprio_dim=int(args.proprio_dim),
        action_dim=int(args.action_dim),
        stats_offset_ms=float(args.stats_offset_ms),
        timeout_ms=int(args.timeout_ms),
        sampler_batch_size=int(args.sampler_batch_size),
        sampler_sleep_ms=float(args.sampler_sleep_ms),
        sampler_hold_ms=float(args.sampler_hold_ms),
        sampler_threads=int(args.sampler_threads),
        control_poll_ms=float(args.control_poll_ms),
        commit_timeout_s=float(args.commit_timeout_s),
        queue_capacity=int(args.queue_capacity),
        random_seed=int(args.seed),
    )

    rng = np.random.default_rng(int(cfg.random_seed))
    total_iterations = int(cfg.warmup + cfg.iterations)
    packed_batches = [
        _make_packed_batch(
            rng=rng,
            batch_size=int(cfg.batch_size),
            image_size=int(cfg.image_size),
            proprio_dim=int(cfg.proprio_dim),
            action_dim=int(cfg.action_dim),
        )
        for _ in range(total_iterations)
    ]
    sample_transition = _nested_take(packed_batches[0], 0, copy_leaf=True)

    scenarios = _make_scenarios()
    if args.scenario:
        selected = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario.name in selected]
        if not scenarios:
            raise ValueError(f"No scenarios matched {sorted(selected)!r}")

    results = [
        _run_scenario(
            scenario=scenario,
            cfg=cfg,
            packed_batches=packed_batches,
            sample_transition=sample_transition,
        )
        for scenario in scenarios
    ]

    _print_summary(results)

    payload = {
        "config": asdict(cfg),
        "results": results,
    }
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote benchmark json to {output_path}")


if __name__ == "__main__":
    main()
