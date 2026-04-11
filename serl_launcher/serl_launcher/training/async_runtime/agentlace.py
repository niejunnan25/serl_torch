"""Asynchronous learner and replay-prefetch helpers for training runtimes."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import socket
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.runtime_agent import ResidualAgentRuntime
from serl_launcher.residual.runtime_agent import create_residual_agent_runtime
from serl_launcher.training.checkpoint import AsyncCheckpointWriter
from serl_launcher.training.checkpoint import CheckpointTask
from serl_launcher.training.checkpoint import write_checkpoint_payload_profiled
from serl_launcher.training.profiling import _RuntimeProfiler
from serl_launcher.training.profiling import _emit_profiling_snapshot
from serl_launcher.training.replay_batch import _PreparedBatch
from serl_launcher.training.replay_batch import _consume_prepared_replay_batch
from serl_launcher.training.replay_batch import _prepare_replay_batch

if TYPE_CHECKING:
    from serl_launcher.data.replay_buffer import ReplayBuffer


def _is_transient_replay_unavailable(exc: BaseException) -> bool:
    message = str(exc).strip().lower()
    if not message:
        return False
    return "no eligible chunk starts" in message


def _make_agentlace_trainer_config(*, port_number: int, broadcast_port: int):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=int(port_number),
        broadcast_port=int(broadcast_port),
        request_types=["send-stats", "save-checkpoint", "get-status", "sync-now"],
    )


def _safe_status_emit(status_queue: Any, item: Dict[str, Any]) -> None:
    if status_queue is None:
        return
    _put_latest_queue_item(status_queue, item)


def _resolve_poll_interval(
    poll_interval_sec: Optional[float],
    *,
    default_sec: float,
) -> float:
    if poll_interval_sec is None:
        return float(max(1e-4, default_sec))
    return float(max(1e-4, poll_interval_sec))


def _default_agentlace_poll_interval_sec(idle_sleep_sec: float) -> float:
    return float(max(0.05, min(0.2, max(1e-4, float(idle_sleep_sec) * 25.0))))


def _resolve_timeout_deadline(timeout_sec: Optional[float]) -> Optional[float]:
    if timeout_sec is None:
        return None
    return time.monotonic() + max(0.0, float(timeout_sec))


def _replay_sampleable_size(buffer: Any) -> int:
    return int(len(buffer))


def _replay_capacity(buffer: Any) -> Optional[int]:
    if hasattr(buffer, "capacity"):
        return int(getattr(buffer, "capacity"))
    if hasattr(buffer, "_capacity"):
        return int(getattr(buffer, "_capacity"))
    return None


def _build_status_payload(
    *,
    update_steps: int,
    last_update_info: Dict[str, Any],
    replay_buffer: Any,
    replay_prefetch_queue_size: Optional[int] = None,
    online_prefill_stats: Optional[Dict[str, Any]] = None,
    sync_payload: Optional[Dict[str, Any]] = None,
    message_type: str = "status",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": str(message_type),
        "update_steps": int(update_steps),
        "last_update_info": dict(last_update_info),
        "replay_num_steps": int(
            getattr(replay_buffer, "num_steps", len(replay_buffer))
        ),
        "replay_sampleable_size": int(_replay_sampleable_size(replay_buffer)),
    }
    if replay_prefetch_queue_size is not None:
        payload["replay_prefetch_queue_size"] = int(replay_prefetch_queue_size)
    if online_prefill_stats is not None:
        payload["online_prefill_stats"] = dict(online_prefill_stats)
    if sync_payload is not None:
        payload["sync_payload"] = sync_payload
    return payload


def _wait_for_tcp_server(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.monotonic() + max(1e-3, float(timeout_sec))
    last_exc: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(
                (str(host), int(port)),
                timeout=min(1.0, max(0.1, float(timeout_sec))),
            ):
                return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.1)
    raise RuntimeError(
        f"Timed out waiting for agentlace trainer server at {host}:{int(port)}"
    ) from last_exc


def _put_latest_queue_item(target_queue: Any, item: Any) -> None:
    while True:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                time.sleep(1e-4)


class _ReplayProgressProxy:
    """Actor-side replay proxy used when authoritative replay lives elsewhere."""

    def __init__(
        self,
        *,
        initial_num_steps: int = 0,
        initial_sampleable_size: Optional[int] = None,
        capacity: Optional[int] = None,
        on_insert: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._capacity = None if capacity is None else max(1, int(capacity))
        self.num_steps = int(initial_num_steps)
        if self._capacity is not None:
            self.num_steps = min(self.num_steps, self._capacity)
        self._size = int(
            self.num_steps
            if initial_sampleable_size is None
            else max(0, int(initial_sampleable_size))
        )
        self._on_insert = on_insert
        self._stats_synced = False

    def insert(self, data_dict: Dict[str, Any]) -> None:
        if self._on_insert is not None:
            self._on_insert(data_dict)
        next_steps = int(self.num_steps + 1)
        if self._capacity is not None:
            next_steps = min(next_steps, self._capacity)
        self.num_steps = next_steps
        optimistic_size = int(self._size + 1)
        if self._capacity is not None:
            optimistic_size = min(optimistic_size, self._capacity)
        if not self._stats_synced:
            optimistic_size = int(self.num_steps)
        self._size = optimistic_size

    def sync_from_status(
        self,
        *,
        num_steps: Optional[int] = None,
        sampleable_size: Optional[int] = None,
    ) -> None:
        if num_steps is not None:
            next_steps = int(num_steps)
            if self._capacity is not None:
                next_steps = min(next_steps, self._capacity)
            self.num_steps = next_steps
        if sampleable_size is not None:
            next_size = max(0, int(sampleable_size))
            if self._capacity is not None:
                next_size = min(next_size, self._capacity)
            self._size = next_size
        self._stats_synced = True

    def __len__(self) -> int:
        return int(self._size)


class _BufferDataStoreAdapter:
    """Minimal agentlace-compatible wrapper around a replay-like buffer."""

    def __init__(self, buffer: Any) -> None:
        self.buffer = buffer
        self._latest_id = int(getattr(buffer, "num_steps", len(buffer)))
        # Agentlace serves inserts from RPC worker threads while the learner loop
        # samples from the same replay object. Guard every access through one
        # shared adapter so chunk replay never sees partially updated internals.
        self._lock = threading.RLock()

    def insert(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self.buffer.insert(data)
            self._latest_id += 1

    def batch_insert(self, batch_data: Sequence[Dict[str, Any]]) -> None:
        # TrainerServer writes batched transitions via DataStore.batch_insert.
        for item in batch_data:
            self.insert(item)

    def sample(self, *args, **kwargs):
        with self._lock:
            return self.buffer.sample(*args, **kwargs)

    def latest_data_id(self) -> int:
        with self._lock:
            return int(self._latest_id)

    def get_latest_data(self, from_id: int):
        del from_id
        raise NotImplementedError

    @property
    def num_steps(self) -> int:
        with self._lock:
            return int(getattr(self.buffer, "num_steps", len(self.buffer)))

    def __len__(self) -> int:
        with self._lock:
            length_fn = getattr(self.buffer, "__len__", None)
            if callable(length_fn):
                return int(length_fn())
            return int(getattr(self.buffer, "num_steps", self._latest_id))

    @property
    def sampleable_size(self) -> int:
        with self._lock:
            return int(_replay_sampleable_size(self.buffer))


def _async_process_worker(
    *,
    cfg_dict: Dict[str, Any],
    sample_obs: Dict[str, np.ndarray],
    action_dim: int,
    critic_action_dim: int,
    image_keys: Tuple[str, ...],
    action_transform: Optional[Dict[str, Any]],
    learner_device: Optional[str],
    update_frequency: int,
    idle_sleep_sec: float,
    initial_payload: Dict[str, Any],
    batch_queue: Any,
    status_queue: Any,
    command_queue: Any,
) -> None:
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(cfg_dict)
        algorithm = create_residual_agent_runtime(cfg)
        learner_agent = algorithm.create_learner_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=int(action_dim),
            image_keys=tuple(image_keys),
            critic_action_dim=int(critic_action_dim),
            action_transform=action_transform,
            device=learner_device,
        )
        apply_checkpoint_payload_to_agent(
            learner_agent, initial_payload, load_optimizers=True
        )

        update_steps = 0
        stop_requested = False
        while not stop_requested:
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break

                command_type = str(command.get("type", ""))
                if command_type == "stop":
                    sync_payload = snapshot_agent_checkpoint_payload(
                        learner_agent,
                        step=int(update_steps),
                    )
                    _put_latest_queue_item(
                        status_queue,
                        {
                            "type": "status",
                            "update_steps": int(update_steps),
                            "last_update_info": {},
                            "sync_payload": sync_payload,
                        },
                    )
                    stop_requested = True
                    break
                if command_type == "sync_now":
                    sync_payload = snapshot_agent_checkpoint_payload(
                        learner_agent,
                        step=int(update_steps),
                    )
                    _put_latest_queue_item(
                        status_queue,
                        {
                            "type": "status",
                            "update_steps": int(update_steps),
                            "last_update_info": {},
                            "sync_payload": sync_payload,
                        },
                    )
                    continue
                if command_type == "save_checkpoint":
                    checkpoint_payload = snapshot_agent_checkpoint_payload(
                        learner_agent,
                        step=int(command["step"]),
                    )
                    write_checkpoint_payload_profiled(
                        profiler=None,
                        checkpoint_dir=str(command["checkpoint_dir"]),
                        payload=checkpoint_payload,
                        step=int(command["step"]),
                        keep=int(command["keep"]),
                    )
                    continue
            if stop_requested:
                break

            try:
                sampled = batch_queue.get(timeout=idle_sleep_sec)
            except queue.Empty:
                continue

            if sampled is None:
                break

            batch, online_bs, offline_bs = sampled
            learner_agent, info = algorithm.update_high_utd(
                learner_agent,
                batch,
                utd_ratio=int(cfg.sac.utd_ratio),
            )
            info["online_batch_size"] = int(online_bs)
            info["offline_batch_size"] = int(offline_bs)
            info["offline_fraction"] = float(
                offline_bs / max(1, online_bs + offline_bs)
            )
            update_steps += 1
            status: Dict[str, Any] = {
                "type": "status",
                "update_steps": int(update_steps),
                "last_update_info": dict(info),
            }
            if update_steps % max(1, int(update_frequency)) == 0:
                status["sync_payload"] = snapshot_agent_checkpoint_payload(
                    learner_agent,
                    step=int(update_steps),
                )
            _put_latest_queue_item(status_queue, status)
    except BaseException as exc:  # noqa: BLE001
        _put_latest_queue_item(
            status_queue,
            {
                "type": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def run_agentlace_learner_service(
    *,
    cfg_dict: Dict[str, Any],
    sample_obs: Dict[str, np.ndarray],
    action_dim: int,
    critic_action_dim: int,
    image_keys: Tuple[str, ...],
    action_transform: Optional[Dict[str, Any]],
    learner_device: Optional[str],
    update_frequency: int,
    idle_sleep_sec: float,
    training_starts: int,
    initial_payload: Dict[str, Any],
    replay_buffer: Any,
    offline_buffer: Optional[Any],
    batch_size: int,
    offline_ratio: float,
    symmetric_replay: bool,
    algorithm: Optional[ResidualAgentRuntime] = None,
    host: str,
    port_number: int,
    broadcast_port: int,
    replay_prefetch_enabled: Optional[bool] = None,
    replay_prefetch_queue_size: Optional[int] = None,
    replay_prefetch_pin_memory: Optional[bool] = None,
    replay_prefetch_to_device: Optional[bool] = None,
    online_prefill_stats: Optional[Dict[str, Any]] = None,
    profiler: Optional[_RuntimeProfiler] = None,
    profiling_log_period_steps: int = 500,
    profiling_logger: Any = None,
    profiling_tb_writer: Any = None,
    profiling_python_logger: Optional[logging.Logger] = None,
    status_queue: Any = None,
    command_queue: Any = None,
    stats_request_callback: Optional[
        Callable[[Dict[str, Any], int, Dict[str, Any], Any, Optional[Any]], None]
    ] = None,
) -> None:
    from omegaconf import OmegaConf
    from agentlace.trainer import TrainerServer

    cfg = OmegaConf.create(cfg_dict)
    if algorithm is None:
        algorithm = create_residual_agent_runtime(cfg)
    replay_prefetch_cfg = cfg.training.get("replay_prefetch", None)
    if replay_prefetch_enabled is None:
        replay_prefetch_enabled = (
            bool(replay_prefetch_cfg.get("enabled", False))
            if replay_prefetch_cfg is not None
            else False
        )
    if replay_prefetch_queue_size is None:
        replay_prefetch_queue_size = (
            int(replay_prefetch_cfg.get("queue_size", 2))
            if replay_prefetch_cfg is not None
            else 2
        )
    if replay_prefetch_pin_memory is None:
        replay_prefetch_pin_memory = (
            bool(replay_prefetch_cfg.get("pin_memory", False))
            if replay_prefetch_cfg is not None
            else False
        )
    if replay_prefetch_to_device is None:
        replay_prefetch_to_device = (
            bool(replay_prefetch_cfg.get("to_device", False))
            if replay_prefetch_cfg is not None
            else False
        )
    learner_agent = algorithm.create_learner_agent(
        cfg,
        sample_obs=sample_obs,
        action_dim=int(action_dim),
        image_keys=tuple(image_keys),
        critic_action_dim=int(critic_action_dim),
        action_transform=action_transform,
        device=learner_device,
    )
    if initial_payload is not None:
        apply_checkpoint_payload_to_agent(
            learner_agent,
            initial_payload,
            load_optimizers=True,
        )

    update_steps = (
        int(initial_payload.get("step", 0))
        if initial_payload is not None
        else int(learner_agent.state.step)
    )
    profiling_log_period_steps = max(1, int(profiling_log_period_steps))
    last_update_info: Dict[str, Any] = {}
    server = None
    prefetcher: Optional[_MixedBatchPrefetcher] = None
    replay_store = _BufferDataStoreAdapter(replay_buffer)
    logger = profiling_python_logger or logging.getLogger(
        "libero_agentlace_learner_service"
    )
    latest_progress: Dict[str, int] = {
        "train_env_step": 0,
        "decision_step": 0,
        "train_episode_id": 0,
    }
    online_prefill_status_payload = (
        dict(online_prefill_stats) if online_prefill_stats is not None else None
    )
    profiling_last_flush_env_step = -1

    def _current_prefetch_queue_size() -> int:
        if prefetcher is None:
            return 0
        return prefetcher.get_queue_size()

    def _update_latest_progress(payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        candidate_payloads = []
        timer_payload = payload.get("timer", None)
        if isinstance(timer_payload, dict):
            candidate_payloads.append(timer_payload)
        train_episode_payload = payload.get("train_episode", None)
        if isinstance(train_episode_payload, dict):
            candidate_payloads.append(train_episode_payload)
        candidate_payloads.append(payload)
        for item in candidate_payloads:
            if not isinstance(item, dict):
                continue
            train_env_step = item.get("train_env_step", None)
            if train_env_step is not None:
                latest_progress["train_env_step"] = max(
                    int(latest_progress.get("train_env_step", 0)),
                    int(train_env_step),
                )
            decision_step = item.get("decision_step", None)
            if decision_step is not None:
                latest_progress["decision_step"] = max(
                    int(latest_progress.get("decision_step", 0)),
                    int(decision_step),
                )
            train_episode_id = item.get("train_episode_id", None)
            if train_episode_id is not None:
                latest_progress["train_episode_id"] = max(
                    int(latest_progress.get("train_episode_id", 0)),
                    int(train_episode_id),
                )

    def _maybe_emit_profiling(force: bool = False) -> None:
        nonlocal profiling_last_flush_env_step
        if profiler is None or (not profiler.enabled) or (not profiler.has_data()):
            return
        current_train_env_step = int(latest_progress.get("train_env_step", 0))
        if (not force) and profiling_last_flush_env_step >= 0:
            if (current_train_env_step - profiling_last_flush_env_step) < int(
                profiling_log_period_steps
            ):
                return
        payload = _emit_profiling_snapshot(
            profiler,
            profile_logger=profiling_logger,
            tb_writer=profiling_tb_writer,
            logger=logger,
            train_env_step=current_train_env_step,
            decision_step=int(latest_progress.get("decision_step", 0)),
            train_episode_id=int(latest_progress.get("train_episode_id", 0)),
            learner_update_steps=int(update_steps),
            replay_prefetch_queue_size=_current_prefetch_queue_size(),
        )
        if payload is not None:
            profiling_last_flush_env_step = int(current_train_env_step)

    def _sample_batch() -> Optional[Tuple[Dict[str, Any], int, int]]:
        if _replay_progress_size(replay_store) < int(training_starts):
            return None
        try:
            return _sample_mixed_batch(
                replay_store,
                offline_buffer,
                batch_size=int(batch_size),
                offline_ratio=float(offline_ratio),
                symmetric_replay=bool(symmetric_replay),
            )
        except (RuntimeError, ValueError) as exc:
            if _is_transient_replay_unavailable(exc):
                return None
            raise

    def _handle_stats_request(payload: Dict[str, Any]) -> None:
        _update_latest_progress(payload)
        if stats_request_callback is not None:
            stats_request_callback(
                dict(payload),
                int(update_steps),
                dict(last_update_info),
                replay_store,
                offline_buffer,
            )
            return
        _safe_status_emit(
            status_queue,
            {
                "type": "actor_stats",
                "payload": dict(payload),
            },
        )

    try:
        server = None

        def _stats_callback(request_type: str, payload: dict) -> dict:
            if request_type == "send-stats":
                _handle_stats_request(dict(payload))
                return {}
            if request_type == "save-checkpoint":
                checkpoint_payload = snapshot_agent_checkpoint_payload(
                    learner_agent,
                    step=int(payload["step"]),
                )
                write_checkpoint_payload_profiled(
                    profiler=profiler,
                    checkpoint_dir=str(payload["checkpoint_dir"]),
                    payload=checkpoint_payload,
                    step=int(payload["step"]),
                    keep=int(payload["keep"]),
                )
                return {}
            if request_type == "get-status":
                return _build_status_payload(
                    update_steps=int(update_steps),
                    last_update_info=dict(last_update_info),
                    replay_buffer=replay_store,
                    replay_prefetch_queue_size=_current_prefetch_queue_size(),
                    online_prefill_stats=online_prefill_status_payload,
                )
            if request_type == "sync-now":
                if server is None:
                    raise RuntimeError("Agentlace trainer server is not ready")
                server.publish_network(
                    snapshot_agent_checkpoint_payload(
                        learner_agent,
                        step=int(update_steps),
                    )
                )
                return _build_status_payload(
                    update_steps=int(update_steps),
                    last_update_info=dict(last_update_info),
                    replay_buffer=replay_store,
                    replay_prefetch_queue_size=_current_prefetch_queue_size(),
                    online_prefill_stats=online_prefill_status_payload,
                )
            raise ValueError(f"Invalid request type: {request_type}")

        server = TrainerServer(
            _make_agentlace_trainer_config(
                port_number=int(port_number),
                broadcast_port=int(broadcast_port),
            ),
            request_callback=_stats_callback,
        )
        server.register_data_store(
            "actor_env",
            replay_store,
        )
        server.start(threaded=True)
        _safe_status_emit(
            status_queue,
            _build_status_payload(
                update_steps=int(update_steps),
                last_update_info=dict(last_update_info),
                replay_buffer=replay_store,
                replay_prefetch_queue_size=_current_prefetch_queue_size(),
                online_prefill_stats=online_prefill_status_payload,
                message_type="server_ready",
            ),
        )
        server.publish_network(
            snapshot_agent_checkpoint_payload(learner_agent, step=update_steps)
        )
        if replay_prefetch_enabled:
            prefetcher = _MixedBatchPrefetcher(
                sample_fn=_sample_batch,
                queue_size=max(1, int(replay_prefetch_queue_size)),
                idle_sleep_sec=float(idle_sleep_sec),
                device=learner_agent.device,
                pin_memory=bool(replay_prefetch_pin_memory),
                to_device=bool(replay_prefetch_to_device),
                profiler=profiler,
            )
            prefetcher.start()

        while int(getattr(replay_store, "num_steps", len(replay_store))) < int(
            training_starts
        ):
            if command_queue is not None:
                try:
                    command = command_queue.get_nowait()
                    if str(command.get("type", "")) == "stop":
                        return
                except queue.Empty:
                    pass
            time.sleep(idle_sleep_sec)

        stop_requested = False
        while not stop_requested:
            if command_queue is not None:
                while True:
                    try:
                        command = command_queue.get_nowait()
                    except queue.Empty:
                        break

                    command_type = str(command.get("type", ""))
                    if command_type == "stop":
                        stop_requested = True
                        break
                    if command_type == "sync_now":
                        server.publish_network(
                            snapshot_agent_checkpoint_payload(
                                learner_agent,
                                step=int(update_steps),
                            )
                        )
                        _safe_status_emit(
                            status_queue,
                            _build_status_payload(
                                update_steps=int(update_steps),
                                last_update_info=dict(last_update_info),
                                replay_buffer=replay_store,
                                replay_prefetch_queue_size=_current_prefetch_queue_size(),
                                online_prefill_stats=online_prefill_status_payload,
                            ),
                        )
                        continue
                    if command_type == "save_checkpoint":
                        checkpoint_payload = snapshot_agent_checkpoint_payload(
                            learner_agent,
                            step=int(command["step"]),
                        )
                        write_checkpoint_payload_profiled(
                            profiler=profiler,
                            checkpoint_dir=str(command["checkpoint_dir"]),
                            payload=checkpoint_payload,
                            step=int(command["step"]),
                            keep=int(command["keep"]),
                        )
                        continue
            if stop_requested:
                break

            if int(getattr(replay_store, "num_steps", len(replay_store))) < int(
                training_starts
            ):
                time.sleep(idle_sleep_sec)
                continue

            if prefetcher is not None:
                sampled = prefetcher.get(
                    timeout=max(0.01, float(idle_sleep_sec)),
                    allow_empty=True,
                )
            else:
                sample_start = time.perf_counter()
                sampled = _sample_batch()
                if sampled is not None and profiler is not None:
                    profiler.record_duration(
                        "replay_sample",
                        (time.perf_counter() - sample_start) * 1000.0,
                    )
            if sampled is None:
                time.sleep(idle_sleep_sec)
                continue
            if prefetcher is not None:
                batch, online_bs, offline_bs = sampled
            else:
                prepare_start = time.perf_counter()
                prepared = _prepare_replay_batch(
                    sampled,
                    device=learner_agent.device,
                    pin_memory=bool(replay_prefetch_pin_memory),
                    to_device=bool(replay_prefetch_to_device),
                    profiler=profiler,
                    cuda_stream=None,
                )
                batch, online_bs, offline_bs = _consume_prepared_replay_batch(
                    prepared,
                    device=learner_agent.device,
                    profiler=profiler,
                )
                if profiler is not None:
                    profiler.record_duration(
                        "replay_prepare",
                        (time.perf_counter() - prepare_start) * 1000.0,
                    )

            update_start = time.perf_counter()
            learner_agent, info = algorithm.update_high_utd(
                learner_agent,
                batch,
                utd_ratio=int(cfg.sac.utd_ratio),
            )
            if profiler is not None:
                profiler.record_duration(
                    "agent_update_high_utd",
                    (time.perf_counter() - update_start) * 1000.0,
                )
            info["online_batch_size"] = int(online_bs)
            info["offline_batch_size"] = int(offline_bs)
            info["offline_fraction"] = float(
                offline_bs / max(1, online_bs + offline_bs)
            )
            last_update_info = dict(info)
            update_steps += 1
            _safe_status_emit(
                status_queue,
                _build_status_payload(
                    update_steps=int(update_steps),
                    last_update_info=dict(last_update_info),
                    replay_buffer=replay_store,
                    replay_prefetch_queue_size=_current_prefetch_queue_size(),
                    online_prefill_stats=online_prefill_status_payload,
                ),
            )
            if update_steps % max(1, int(update_frequency)) == 0:
                server.publish_network(
                    snapshot_agent_checkpoint_payload(
                        learner_agent,
                        step=int(update_steps),
                    )
                )
            _maybe_emit_profiling(force=False)

            time.sleep(max(0.0, float(idle_sleep_sec) * 0.25))
    finally:
        _maybe_emit_profiling(force=True)
        if prefetcher is not None:
            prefetcher.stop()
        if server is not None:
            server.stop()


def _agentlace_async_worker(
    *,
    cfg_dict: Dict[str, Any],
    sample_obs: Dict[str, np.ndarray],
    action_dim: int,
    critic_action_dim: int,
    image_keys: Tuple[str, ...],
    action_transform: Optional[Dict[str, Any]],
    learner_device: Optional[str],
    update_frequency: int,
    idle_sleep_sec: float,
    training_starts: int,
    initial_payload: Dict[str, Any],
    replay_buffer: Any,
    offline_buffer: Optional[Any],
    batch_size: int,
    offline_ratio: float,
    symmetric_replay: bool,
    host: str,
    port_number: int,
    broadcast_port: int,
    status_queue: Any = None,
    command_queue: Any = None,
) -> None:
    try:
        run_agentlace_learner_service(
            cfg_dict=cfg_dict,
            sample_obs=sample_obs,
            action_dim=action_dim,
            critic_action_dim=critic_action_dim,
            image_keys=image_keys,
            action_transform=action_transform,
            learner_device=learner_device,
            update_frequency=update_frequency,
            idle_sleep_sec=idle_sleep_sec,
            training_starts=training_starts,
            initial_payload=initial_payload,
            replay_buffer=replay_buffer,
            offline_buffer=offline_buffer,
            batch_size=batch_size,
            offline_ratio=offline_ratio,
            symmetric_replay=symmetric_replay,
            host=host,
            port_number=port_number,
            broadcast_port=broadcast_port,
            status_queue=status_queue,
            command_queue=command_queue,
        )
    except BaseException as exc:  # noqa: BLE001
        _safe_status_emit(
            status_queue,
            {
                "type": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


class _MixedBatchPrefetcher:
    """Background sampler that prefetches replay batches for the learner."""

    def __init__(
        self,
        *,
        sample_fn: Callable[[], Optional[Tuple[Dict[str, Any], int, int]]],
        queue_size: int,
        idle_sleep_sec: float,
        device: Optional[torch.device],
        pin_memory: bool,
        to_device: bool,
        profiler: Optional[_RuntimeProfiler] = None,
    ) -> None:
        self.sample_fn = sample_fn
        self.queue_size = max(1, int(queue_size))
        self.idle_sleep_sec = float(max(1e-4, idle_sleep_sec))
        self.device = torch.device(device) if device is not None else None
        self.pin_memory = bool(pin_memory)
        self.to_device = bool(to_device) and self.device is not None
        self.profiler = profiler

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[_PreparedBatch]" = queue.Queue(
            maxsize=self.queue_size
        )
        self._exception: Optional[BaseException] = None

        self._use_cuda_stream = bool(
            self.to_device
            and self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._cuda_stream = (
            torch.cuda.Stream(device=self.device)
            if self._use_cuda_stream and self.device is not None
            else None
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="libero-batch-prefetch"
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Replay batch prefetcher failed") from self._exception

    def get_queue_size(self) -> int:
        return int(self._queue.qsize())

    def _prepare(self, sampled: Tuple[Dict[str, Any], int, int]) -> _PreparedBatch:
        return _prepare_replay_batch(
            sampled,
            device=self.device,
            pin_memory=self.pin_memory,
            to_device=self.to_device,
            profiler=self.profiler,
            cuda_stream=self._cuda_stream,
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._queue.full():
                    time.sleep(self.idle_sleep_sec)
                    continue

                sample_start = time.perf_counter()
                sampled = self.sample_fn()
                if sampled is not None and self.profiler is not None:
                    self.profiler.record_duration(
                        "replay_sample",
                        (time.perf_counter() - sample_start) * 1000.0,
                    )
                if sampled is None:
                    time.sleep(self.idle_sleep_sec)
                    continue

                prepare_start = time.perf_counter()
                prepared = self._prepare(sampled)
                if self.profiler is not None:
                    self.profiler.record_duration(
                        "replay_prepare",
                        (time.perf_counter() - prepare_start) * 1000.0,
                    )
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(prepared, timeout=self.idle_sleep_sec)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001
            self._exception = exc
            self._stop_event.set()

    def get(
        self,
        timeout: float = 1.0,
        *,
        allow_empty: bool = False,
    ) -> Optional[Tuple[Dict[str, Any], int, int]]:
        while not self._stop_event.is_set():
            self._raise_if_failed()
            try:
                prepared = self._queue.get(timeout=timeout)
            except queue.Empty:
                if allow_empty:
                    return None
                continue

            return _consume_prepared_replay_batch(
                prepared,
                device=self.device,
                profiler=self.profiler,
            )

        self._raise_if_failed()
        return None


def _sample_mixed_batch(
    online_buffer: "ReplayBuffer",
    offline_buffer: Optional["ReplayBuffer"],
    *,
    batch_size: int,
    offline_ratio: float,
    symmetric_replay: bool = False,
) -> Tuple[Dict[str, Any], int, int]:
    if (
        offline_buffer is None
        or len(offline_buffer) == 0
        or ((not symmetric_replay) and offline_ratio <= 0.0)
    ):
        return online_buffer.sample(batch_size=batch_size), int(batch_size), 0

    if symmetric_replay:
        offline_bs = int(batch_size // 2)
        online_bs = int(batch_size - offline_bs)
    else:
        offline_bs = int(round(batch_size * offline_ratio))
        offline_bs = max(0, min(batch_size, offline_bs))
        online_bs = int(batch_size - offline_bs)

    if offline_bs == 0:
        return online_buffer.sample(batch_size=batch_size), int(batch_size), 0
    if online_bs == 0:
        return offline_buffer.sample(batch_size=batch_size), 0, int(batch_size)

    from serl_launcher.utils.train_utils import concat_batches

    online_batch = online_buffer.sample(batch_size=online_bs)
    offline_batch = offline_buffer.sample(batch_size=offline_bs)
    mixed_batch = concat_batches(offline_batch, online_batch, axis=0)
    return mixed_batch, int(online_bs), int(offline_bs)


def _replay_progress_size(buffer: Any) -> int:
    return int(getattr(buffer, "num_steps", len(buffer)))


class _AsyncLearner:
    """In-process async collection-learning coordinator."""

    def __init__(
        self,
        *,
        algorithm: ResidualAgentRuntime,
        learner_agent: Any,
        actor_agent: Any,
        online_buffer: "ReplayBuffer",
        offline_buffer: Optional["ReplayBuffer"],
        batch_size: int,
        offline_ratio: float,
        symmetric_replay: bool,
        training_starts: int,
        utd_ratio: int,
        update_frequency: int,
        idle_sleep_sec: float,
        replay_prefetch_enabled: bool,
        replay_prefetch_queue_size: int,
        replay_prefetch_pin_memory: bool,
        replay_prefetch_to_device: bool,
        checkpoint_writer: Optional[AsyncCheckpointWriter] = None,
        profiler: Optional[_RuntimeProfiler] = None,
    ) -> None:
        self.algorithm = algorithm
        self.learner_agent = learner_agent
        self.actor_agent = actor_agent
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.batch_size = int(batch_size)
        self.offline_ratio = float(offline_ratio)
        self.symmetric_replay = bool(symmetric_replay)
        self.training_starts = int(training_starts)
        self.utd_ratio = int(utd_ratio)
        self.update_frequency = max(1, int(update_frequency))
        self.idle_sleep_sec = float(max(1e-4, idle_sleep_sec))

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prefetcher: Optional[_MixedBatchPrefetcher] = None

        self.replay_lock = threading.Lock()
        self.actor_lock = threading.Lock()
        self.learner_lock = threading.Lock()

        self.update_steps = 0
        self.actor_synced_update_steps = int(self.actor_agent.state.step)
        self.last_update_info: Dict[str, Any] = {}
        self.prefetch_enabled = bool(replay_prefetch_enabled)
        self.prefetch_queue_size = max(1, int(replay_prefetch_queue_size))
        self.prefetch_pin_memory = bool(replay_prefetch_pin_memory)
        self.prefetch_to_device = bool(replay_prefetch_to_device)
        self.checkpoint_writer = checkpoint_writer
        self.profiler = profiler

    def _sync_actor(self) -> None:
        with self.learner_lock:
            with self.actor_lock:
                self.algorithm.sync_modules(self.actor_agent, self.learner_agent)
                self.actor_synced_update_steps = int(self.update_steps)

    def sync_now(self, timeout_sec: Optional[float] = None) -> int:
        self._sync_actor()
        return int(self.actor_synced_update_steps)

    def start(self) -> None:
        if self._thread is not None:
            return
        if self.prefetch_enabled and self._prefetcher is None:
            self._prefetcher = _MixedBatchPrefetcher(
                sample_fn=self._sample_batch,
                queue_size=self.prefetch_queue_size,
                idle_sleep_sec=self.idle_sleep_sec,
                device=self.learner_agent.device,
                pin_memory=self.prefetch_pin_memory,
                to_device=self.prefetch_to_device,
                profiler=self.profiler,
            )
            self._prefetcher.start()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="libero-async-learner"
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._prefetcher is not None:
            self._prefetcher.stop(timeout=min(timeout, 5.0))
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.sync_now()

    def sample_actor_action(
        self, obs_input: Dict[str, np.ndarray], action_dim: int
    ) -> np.ndarray:
        start = time.perf_counter()
        with self.actor_lock:
            sampled = self.algorithm.sample_actions(
                self.actor_agent,
                obs_input,
                deterministic=False,
            )
        if self.profiler is not None:
            self.profiler.record_duration(
                "agent_sample_actions", (time.perf_counter() - start) * 1000.0
            )
        return as_numpy_action(sampled, action_dim)

    def save_checkpoint(self, checkpoint_dir: str, *, step: int, keep: int) -> None:
        with self.learner_lock:
            payload = snapshot_agent_checkpoint_payload(
                self.learner_agent,
                step=step,
            )
        if self.checkpoint_writer is not None:
            self.checkpoint_writer.submit(
                CheckpointTask(
                    checkpoint_dir=checkpoint_dir,
                    payload=payload,
                    step=int(step),
                    keep=int(keep),
                )
            )
        else:
            write_checkpoint_payload_profiled(
                self.profiler,
                checkpoint_dir,
                payload,
                step=step,
                keep=keep,
            )

    def get_last_update_info(self) -> Dict[str, Any]:
        with self.learner_lock:
            return dict(self.last_update_info)

    def get_update_steps(self) -> int:
        with self.learner_lock:
            return int(self.update_steps)

    def wait_for_update_steps(
        self,
        min_update_steps: int,
        *,
        poll_interval_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
    ) -> int:
        target = max(0, int(min_update_steps))
        poll_sec = _resolve_poll_interval(
            poll_interval_sec,
            default_sec=self.idle_sleep_sec,
        )
        deadline = _resolve_timeout_deadline(timeout_sec)
        while True:
            current = self.get_update_steps()
            if current >= target:
                return int(current)
            if deadline is not None and time.monotonic() >= deadline:
                return int(current)
            time.sleep(poll_sec)

    def get_prefetch_queue_size(self) -> int:
        if self._prefetcher is None:
            return 0
        return self._prefetcher.get_queue_size()

    def _sample_batch(self) -> Optional[Tuple[Dict[str, Any], int, int]]:
        with self.replay_lock:
            if _replay_progress_size(self.online_buffer) < self.training_starts:
                return None
            try:
                batch, online_bs, offline_bs = _sample_mixed_batch(
                    self.online_buffer,
                    self.offline_buffer,
                    batch_size=self.batch_size,
                    offline_ratio=self.offline_ratio,
                    symmetric_replay=self.symmetric_replay,
                )
            except (RuntimeError, ValueError) as exc:
                # StepChunk replay can temporarily have inserted steps but still lack a
                # valid chunk start. Treat this as "not ready yet" so the async learner
                # keeps waiting instead of killing the prefetch thread.
                if _is_transient_replay_unavailable(exc):
                    return None
                raise
        return batch, online_bs, offline_bs

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._prefetcher is not None:
                sampled = self._prefetcher.get(timeout=self.idle_sleep_sec)
            else:
                sample_start = time.perf_counter()
                sampled = self._sample_batch()
                if sampled is not None and self.profiler is not None:
                    self.profiler.record_duration(
                        "replay_sample",
                        (time.perf_counter() - sample_start) * 1000.0,
                    )
            if sampled is None:
                time.sleep(self.idle_sleep_sec)
                continue
            if self._prefetcher is not None:
                batch, online_bs, offline_bs = sampled
            else:
                prepare_start = time.perf_counter()
                prepared = _prepare_replay_batch(
                    sampled,
                    device=self.learner_agent.device,
                    pin_memory=self.prefetch_pin_memory,
                    to_device=self.prefetch_to_device,
                    profiler=self.profiler,
                    cuda_stream=None,
                )
                batch, online_bs, offline_bs = _consume_prepared_replay_batch(
                    prepared,
                    device=self.learner_agent.device,
                    profiler=self.profiler,
                )
                if self.profiler is not None:
                    self.profiler.record_duration(
                        "replay_prepare",
                        (time.perf_counter() - prepare_start) * 1000.0,
                    )

            should_sync_actor = False
            with self.learner_lock:
                update_start = time.perf_counter()
                self.learner_agent, info = self.algorithm.update_high_utd(
                    self.learner_agent,
                    batch,
                    utd_ratio=self.utd_ratio,
                )
                if self.profiler is not None:
                    self.profiler.record_duration(
                        "agent_update_high_utd",
                        (time.perf_counter() - update_start) * 1000.0,
                    )
                info["online_batch_size"] = int(online_bs)
                info["offline_batch_size"] = int(offline_bs)
                info["offline_fraction"] = float(
                    offline_bs / max(1, online_bs + offline_bs)
                )
                self.last_update_info = info
                self.update_steps += 1
                if self.update_steps % self.update_frequency == 0:
                    should_sync_actor = True

            if should_sync_actor:
                self._sync_actor()


class _ProcessAsyncLearner:
    """SERL-style local-process learner: actor in main process, learner in worker."""

    def __init__(
        self,
        *,
        algorithm: ResidualAgentRuntime,
        actor_agent: Any,
        online_buffer: "ReplayBuffer",
        offline_buffer: Optional["ReplayBuffer"],
        batch_size: int,
        offline_ratio: float,
        symmetric_replay: bool,
        training_starts: int,
        update_frequency: int,
        idle_sleep_sec: float,
        cfg_dict: Dict[str, Any],
        sample_obs: Dict[str, np.ndarray],
        action_dim: int,
        critic_action_dim: int,
        image_keys: Tuple[str, ...],
        action_transform: Optional[Dict[str, Any]],
        actor_device: Optional[str],
        learner_device: Optional[str],
        batch_queue_size: int = 2,
    ) -> None:
        self.algorithm = algorithm
        self.actor_agent = actor_agent
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.batch_size = int(batch_size)
        self.offline_ratio = float(offline_ratio)
        self.symmetric_replay = bool(symmetric_replay)
        self.training_starts = int(training_starts)
        self.update_frequency = max(1, int(update_frequency))
        self.idle_sleep_sec = float(max(1e-4, idle_sleep_sec))
        self.cfg_dict = dict(cfg_dict)
        self.sample_obs = sample_obs
        self.action_dim = int(action_dim)
        self.critic_action_dim = int(critic_action_dim)
        self.image_keys = tuple(image_keys)
        self.action_transform = action_transform
        self.actor_device = actor_device
        self.learner_device = learner_device

        self.replay_lock = threading.Lock()
        self.actor_lock = threading.Lock()
        self.learner_lock = threading.Lock()

        self.update_steps = 0
        self.actor_synced_update_steps = int(self.actor_agent.state.step)
        self.last_update_info: Dict[str, Any] = {}
        self._exception: Optional[BaseException] = None
        self._process_traceback: Optional[str] = None
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._ctx = mp.get_context("spawn")
        self._batch_queue = self._ctx.Queue(maxsize=max(1, int(batch_queue_size)))
        self._status_queue = self._ctx.Queue(maxsize=1)
        self._command_queue = self._ctx.Queue(maxsize=8)
        self._process: Optional[mp.Process] = None
        self._initial_payload = snapshot_agent_checkpoint_payload(
            self.actor_agent,
            step=int(self.actor_agent.state.step),
        )

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            if self._process_traceback:
                raise RuntimeError(self._process_traceback) from self._exception
            raise RuntimeError("Async learner process failed") from self._exception

    def _drain_status_queue(self) -> None:
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                break

            message_type = str(message.get("type", "status"))
            if message_type == "error":
                self._process_traceback = str(message.get("traceback", ""))
                self._exception = RuntimeError(str(message.get("message", "unknown")))
                self._stop_event.set()
                break

            self.update_steps = int(message.get("update_steps", self.update_steps))
            info = message.get("last_update_info", None)
            if info:
                self.last_update_info = dict(info)
            sync_payload = message.get("sync_payload", None)
            if sync_payload is not None:
                with self.actor_lock:
                    apply_checkpoint_payload_to_agent(
                        self.actor_agent,
                        sync_payload,
                        load_optimizers=False,
                    )
                self.actor_synced_update_steps = int(
                    message.get("update_steps", self.update_steps)
                )

    def _sample_batch(self) -> Optional[Tuple[Dict[str, Any], int, int]]:
        with self.replay_lock:
            if _replay_progress_size(self.online_buffer) < self.training_starts:
                return None
            try:
                return _sample_mixed_batch(
                    self.online_buffer,
                    self.offline_buffer,
                    batch_size=self.batch_size,
                    offline_ratio=self.offline_ratio,
                    symmetric_replay=self.symmetric_replay,
                )
            except (RuntimeError, ValueError) as exc:
                if _is_transient_replay_unavailable(exc):
                    return None
                raise

    def _sampler_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    if self._batch_queue.full():
                        self._drain_status_queue()
                        time.sleep(self.idle_sleep_sec)
                        continue
                except NotImplementedError:
                    pass

                sampled = self._sample_batch()
                if sampled is None:
                    self._drain_status_queue()
                    time.sleep(self.idle_sleep_sec)
                    continue

                while not self._stop_event.is_set():
                    self._drain_status_queue()
                    try:
                        self._batch_queue.put(sampled, timeout=self.idle_sleep_sec)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001
            self._exception = exc
            self._stop_event.set()

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = self._ctx.Process(
            target=_async_process_worker,
            kwargs=dict(
                cfg_dict=self.cfg_dict,
                sample_obs=self.sample_obs,
                action_dim=self.action_dim,
                critic_action_dim=self.critic_action_dim,
                image_keys=self.image_keys,
                action_transform=self.action_transform,
                learner_device=self.learner_device,
                update_frequency=self.update_frequency,
                idle_sleep_sec=self.idle_sleep_sec,
                initial_payload=self._initial_payload,
                batch_queue=self._batch_queue,
                status_queue=self._status_queue,
                command_queue=self._command_queue,
            ),
            daemon=True,
            name="libero-process-learner",
        )
        self._process.start()
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            daemon=True,
            name="libero-process-batch-sampler",
        )
        self._sampler_thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if self._process is None:
            return
        self._drain_status_queue()
        try:
            _put_latest_queue_item(self._command_queue, {"type": "sync_now"})
            _put_latest_queue_item(self._command_queue, {"type": "stop"})
        except Exception:  # noqa: BLE001
            pass
        self._stop_event.set()
        try:
            self._batch_queue.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=timeout)
        self._process.join(timeout=timeout)
        self._drain_status_queue()
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._raise_if_failed()

    def sample_actor_action(
        self, obs_input: Dict[str, np.ndarray], action_dim: int
    ) -> np.ndarray:
        self._drain_status_queue()
        self._raise_if_failed()
        with self.actor_lock:
            sampled = self.algorithm.sample_actions(
                self.actor_agent,
                obs_input,
                deterministic=False,
            )
        return as_numpy_action(sampled, action_dim)

    def save_checkpoint(self, checkpoint_dir: str, *, step: int, keep: int) -> None:
        self._drain_status_queue()
        self._raise_if_failed()
        _put_latest_queue_item(
            self._command_queue,
            {
                "type": "save_checkpoint",
                "checkpoint_dir": str(checkpoint_dir),
                "step": int(step),
                "keep": int(keep),
            },
        )

    def get_last_update_info(self) -> Dict[str, Any]:
        self._drain_status_queue()
        self._raise_if_failed()
        return dict(self.last_update_info)

    def get_update_steps(self) -> int:
        self._drain_status_queue()
        self._raise_if_failed()
        return int(self.update_steps)

    def sync_now(self, timeout_sec: Optional[float] = None) -> int:
        self._drain_status_queue()
        self._raise_if_failed()
        target_update_steps = int(self.update_steps)
        _put_latest_queue_item(self._command_queue, {"type": "sync_now"})
        deadline = _resolve_timeout_deadline(timeout_sec)
        while True:
            self._drain_status_queue()
            self._raise_if_failed()
            if int(self.actor_synced_update_steps) >= target_update_steps:
                return int(self.actor_synced_update_steps)
            if deadline is not None and time.monotonic() >= deadline:
                return int(self.actor_synced_update_steps)
            time.sleep(self.idle_sleep_sec)

    def wait_for_update_steps(
        self,
        min_update_steps: int,
        *,
        poll_interval_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
    ) -> int:
        target = max(0, int(min_update_steps))
        poll_sec = _resolve_poll_interval(
            poll_interval_sec,
            default_sec=self.idle_sleep_sec,
        )
        deadline = _resolve_timeout_deadline(timeout_sec)
        while True:
            current = self.get_update_steps()
            if current >= target:
                return int(current)
            if deadline is not None and time.monotonic() >= deadline:
                return int(current)
            time.sleep(poll_sec)

    def get_prefetch_queue_size(self) -> int:
        self._drain_status_queue()
        self._raise_if_failed()
        try:
            return int(self._batch_queue.qsize())
        except (NotImplementedError, OSError):
            return 0


class _AgentlaceAsyncLearner:
    """Agentlace-backed async learner with learner-owned replay and network sync."""

    def __init__(
        self,
        *,
        algorithm: ResidualAgentRuntime,
        actor_agent: Any,
        replay_buffer: Any,
        offline_buffer: Optional[Any],
        batch_size: int,
        offline_ratio: float,
        symmetric_replay: bool,
        training_starts: int,
        update_frequency: int,
        idle_sleep_sec: float,
        cfg_dict: Dict[str, Any],
        sample_obs: Dict[str, np.ndarray],
        action_dim: int,
        critic_action_dim: int,
        image_keys: Tuple[str, ...],
        action_transform: Optional[Dict[str, Any]],
        learner_device: Optional[str],
        host: str,
        port_number: int,
        broadcast_port: int,
        data_store_queue_size: int = 2000,
        replay_capacity: Optional[int] = None,
        spawn_local_worker: bool = True,
        connect_timeout_sec: float = 120.0,
        request_timeout_ms: Optional[float] = None,
    ) -> None:
        self.algorithm = algorithm
        self.actor_agent = actor_agent
        self.training_starts = int(training_starts)
        self.update_frequency = max(1, int(update_frequency))
        self.idle_sleep_sec = float(max(1e-4, idle_sleep_sec))
        self.actor_lock = threading.Lock()
        self.learner_lock = threading.Lock()
        self.replay_lock = threading.Lock()
        self.update_steps = 0
        self.actor_synced_update_steps = int(self.actor_agent.state.step)
        self.last_update_info: Dict[str, Any] = {}
        self._exception: Optional[BaseException] = None
        self._process_traceback: Optional[str] = None
        self._ctx = mp.get_context("spawn")
        self._status_queue = self._ctx.Queue(maxsize=64)
        self._command_queue = self._ctx.Queue(maxsize=16)
        self._process: Optional[mp.Process] = None
        self._spawn_local_worker = bool(spawn_local_worker)
        self._connect_timeout_sec = float(max(1.0, connect_timeout_sec))
        resolved_request_timeout_ms = (
            3000.0 if request_timeout_ms is None else float(request_timeout_ms)
        )
        # pyzmq socket timeouts must be integer milliseconds.
        self._request_timeout_ms = int(max(800.0, round(resolved_request_timeout_ms)))
        self._client = None
        self._data_store = None
        self._replay_proxy = _ReplayProgressProxy(
            initial_num_steps=int(
                getattr(replay_buffer, "num_steps", len(replay_buffer))
            ),
            initial_sampleable_size=int(_replay_sampleable_size(replay_buffer)),
            capacity=(
                int(replay_capacity)
                if replay_capacity is not None
                else _replay_capacity(replay_buffer)
            ),
            on_insert=self._insert_remote_transition,
        )
        self._cfg_dict = dict(cfg_dict)
        self._sample_obs = sample_obs
        self._action_dim = int(action_dim)
        self._critic_action_dim = int(critic_action_dim)
        self._image_keys = tuple(image_keys)
        self._action_transform = action_transform
        self._learner_device = learner_device
        self._host = str(host)
        self._port_number = int(port_number)
        self._broadcast_port = int(broadcast_port)
        self._data_store_queue_size = max(1, int(data_store_queue_size))
        self._replay_buffer = replay_buffer
        self._offline_buffer = offline_buffer
        self._batch_size = int(batch_size)
        self._offline_ratio = float(offline_ratio)
        self._symmetric_replay = bool(symmetric_replay)
        self._initial_payload = snapshot_agent_checkpoint_payload(
            self.actor_agent,
            step=int(self.actor_agent.state.step),
        )
        self._actor_stats: Dict[str, Any] = {}
        self._server_ready = False
        self._remote_prefetch_queue_size: Optional[int] = None
        self._online_prefill_stats: Dict[str, Any] = {}
        self._client_update_interval = max(1, int(update_frequency))
        self._pending_client_steps = 0
        self._last_client_update_ts = 0.0
        self._status_request_period_sec = max(0.05, float(idle_sleep_sec) * 5.0)
        self._last_status_request_ts = 0.0
        self._closed = False

    @property
    def replay_proxy(self) -> _ReplayProgressProxy:
        return self._replay_proxy

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            if self._process_traceback:
                raise RuntimeError(self._process_traceback) from self._exception
            raise RuntimeError("Agentlace async learner failed") from self._exception

    def _ingest_status_payload(self, payload: Dict[str, Any]) -> None:
        self.update_steps = int(payload.get("update_steps", self.update_steps))
        info = payload.get("last_update_info", None)
        if info:
            self.last_update_info = dict(info)
        online_prefill_stats = payload.get("online_prefill_stats", None)
        if isinstance(online_prefill_stats, dict):
            self._online_prefill_stats = dict(online_prefill_stats)
        prefetch_queue_size = payload.get("replay_prefetch_queue_size", None)
        if prefetch_queue_size is not None:
            self._remote_prefetch_queue_size = int(prefetch_queue_size)
        replay_num_steps = payload.get("replay_num_steps", None)
        replay_sampleable_size = payload.get("replay_sampleable_size", None)
        self._replay_proxy.sync_from_status(
            num_steps=(None if replay_num_steps is None else int(replay_num_steps)),
            sampleable_size=(
                None if replay_sampleable_size is None else int(replay_sampleable_size)
            ),
        )

    def _update_params(self, payload: Dict[str, Any]) -> None:
        with self.actor_lock:
            apply_checkpoint_payload_to_agent(
                self.actor_agent,
                payload,
                load_optimizers=False,
            )
            self.actor_synced_update_steps = int(
                payload.get("step", self.actor_agent.state.step)
            )

    def _ensure_agentlace(self) -> None:
        try:
            from agentlace.data.data_store import QueuedDataStore  # noqa: F401
            from agentlace.trainer import TrainerClient  # noqa: F401
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise RuntimeError(
                "training.async.backend=agentlace requires the 'agentlace' package "
                "in the training environment"
            ) from exc

    def _drain_status_queue(self) -> None:
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                break
            message_type = str(message.get("type", "status"))
            if message_type == "error":
                self._process_traceback = str(message.get("traceback", ""))
                self._exception = RuntimeError(str(message.get("message", "unknown")))
                break
            if message_type == "actor_stats":
                self._actor_stats = dict(message.get("payload", {}))
                continue
            if message_type == "server_ready":
                self._server_ready = True
            self._ingest_status_payload(message)

    def _request_remote_status(self, *, force: bool = False) -> None:
        if self._client is None:
            return
        now = time.monotonic()
        if (not force) and (
            (now - self._last_status_request_ts) < self._status_request_period_sec
        ):
            return
        payload = self._client.request("get-status", {})
        self._last_status_request_ts = now
        if isinstance(payload, dict):
            self._ingest_status_payload(payload)

    def _pump_client(
        self,
        *,
        force_update: bool = False,
        allow_empty_update: bool = False,
    ) -> None:
        self._drain_status_queue()
        self._raise_if_failed()
        if self._client is not None:
            should_update = bool(force_update)
            if not should_update and (
                self._pending_client_steps >= self._client_update_interval
            ):
                should_update = True
            if not should_update and allow_empty_update:
                now = time.monotonic()
                if (
                    now - self._last_client_update_ts
                ) >= self._status_request_period_sec:
                    should_update = True

            if should_update:
                self._client.update()
                self._last_client_update_ts = time.monotonic()
                self._pending_client_steps = 0
                self._request_remote_status(force=True)
            else:
                self._request_remote_status(force=False)
        self._drain_status_queue()
        self._raise_if_failed()

    def flush(self) -> None:
        self._pump_client(force_update=True, allow_empty_update=True)

    def sync_now(self, timeout_sec: Optional[float] = None) -> int:
        if self._closed:
            return int(self.actor_synced_update_steps)
        self._pump_client(force_update=True, allow_empty_update=True)
        target_update_steps = int(self.update_steps)
        if self._client is None:
            return int(self.actor_synced_update_steps)
        payload = self._client.request("sync-now", {})
        if isinstance(payload, dict):
            self._ingest_status_payload(payload)
        deadline = _resolve_timeout_deadline(timeout_sec)
        while True:
            self._drain_status_queue()
            self._raise_if_failed()
            if int(self.actor_synced_update_steps) >= target_update_steps:
                return int(self.actor_synced_update_steps)
            if deadline is not None and time.monotonic() >= deadline:
                return int(self.actor_synced_update_steps)
            time.sleep(self.idle_sleep_sec)

    def start(self) -> None:
        if self._client is not None:
            return
        self._ensure_agentlace()
        from agentlace.data.data_store import QueuedDataStore
        from agentlace.trainer import TrainerClient

        if self._spawn_local_worker:
            self._process = self._ctx.Process(
                target=_agentlace_async_worker,
                kwargs=dict(
                    cfg_dict=self._cfg_dict,
                    sample_obs=self._sample_obs,
                    action_dim=self._action_dim,
                    critic_action_dim=self._critic_action_dim,
                    image_keys=self._image_keys,
                    action_transform=self._action_transform,
                    learner_device=self._learner_device,
                    update_frequency=self.update_frequency,
                    idle_sleep_sec=self.idle_sleep_sec,
                    training_starts=self.training_starts,
                    initial_payload=self._initial_payload,
                    replay_buffer=self._replay_buffer,
                    offline_buffer=self._offline_buffer,
                    batch_size=self._batch_size,
                    offline_ratio=self._offline_ratio,
                    symmetric_replay=self._symmetric_replay,
                    host=self._host,
                    port_number=self._port_number,
                    broadcast_port=self._broadcast_port,
                    status_queue=self._status_queue,
                    command_queue=self._command_queue,
                ),
                daemon=True,
                name="libero-agentlace-learner",
            )
            self._process.start()
            deadline = time.monotonic() + self._connect_timeout_sec
            while (not self._server_ready) and time.monotonic() < deadline:
                self._drain_status_queue()
                self._raise_if_failed()
                if self._server_ready:
                    break
                time.sleep(min(0.1, self.idle_sleep_sec))
            self._drain_status_queue()
            self._raise_if_failed()
            if not self._server_ready:
                raise RuntimeError(
                    "Timed out waiting for local agentlace learner server to start"
                )
        _wait_for_tcp_server(
            self._host,
            self._port_number,
            timeout_sec=self._connect_timeout_sec,
        )
        self._data_store = QueuedDataStore(self._data_store_queue_size)
        self._client = TrainerClient(
            "actor_env",
            self._host,
            _make_agentlace_trainer_config(
                port_number=self._port_number,
                broadcast_port=self._broadcast_port,
            ),
            self._data_store,
            wait_for_server=False,
            timeout_ms=self._request_timeout_ms,
        )
        self._client.recv_network_callback(self._update_params)
        self.flush()
        self._closed = False

    def stop(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        if self._client is not None:
            try:
                self.flush()
            except Exception:  # noqa: BLE001
                pass
        if self._spawn_local_worker:
            try:
                _put_latest_queue_item(self._command_queue, {"type": "sync_now"})
                _put_latest_queue_item(self._command_queue, {"type": "stop"})
            except Exception:  # noqa: BLE001
                pass
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._client = None
        if self._process is not None:
            self._process.join(timeout=timeout)
        self._drain_status_queue()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._process = None
        self._closed = True
        self._raise_if_failed()

    def sample_actor_action(
        self, obs_input: Dict[str, np.ndarray], action_dim: int
    ) -> np.ndarray:
        self._pump_client(force_update=False, allow_empty_update=True)
        with self.actor_lock:
            sampled = self.algorithm.sample_actions(
                self.actor_agent,
                obs_input,
                deterministic=False,
            )
        return as_numpy_action(sampled, action_dim)

    def _insert_remote_transition(self, transition_payload: Dict[str, Any]) -> None:
        if self._data_store is None:
            raise RuntimeError("Agentlace async learner has not been started")
        self._data_store.insert(transition_payload)
        self._pending_client_steps += 1
        self._pump_client(force_update=False, allow_empty_update=False)

    def insert_transition(self, transition_payload: Dict[str, Any]) -> None:
        self._replay_proxy.insert(transition_payload)

    def request_stats(self, payload: Dict[str, Any]) -> None:
        self.flush()
        if self._client is None:
            return
        self._client.request("send-stats", dict(payload))

    def save_checkpoint(self, checkpoint_dir: str, *, step: int, keep: int) -> None:
        self.flush()
        if self._client is None:
            raise RuntimeError("Agentlace async learner has not been started")
        self._client.request(
            "save-checkpoint",
            {
                "checkpoint_dir": str(checkpoint_dir),
                "step": int(step),
                "keep": int(keep),
            },
        )

    def get_last_update_info(self) -> Dict[str, Any]:
        if not self._closed:
            self._pump_client(force_update=False, allow_empty_update=False)
        return dict(self.last_update_info)

    def get_update_steps(self) -> int:
        if not self._closed:
            self._pump_client(force_update=False, allow_empty_update=False)
        return int(self.update_steps)

    def wait_for_update_steps(
        self,
        min_update_steps: int,
        *,
        poll_interval_sec: Optional[float] = None,
        timeout_sec: Optional[float] = None,
    ) -> int:
        target = max(0, int(min_update_steps))
        poll_sec = _resolve_poll_interval(
            poll_interval_sec,
            default_sec=_default_agentlace_poll_interval_sec(self.idle_sleep_sec),
        )
        deadline = _resolve_timeout_deadline(timeout_sec)
        while True:
            if not self._closed:
                self._pump_client(force_update=False, allow_empty_update=True)
            current = int(self.update_steps)
            if current >= target:
                return int(current)
            if deadline is not None and time.monotonic() >= deadline:
                return int(current)
            time.sleep(poll_sec)

    def get_prefetch_queue_size(self) -> int:
        if not self._closed:
            self._pump_client(force_update=False, allow_empty_update=False)
        if self._remote_prefetch_queue_size is not None:
            return int(self._remote_prefetch_queue_size)
        if self._data_store is None:
            return 0
        try:
            return int(self._data_store.qsize())
        except Exception:  # noqa: BLE001
            return 0

    def get_online_prefill_stats(self) -> Dict[str, Any]:
        if not self._closed:
            self._pump_client(force_update=False, allow_empty_update=True)
        return dict(self._online_prefill_stats)
