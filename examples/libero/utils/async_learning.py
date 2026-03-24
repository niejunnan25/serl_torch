"""Asynchronous learner and replay prefetch helpers."""
from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from ..policy import as_numpy_action
from .checkpoint import (
    _AsyncCheckpointWriter,
    _CheckpointTask,
    _snapshot_agent_checkpoint_payload,
    _write_checkpoint_payload,
)
from .profiling import _RuntimeProfiler
from .replay_batch import (
    _PreparedBatch,
    _consume_prepared_replay_batch,
    _prepare_replay_batch,
)

if TYPE_CHECKING:
    from serl_launcher.data.replay_buffer import ReplayBuffer


@torch.no_grad()
def _sync_agent_modules_inplace(target_agent: Any, source_agent: Any) -> None:
    for name, source_module in source_agent.state.modules.items():
        if name in target_agent.state.modules:
            target_agent.state.modules[name].load_state_dict(
                source_module.state_dict(), strict=True
            )
    for name, source_module in source_agent.state.target_modules.items():
        if name in target_agent.state.target_modules:
            target_agent.state.target_modules[name].load_state_dict(
                source_module.state_dict(), strict=True
            )
    target_agent.state.step = int(source_agent.state.step)


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

    def get(self, timeout: float = 1.0) -> Optional[Tuple[Dict[str, Any], int, int]]:
        while not self._stop_event.is_set():
            self._raise_if_failed()
            try:
                prepared = self._queue.get(timeout=timeout)
            except queue.Empty:
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
        checkpoint_writer: Optional[_AsyncCheckpointWriter] = None,
        profiler: Optional[_RuntimeProfiler] = None,
    ) -> None:
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
                _sync_agent_modules_inplace(self.actor_agent, self.learner_agent)

    def sync_now(self) -> None:
        self._sync_actor()

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
            sampled = self.actor_agent.sample_actions(obs_input, deterministic=False)
        if self.profiler is not None:
            self.profiler.record_duration(
                "agent_sample_actions", (time.perf_counter() - start) * 1000.0
            )
        return as_numpy_action(sampled, action_dim)

    def save_checkpoint(self, checkpoint_dir: str, *, step: int, keep: int) -> None:
        with self.learner_lock:
            payload = _snapshot_agent_checkpoint_payload(
                self.learner_agent,
                step=step,
            )
        if self.checkpoint_writer is not None:
            self.checkpoint_writer.submit(
                _CheckpointTask(
                    checkpoint_dir=checkpoint_dir,
                    payload=payload,
                    step=int(step),
                    keep=int(keep),
                )
            )
        else:
            _write_checkpoint_payload(
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

    def get_prefetch_queue_size(self) -> int:
        if self._prefetcher is None:
            return 0
        return self._prefetcher.get_queue_size()

    def _sample_batch(self) -> Optional[Tuple[Dict[str, Any], int, int]]:
        with self.replay_lock:
            if _replay_progress_size(self.online_buffer) < self.training_starts:
                return None
            batch, online_bs, offline_bs = _sample_mixed_batch(
                self.online_buffer,
                self.offline_buffer,
                batch_size=self.batch_size,
                offline_ratio=self.offline_ratio,
                symmetric_replay=self.symmetric_replay,
            )
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
                self.learner_agent, info = self.learner_agent.update_high_utd(
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
