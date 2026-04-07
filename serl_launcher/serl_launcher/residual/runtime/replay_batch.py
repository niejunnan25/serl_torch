"""Replay batch preparation and transfer helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from serl_launcher.residual.runtime.profiling import _RuntimeProfiler


def _recursive_to_torch(batch: Any) -> Any:
    if isinstance(batch, dict):
        return {key: _recursive_to_torch(value) for key, value in batch.items()}
    if isinstance(batch, np.ndarray):
        batch = batch if batch.flags.writeable else np.array(batch, copy=True)
        tensor = torch.from_numpy(batch)
    else:
        tensor = torch.as_tensor(batch)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def _recursive_pin_memory(batch: Any) -> Any:
    if isinstance(batch, dict):
        return {key: _recursive_pin_memory(value) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor) and batch.device.type == "cpu":
        try:
            return batch.pin_memory()
        except RuntimeError:
            return batch
    return batch


def _recursive_to_device(batch: Any, device: torch.device, *, non_blocking: bool) -> Any:
    if isinstance(batch, dict):
        return {
            key: _recursive_to_device(value, device=device, non_blocking=non_blocking)
            for key, value in batch.items()
        }
    tensor = batch if isinstance(batch, torch.Tensor) else torch.as_tensor(batch)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor.to(device, non_blocking=non_blocking)


@dataclass
class _PreparedBatch:
    batch: Dict[str, Any]
    online_bs: int
    offline_bs: int
    ready_event: Optional[torch.cuda.Event] = None
    h2d_submit_ms: Optional[float] = None


def _prepare_replay_batch(
    sampled: Tuple[Dict[str, Any], int, int],
    *,
    device: Optional[torch.device],
    pin_memory: bool,
    to_device: bool,
    profiler: Optional[_RuntimeProfiler] = None,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> _PreparedBatch:
    batch, online_bs, offline_bs = sampled
    ready_event: Optional[torch.cuda.Event] = None
    prepared_batch: Any = batch
    h2d_submit_ms: Optional[float] = None

    to_torch_start = time.perf_counter()
    prepared_batch = _recursive_to_torch(prepared_batch)
    if profiler is not None:
        profiler.record_duration("replay_to_torch", (time.perf_counter() - to_torch_start) * 1000.0)

    if pin_memory:
        pin_start = time.perf_counter()
        prepared_batch = _recursive_pin_memory(prepared_batch)
        if profiler is not None:
            profiler.record_duration(
                "replay_pin_memory",
                (time.perf_counter() - pin_start) * 1000.0,
            )

    if to_device and device is not None:
        non_blocking = bool(pin_memory)
        h2d_start = time.perf_counter()
        if cuda_stream is not None:
            with torch.cuda.stream(cuda_stream):
                prepared_batch = _recursive_to_device(
                    prepared_batch,
                    device=device,
                    non_blocking=non_blocking,
                )
            ready_event = torch.cuda.Event()
            ready_event.record(cuda_stream)
        else:
            prepared_batch = _recursive_to_device(
                prepared_batch,
                device=device,
                non_blocking=False,
            )
        h2d_submit_ms = (time.perf_counter() - h2d_start) * 1000.0
        if profiler is not None:
            profiler.record_duration("replay_h2d_submit", h2d_submit_ms)

    return _PreparedBatch(
        batch=prepared_batch,
        online_bs=int(online_bs),
        offline_bs=int(offline_bs),
        ready_event=ready_event,
        h2d_submit_ms=h2d_submit_ms,
    )


def _consume_prepared_replay_batch(
    prepared: _PreparedBatch,
    *,
    device: Optional[torch.device],
    profiler: Optional[_RuntimeProfiler] = None,
) -> Tuple[Dict[str, Any], int, int]:
    wait_ms = 0.0
    profiling_enabled = bool(profiler is not None and profiler.enabled)
    if prepared.ready_event is not None and device is not None and device.type == "cuda":
        if profiling_enabled:
            wait_start = time.perf_counter()
            # Profiling mode intentionally synchronizes here so replay_h2d
            # reflects the true end-to-end copy latency instead of launch time.
            prepared.ready_event.synchronize()
            wait_ms = (time.perf_counter() - wait_start) * 1000.0
        torch.cuda.current_stream(device=device).wait_event(prepared.ready_event)

    if profiler is not None and prepared.h2d_submit_ms is not None:
        profiler.record_duration("replay_h2d_wait", wait_ms)
        profiler.record_duration("replay_h2d", float(prepared.h2d_submit_ms + wait_ms))

    return prepared.batch, int(prepared.online_bs), int(prepared.offline_bs)
