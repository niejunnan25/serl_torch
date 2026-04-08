"""Checkpoint payload and writing utilities for training runtimes."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from serl_launcher.training.profiling import _RuntimeProfiler


def _checkpoint_dir_size_bytes(checkpoint_dir: Path) -> int:
    total = 0
    if not checkpoint_dir.exists():
        return 0
    for path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            total += int(path.stat().st_size)
        except OSError:
            continue
    return int(total)


def _clone_to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu_tree(item) for item in value)
    return value


def _snapshot_agent_checkpoint_payload(agent: Any, *, step: int) -> Dict[str, Any]:
    return {
        "step": int(step),
        "params": {
            name: _clone_to_cpu_tree(module.state_dict())
            for name, module in agent.state.modules.items()
        },
        "target_params": {
            name: _clone_to_cpu_tree(module.state_dict())
            for name, module in agent.state.target_modules.items()
        },
        "optimizer": {
            name: _clone_to_cpu_tree(opt.state_dict())
            for name, opt in agent.state.optimizers.items()
        },
    }


def _write_checkpoint_payload(
    profiler: Optional[_RuntimeProfiler],
    checkpoint_dir: str,
    payload: Dict[str, Any],
    *,
    step: int,
    keep: int,
) -> None:
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{int(step)}.pt"
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    size_before = _checkpoint_dir_size_bytes(checkpoint_dir_path)
    start = time.perf_counter()
    torch.save(payload, checkpoint_path)
    if keep is not None and keep > 0:
        checkpoint_paths = sorted(
            checkpoint_dir_path.glob("checkpoint_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
        )
        for stale_path in checkpoint_paths[:-keep]:
            try:
                stale_path.unlink()
            except OSError:
                pass
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    size_after = _checkpoint_dir_size_bytes(checkpoint_dir_path)
    file_size_mb = 0.0
    if checkpoint_path.exists():
        try:
            file_size_mb = float(checkpoint_path.stat().st_size) / (1024.0 * 1024.0)
        except OSError:
            file_size_mb = 0.0
    if profiler is not None:
        profiler.record_duration("checkpoint_save", elapsed_ms)
        profiler.record_value("checkpoint_size_mb", file_size_mb)
        profiler.record_value("checkpoint_dir_size_mb", float(size_after) / (1024.0 * 1024.0))
        profiler.record_value(
            "checkpoint_dir_delta_mb",
            float(size_after - size_before) / (1024.0 * 1024.0),
        )


def _save_checkpoint_profiled(
    profiler: Optional[_RuntimeProfiler],
    checkpoint_dir: str,
    agent: Any,
    *,
    step: int,
    keep: int,
) -> None:
    payload = _snapshot_agent_checkpoint_payload(agent, step=step)
    _write_checkpoint_payload(
        profiler,
        checkpoint_dir,
        payload,
        step=step,
        keep=keep,
    )


@dataclass
class _CheckpointTask:
    checkpoint_dir: str
    payload: Dict[str, Any]
    step: int
    keep: int


class _AsyncCheckpointWriter:
    def __init__(self, profiler: Optional[_RuntimeProfiler] = None) -> None:
        self.profiler = profiler
        self._queue: "queue.Queue[Optional[_CheckpointTask]]" = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True, name="libero-checkpoint-writer")
        self._exception: Optional[BaseException] = None
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Async checkpoint writer failed") from self._exception

    def submit(self, task: _CheckpointTask) -> None:
        if self._closed:
            raise RuntimeError("Async checkpoint writer is closed")
        self.start()
        self._raise_if_failed()
        self._queue.put(task)

    def _run(self) -> None:
        try:
            while True:
                task = self._queue.get()
                if task is None:
                    self._queue.task_done()
                    break
                _write_checkpoint_payload(
                    self.profiler,
                    task.checkpoint_dir,
                    task.payload,
                    step=task.step,
                    keep=task.keep,
                )
                self._queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            self._exception = exc
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
                if pending is None:
                    break

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._raise_if_failed()
        self.start()
        self._queue.put(None)
        if wait:
            self._queue.join()
            self._thread.join(timeout=10.0)
        self._raise_if_failed()
