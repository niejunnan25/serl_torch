"""Checkpoint runtime helpers for training loops and async writers."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from serl_launcher.training.profiling import _RuntimeProfiler
from serl_launcher.utils.checkpoint_utils import checkpoint_dir_size_bytes
from serl_launcher.utils.checkpoint_utils import write_checkpoint_payload


def write_checkpoint_payload_profiled(
    profiler: Optional[_RuntimeProfiler],
    checkpoint_dir: str,
    payload: Dict[str, Any],
    *,
    step: int,
    keep: int,
) -> None:
    checkpoint_dir_path = Path(checkpoint_dir)
    size_before = checkpoint_dir_size_bytes(checkpoint_dir_path)
    start = time.perf_counter()
    checkpoint_path = write_checkpoint_payload(
        checkpoint_dir_path,
        payload,
        step=int(step),
        keep=int(keep),
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    size_after = checkpoint_dir_size_bytes(checkpoint_dir_path)
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


@dataclass
class CheckpointTask:
    checkpoint_dir: str
    payload: Dict[str, Any]
    step: int
    keep: int


class AsyncCheckpointWriter:
    def __init__(self, profiler: Optional[_RuntimeProfiler] = None) -> None:
        self.profiler = profiler
        self._queue: "queue.Queue[Optional[CheckpointTask]]" = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="checkpoint-writer",
        )
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

    def submit(self, task: CheckpointTask) -> None:
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
                write_checkpoint_payload_profiled(
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
