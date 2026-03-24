"""Runtime profiling helpers for LIBERO training scripts."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..policy import build_residual_step_obs
from .logger import JsonlLogger


class _RuntimeProfiler:
    """Thread-safe rolling profiler for stage latencies and scalar values."""

    def __init__(self, *, enabled: bool, window_size: int = 2048) -> None:
        self.enabled = bool(enabled)
        self.window_size = max(1, int(window_size))
        self._lock = threading.Lock()
        self._duration_windows: Dict[str, deque[float]] = {}
        self._duration_totals_ms: Dict[str, float] = {}
        self._duration_total_counts: Dict[str, int] = {}
        self._value_windows: Dict[str, deque[float]] = {}
        self._value_totals: Dict[str, float] = {}
        self._value_total_counts: Dict[str, int] = {}

    def _record(
        self,
        name: str,
        value: float,
        *,
        windows: Dict[str, deque[float]],
        totals: Dict[str, float],
        counts: Dict[str, int],
    ) -> None:
        if not self.enabled:
            return
        numeric_value = float(value)
        with self._lock:
            window = windows.get(name, None)
            if window is None:
                window = deque(maxlen=self.window_size)
                windows[name] = window
            window.append(numeric_value)
            totals[name] = float(totals.get(name, 0.0) + numeric_value)
            counts[name] = int(counts.get(name, 0) + 1)

    def record_duration(self, name: str, value_ms: float) -> None:
        self._record(
            name,
            value_ms,
            windows=self._duration_windows,
            totals=self._duration_totals_ms,
            counts=self._duration_total_counts,
        )

    def record_value(self, name: str, value: float) -> None:
        self._record(
            name,
            value,
            windows=self._value_windows,
            totals=self._value_totals,
            counts=self._value_total_counts,
        )

    def has_data(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            return bool(self._duration_windows or self._value_windows)

    @staticmethod
    def _summarize(
        values: List[float], *, total_count: int, total_sum: float, suffix: str
    ) -> Dict[str, Any]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count_window": int(arr.size),
            "count_total": int(total_count),
            f"mean{suffix}": float(np.mean(arr)),
            f"p95{suffix}": float(np.percentile(arr, 95)),
            f"max{suffix}": float(np.max(arr)),
            f"min{suffix}": float(np.min(arr)),
            f"sum_window{suffix}": float(np.sum(arr)),
            f"sum_total{suffix}": float(total_sum),
        }

    def snapshot(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"durations": {}, "values": {}}
        with self._lock:
            durations = {
                name: self._summarize(
                    list(window),
                    total_count=int(self._duration_total_counts.get(name, 0)),
                    total_sum=float(self._duration_totals_ms.get(name, 0.0)),
                    suffix="_ms",
                )
                for name, window in self._duration_windows.items()
                if len(window) > 0
            }
            values = {
                name: self._summarize(
                    list(window),
                    total_count=int(self._value_total_counts.get(name, 0)),
                    total_sum=float(self._value_totals.get(name, 0.0)),
                    suffix="",
                )
                for name, window in self._value_windows.items()
                if len(window) > 0
            }
        return {"durations": durations, "values": values}


def _profile_call(
    profiler: Optional[_RuntimeProfiler],
    metric_name: str,
    fn: Callable[..., Any],
    *args,
    **kwargs,
) -> Any:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if profiler is not None:
        profiler.record_duration(metric_name, (time.perf_counter() - start) * 1000.0)
    return result


def _build_residual_step_obs_profiled(
    profiler: Optional[_RuntimeProfiler],
    *args,
    **kwargs,
) -> Dict[str, np.ndarray]:
    return _profile_call(
        profiler, "build_residual_step_obs", build_residual_step_obs, *args, **kwargs
    )


def _tb_safe_metric_name(name: str) -> str:
    return str(name).replace(".", "_").replace("/", "_").replace(" ", "_")


def _emit_profiling_snapshot(
    profiler: Optional[_RuntimeProfiler],
    *,
    profile_logger: Optional[JsonlLogger],
    tb_writer: Optional[SummaryWriter],
    logger: logging.Logger,
    train_env_step: int,
    decision_step: int,
    train_episode_id: int,
    learner_update_steps: int,
    replay_prefetch_queue_size: int,
) -> Optional[Dict[str, Any]]:
    if profiler is None or (not profiler.enabled) or (not profiler.has_data()):
        return None

    snapshot = profiler.snapshot()
    payload = {
        "type": "profiling",
        "train_env_step": int(train_env_step),
        "decision_step": int(decision_step),
        "train_episode_id": int(train_episode_id),
        "learner_update_steps": int(learner_update_steps),
        "replay_prefetch_queue_size": int(replay_prefetch_queue_size),
        "metrics": snapshot,
    }
    if profile_logger is not None:
        profile_logger.write(payload)

    if tb_writer is not None:
        for name, stats in snapshot.get("durations", {}).items():
            metric_name = _tb_safe_metric_name(name)
            tb_writer.add_scalar(
                f"profiling/{metric_name}/mean_ms",
                float(stats.get("mean_ms", 0.0)),
                train_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/p95_ms",
                float(stats.get("p95_ms", 0.0)),
                train_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/max_ms",
                float(stats.get("max_ms", 0.0)),
                train_env_step,
            )
        for name, stats in snapshot.get("values", {}).items():
            metric_name = _tb_safe_metric_name(name)
            tb_writer.add_scalar(
                f"profiling/{metric_name}/mean",
                float(stats.get("mean", 0.0)),
                train_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/p95",
                float(stats.get("p95", 0.0)),
                train_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/max",
                float(stats.get("max", 0.0)),
                train_env_step,
            )

    def _summary(name: str) -> str:
        stats = snapshot.get("durations", {}).get(name, None)
        if not stats:
            return f"{name}=n/a"
        return (
            f"{name}=mean:{float(stats.get('mean_ms', 0.0)):.2f}ms "
            f"p95:{float(stats.get('p95_ms', 0.0)):.2f}ms"
        )

    logger.info(
        "profiling train_env_step=%s decision_step=%s learner_updates=%s %s | %s | %s | %s | %s | %s",
        train_env_step,
        decision_step,
        learner_update_steps,
        _summary("env_step"),
        _summary("build_residual_step_obs"),
        _summary("agent_sample_actions"),
        _summary("agent_update_high_utd"),
        _summary("replay_sample"),
        _summary("checkpoint_save"),
    )
    return payload
