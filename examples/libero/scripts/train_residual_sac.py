from __future__ import annotations

"""
LIBERO residual policy training (OpenPI + DrQ-SAC).

Minimal stage-1 residual RL loop:
1. OpenPI predicts a base action chunk.
2. Residual policy predicts one residual action per environment step.
3. Final action = base action + bounded residual.
4. Transitions are written step-wise into replay.
5. DrQ-SAC updates from online replay only.

Recommended runtime split:
- OpenPI server + LIBERO env server: run in the `libero` conda env.
- This trainer: run in a serl_torch / newer PyTorch env.
"""

import json
import logging
import pickle
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Tuple

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

    # Keep legacy `gym.*` imports working when only Gymnasium is installed.
    sys.modules["gym"] = gym
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.data import StateActionNormalizer, load_normalizer
from serl_torch.examples.libero.env_wrappers import (
    LiberoTaskEnv,
    RemoteLiberoTaskEnv,
    resolve_openpi_root,
    setup_openpi_client_pythonpath,
)
from serl_torch.examples.libero.policy import (
    LiberoObservationCache,
    OpenPIChunkClient,
    as_numpy_action,
    build_residual_limits,
    build_residual_step_obs,
    compose_residual_action,
    select_action_chunk_window,
)
from serl_torch.examples.libero.utils import JsonlLogger, ensure_serl_launcher_importable
from serl_torch.examples.libero.utils.config_utils import (
    build_drq_agent,
    resolve_control_indices_from_cfg,
    resolve_image_keys,
    sample_probing_steps,
    set_global_seeds,
)

ensure_serl_launcher_importable()

from torch.utils.tensorboard import SummaryWriter

from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches


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
    def _summarize(values: List[float], *, total_count: int, total_sum: float, suffix: str) -> Dict[str, Any]:
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


def _obs_space_from_sample(sample_obs: Dict[str, np.ndarray]) -> gym.spaces.Dict:
    spaces: Dict[str, gym.spaces.Space] = {}
    for key, value in sample_obs.items():
        arr = np.asarray(value)
        if key == "state":
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=np.float32,
            )
        elif np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            spaces[key] = gym.spaces.Box(
                low=info.min,
                high=info.max,
                shape=arr.shape,
                dtype=arr.dtype,
            )
        else:
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=np.float32,
            )
    return gym.spaces.Dict(spaces)


def _clone_obs_dict(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in obs_dict.items()}


def _zero_obs_like(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in obs_dict.items()}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


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
    return _profile_call(profiler, "build_residual_step_obs", build_residual_step_obs, *args, **kwargs)


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


@torch.no_grad()
def _sync_agent_modules_inplace(target_agent: Any, source_agent: Any) -> None:
    for name, source_module in source_agent.state.modules.items():
        if name in target_agent.state.modules:
            target_agent.state.modules[name].load_state_dict(source_module.state_dict(), strict=True)
    for name, source_module in source_agent.state.target_modules.items():
        if name in target_agent.state.target_modules:
            target_agent.state.target_modules[name].load_state_dict(source_module.state_dict(), strict=True)
    target_agent.state.step = int(source_agent.state.step)


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


class _AsyncOpenPIChunkPrefetcher:
    def __init__(self, *, host: str, port: int, logger: logging.Logger) -> None:
        self.host = str(host)
        self.port = int(port)
        self.logger = logger
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-openpi")
        self._client: Optional[OpenPIChunkClient] = None
        self._lock = threading.Lock()
        self._closed = False

    def _get_client(self) -> OpenPIChunkClient:
        with self._lock:
            if self._client is None:
                self._client = OpenPIChunkClient(host=self.host, port=self.port, logger=self.logger)
            return self._client

    def _infer_chunk(
        self,
        obs: Dict[str, Any],
        prompt: str,
        *,
        obs_cache: Optional[LiberoObservationCache] = None,
        cache_key: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
        return self._get_client().infer_chunk(
            obs,
            prompt,
            obs_cache=obs_cache,
            cache_key=cache_key,
        )

    def submit(
        self,
        obs: Dict[str, Any],
        prompt: str,
        *,
        obs_cache: Optional[LiberoObservationCache] = None,
        cache_key: Optional[Any] = None,
    ) -> Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]:
        if self._closed:
            raise RuntimeError("OpenPI prefetcher is closed")
        return self._executor.submit(
            self._infer_chunk,
            obs,
            prompt,
            obs_cache=obs_cache,
            cache_key=cache_key,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)


def _tb_safe_metric_name(name: str) -> str:
    return str(name).replace(".", "_").replace("/", "_").replace(" ", "_")


def _emit_profiling_snapshot(
    profiler: Optional[_RuntimeProfiler],
    *,
    profile_logger: Optional[JsonlLogger],
    tb_writer: Optional[SummaryWriter],
    logger: logging.Logger,
    global_env_step: int,
    global_policy_step: int,
    episode_id: int,
    learner_update_steps: int,
    replay_prefetch_queue_size: int,
) -> Optional[Dict[str, Any]]:
    if profiler is None or (not profiler.enabled) or (not profiler.has_data()):
        return None

    snapshot = profiler.snapshot()
    payload = {
        "type": "profiling",
        "global_env_step": int(global_env_step),
        "global_policy_step": int(global_policy_step),
        "episode_id": int(episode_id),
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
                global_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/p95_ms",
                float(stats.get("p95_ms", 0.0)),
                global_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/max_ms",
                float(stats.get("max_ms", 0.0)),
                global_env_step,
            )
        for name, stats in snapshot.get("values", {}).items():
            metric_name = _tb_safe_metric_name(name)
            tb_writer.add_scalar(
                f"profiling/{metric_name}/mean",
                float(stats.get("mean", 0.0)),
                global_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/p95",
                float(stats.get("p95", 0.0)),
                global_env_step,
            )
            tb_writer.add_scalar(
                f"profiling/{metric_name}/max",
                float(stats.get("max", 0.0)),
                global_env_step,
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
        "profiling step=%s policy_step=%s learner_updates=%s %s | %s | %s | %s | %s | %s",
        global_env_step,
        global_policy_step,
        learner_update_steps,
        _summary("env_step"),
        _summary("build_residual_step_obs"),
        _summary("agent_sample_actions"),
        _summary("agent_update_high_utd"),
        _summary("replay_sample"),
        _summary("checkpoint_save"),
    )
    return payload


def _log_info_scalars(tb_writer, info: Dict[str, Any], global_step: int, pairs: Tuple[Tuple[str, str], ...]) -> None:
    for tb_key, info_key in pairs:
        if info_key in info and info[info_key] is not None:
            tb_writer.add_scalar(tb_key, float(info[info_key]), global_step)


def _new_tb_step_window() -> Dict[str, List[Any]]:
    return {
        "reward": [],
        "residual_scale": [],
        "xi": [],
        "delta_norm": [],
        "policy_norm": [],
        "base_norm": [],
        "final_norm": [],
        "residual_actions": [],
        "delta_actions": [],
        "infer_e2e_ms": [],
        "infer_policy_ms": [],
        "infer_server_ms": [],
    }


def _append_tb_step_window(
    step_window: Dict[str, List[Any]],
    *,
    reward: float,
    residual_scale: float,
    xi: float,
    residual_action: np.ndarray,
    delta_action: np.ndarray,
    base_action: np.ndarray,
    final_action: np.ndarray,
    infer_info: Dict[str, Any],
    replan_point: bool,
) -> None:
    residual_action = np.asarray(residual_action, dtype=np.float32).reshape(-1)
    delta_action = np.asarray(delta_action, dtype=np.float32).reshape(-1)
    base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
    final_action = np.asarray(final_action, dtype=np.float32).reshape(-1)

    step_window["reward"].append(float(reward))
    step_window["residual_scale"].append(float(residual_scale))
    step_window["xi"].append(float(xi))
    step_window["delta_norm"].append(float(np.linalg.norm(delta_action)))
    step_window["policy_norm"].append(float(np.linalg.norm(residual_action)))
    step_window["base_norm"].append(float(np.linalg.norm(base_action)))
    step_window["final_norm"].append(float(np.linalg.norm(final_action)))
    step_window["residual_actions"].append(residual_action.copy())
    step_window["delta_actions"].append(delta_action.copy())

    if replan_point:
        for key, store_key in (
            ("e2e_ms", "infer_e2e_ms"),
            ("policy_ms", "infer_policy_ms"),
            ("server_ms", "infer_server_ms"),
        ):
            value = infer_info.get(key, None)
            if value is not None:
                step_window[store_key].append(float(value))


def _flush_tb_step_window(
    tb_writer,
    *,
    step_window: Dict[str, List[Any]],
    global_env_step: int,
    control_indices: np.ndarray,
    histogram: bool = False,
) -> None:
    if not step_window["reward"]:
        return

    scalar_lists = (
        ("step/reward", "reward"),
        ("step/reward_nonzero_rate", "reward"),
        ("step/residual_scale", "residual_scale"),
        ("step/xi", "xi"),
        ("step/residual_action_magnitude", "delta_norm"),
        ("step/residual_policy_action_magnitude", "policy_norm"),
        ("step/base_action_magnitude", "base_norm"),
        ("step/final_action_magnitude", "final_norm"),
        ("step/infer_e2e_ms", "infer_e2e_ms"),
        ("step/infer_policy_ms", "infer_policy_ms"),
        ("step/infer_server_ms", "infer_server_ms"),
    )
    for tb_key, value_key in scalar_lists:
        values = step_window[value_key]
        if not values:
            continue
        if tb_key == "step/reward_nonzero_rate":
            metric_value = float(np.mean(np.asarray(values, dtype=np.float32) != 0.0))
        else:
            metric_value = float(np.mean(np.asarray(values, dtype=np.float32)))
        tb_writer.add_scalar(tb_key, metric_value, global_env_step)

    residual_actions = np.asarray(step_window["residual_actions"], dtype=np.float32)
    delta_actions = np.asarray(step_window["delta_actions"], dtype=np.float32)
    if residual_actions.size > 0:
        residual_abs = np.abs(residual_actions)
        tb_writer.add_scalar(
            "step/residual_policy_action_abs_mean",
            float(np.mean(residual_abs)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_policy_action_abs_p95",
            float(np.percentile(residual_abs, 95)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_policy_action_saturation_frac",
            float(np.mean(residual_abs >= 0.999)),
            global_env_step,
        )
        for dim_idx, control_idx in enumerate(np.asarray(control_indices, dtype=np.int64).tolist()):
            dim_abs = residual_abs[:, dim_idx]
            tb_writer.add_scalar(
                f"step/residual_policy_action_abs_dim_{int(control_idx)}",
                float(np.mean(dim_abs)),
                global_env_step,
            )
            tb_writer.add_scalar(
                f"step/residual_policy_action_sat_dim_{int(control_idx)}",
                float(np.mean(dim_abs >= 0.999)),
                global_env_step,
            )

    if delta_actions.size > 0:
        controlled_delta = delta_actions[:, np.asarray(control_indices, dtype=np.int64)]
        controlled_delta_abs = np.abs(controlled_delta)
        tb_writer.add_scalar(
            "step/residual_delta_abs_mean",
            float(np.mean(controlled_delta_abs)),
            global_env_step,
        )
        tb_writer.add_scalar(
            "step/residual_delta_abs_p95",
            float(np.percentile(controlled_delta_abs, 95)),
            global_env_step,
        )
        for dim_idx, control_idx in enumerate(np.asarray(control_indices, dtype=np.int64).tolist()):
            dim_abs = controlled_delta_abs[:, dim_idx]
            tb_writer.add_scalar(
                f"step/residual_delta_abs_dim_{int(control_idx)}",
                float(np.mean(dim_abs)),
                global_env_step,
            )
        base_norm = np.mean(np.asarray(step_window["base_norm"], dtype=np.float32))
        delta_norm = np.mean(np.asarray(step_window["delta_norm"], dtype=np.float32))
        tb_writer.add_scalar(
            "step/residual_to_base_ratio",
            float(delta_norm / max(base_norm, 1e-6)),
            global_env_step,
        )

    if histogram and residual_actions.size > 0:
        tb_writer.add_histogram("hist/residual_policy_action", residual_actions.reshape(-1), global_env_step)
    if histogram and delta_actions.size > 0:
        controlled_delta = delta_actions[:, np.asarray(control_indices, dtype=np.int64)]
        tb_writer.add_histogram("hist/residual_delta_action", controlled_delta.reshape(-1), global_env_step)

    for values in step_window.values():
        values.clear()


def _log_update_metrics(tb_writer, update_info: Dict[str, Any], global_env_step: int) -> None:
    _log_info_scalars(
        tb_writer,
        update_info,
        global_env_step,
        (
            ("critic/loss", "critic_loss"),
            ("critic/td_loss", "critic_td_loss"),
            ("critic/cql_penalty", "critic_cql_penalty"),
            ("critic/predicted_qs", "predicted_qs"),
            ("critic/target_qs", "target_qs"),
            ("critic/predicted_q_min", "predicted_q_min"),
            ("critic/predicted_q_max", "predicted_q_max"),
            ("critic/predicted_q_std", "predicted_q_std"),
            ("critic/predicted_q_gap", "predicted_q_gap"),
            ("actor/loss", "actor_loss"),
            ("actor/entropy", "entropy"),
            ("actor/log_prob", "log_prob"),
            ("actor/temperature", "temperature"),
            ("actor/temperature_loss", "temperature_loss"),
            ("actor/temperature_entropy", "temperature_entropy"),
            ("actor/target_entropy", "target_entropy"),
            ("actor/target_entropy_abs", "target_entropy_abs"),
            ("actor/target_entropy_gap", "target_entropy_gap"),
            ("actor/temperature_constraint_gap", "temperature_constraint_gap"),
            ("actor/predicted_q", "actor_predicted_q"),
            ("actor/predicted_q_min", "actor_predicted_q_min"),
            ("actor/predicted_q_std", "actor_predicted_q_std"),
            ("data/online_batch_size", "online_batch_size"),
            ("data/offline_batch_size", "offline_batch_size"),
            ("data/offline_fraction", "offline_fraction"),
            ("optim/actor_lr", "actor_lr"),
            ("optim/critic_lr", "critic_lr"),
            ("optim/temperature_lr", "temperature_lr"),
        ),
    )


def _resolve_offline_paths(dataset_paths: Any, base_dir: Path) -> List[Path]:
    resolved: List[Path] = []
    if dataset_paths is None:
        return resolved

    if isinstance(dataset_paths, (str, Path)):
        items = [dataset_paths]
    else:
        items = list(dataset_paths)

    for item in items:
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if candidate.is_file():
            if candidate.name == "manifest.json":
                with open(candidate, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for episode_file in manifest.get("episode_files", []):
                    resolved.append(Path(str(episode_file)).expanduser().resolve())
            elif candidate.suffix == ".pkl":
                resolved.append(candidate)
        elif candidate.is_dir():
            manifest_path = candidate / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for episode_file in manifest.get("episode_files", []):
                    resolved.append(Path(str(episode_file)).expanduser().resolve())
            else:
                resolved.extend(sorted(candidate.glob("episode_*.pkl")))

    deduped: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _build_offline_frame_obs(payload: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
    return {
        "agentview_rgb": np.asarray(payload["agentview_rgb"][frame_idx], dtype=np.uint8),
        "eye_in_hand_rgb": np.asarray(payload["eye_in_hand_rgb"][frame_idx], dtype=np.uint8),
        "ee_pos": np.asarray(payload["ee_pos"][frame_idx], dtype=np.float32),
        "ee_ori": np.asarray(payload["ee_ori"][frame_idx], dtype=np.float32),
        "gripper_states": np.asarray(payload["gripper_states"][frame_idx], dtype=np.float32),
    }


def _get_episode_prompt(payload: Dict[str, Any], fallback_prompt: str) -> str:
    prompt = payload.get("task_description", payload.get("prompt", fallback_prompt))
    return str(prompt)


def _get_expert_chunk(actions: np.ndarray, chunk_start: int, horizon: int) -> np.ndarray:
    chunk = np.asarray(actions[chunk_start : chunk_start + horizon], dtype=np.float32)
    return select_action_chunk_window(chunk, horizon=horizon)


def _get_base_chunk_for_start(
    payload: Dict[str, Any],
    *,
    chunk_start: int,
    chunk_horizon: int,
    prompt: str,
    openpi_client: OpenPIChunkClient,
    cache: Dict[int, np.ndarray],
    obs_cache: Optional[LiberoObservationCache] = None,
    obs_cache_key: Optional[Any] = None,
) -> np.ndarray:
    if chunk_start in cache:
        return cache[chunk_start]

    stored_base_chunks = payload.get("base_chunks", None)
    stored_horizon = int(payload.get("chunk_horizon", chunk_horizon))
    if stored_base_chunks is not None and stored_horizon == int(chunk_horizon):
        chunk_index = int(chunk_start // chunk_horizon)
        base_chunks = np.asarray(stored_base_chunks, dtype=np.float32)
        if base_chunks.ndim == 3 and chunk_index < base_chunks.shape[0]:
            chunk = select_action_chunk_window(base_chunks[chunk_index], horizon=chunk_horizon)
            cache[chunk_start] = chunk
            return chunk

    obs_raw = _build_offline_frame_obs(payload, chunk_start)
    openpi_chunk, _ = openpi_client.infer_chunk(
        obs_raw,
        prompt,
        obs_cache=obs_cache,
        cache_key=obs_cache_key,
    )
    chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
    cache[chunk_start] = chunk
    return chunk


def _load_offline_residual_buffer(
    cfg: DictConfig,
    *,
    sample_obs_template: Dict[str, np.ndarray],
    offline_buffer: ReplayBuffer,
    action_dim: int,
    full_action_dim: int,
    chunk_horizon: int,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    residual_xi: float,
    openpi_client: OpenPIChunkClient,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    logger: logging.Logger,
    normalizer: Optional[StateActionNormalizer] = None,
    profiler: Optional[_RuntimeProfiler] = None,
) -> Dict[str, int]:
    del full_action_dim
    stats = {
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "clipped_values": 0,
        "errors": 0,
    }

    offline_paths = _resolve_offline_paths(cfg.offline.dataset_paths, Path.cwd())
    stats["files_total"] = len(offline_paths)
    if not offline_paths:
        logger.warning("offline.enabled=true but offline.dataset_paths is empty")
        return stats

    max_transitions = int(cfg.offline.max_transitions) if cfg.offline.max_transitions is not None else None
    expert_reference_scale = max(float(cfg.offline.get("expert_reference_scale", 1.0)), 1e-6)
    xi = max(float(residual_xi), 1e-6)
    denom = residual_limits * xi * expert_reference_scale
    clip_residual_to_unit = bool(cfg.offline.get("clip_residual_to_unit", True))
    fallback_prompt = str(getattr(cfg.task, "prompt", ""))
    obs_cache = LiberoObservationCache(max_obs_entries=256, max_step_obs_entries=512)

    logger.info("offline dataset_paths resolved: %d episode PKL files found", len(offline_paths))
    for path in offline_paths:
        if max_transitions is not None and stats["inserted"] >= max_transitions:
            break
        if not path.exists():
            stats["files_missing"] += 1
            logger.warning("offline dataset not found: %s", path)
            continue

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["skipped"] += 1
            logger.warning("failed to load offline dataset %s: %s", path, exc)
            continue

        if not isinstance(payload, dict) or payload.get("format") != "libero_offline_episode_v1":
            stats["skipped"] += 1
            logger.warning("unsupported offline payload format: %s", path)
            continue

        actions = np.asarray(payload.get("actions", []), dtype=np.float32)
        rewards = np.asarray(payload.get("rewards", np.zeros((actions.shape[0],))), dtype=np.float32).reshape(-1)
        dones = np.asarray(payload.get("dones", np.zeros((actions.shape[0],))), dtype=bool).reshape(-1)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != int(control_indices.max()) + 1:
            if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7:
                stats["skipped"] += 1
                logger.warning("invalid action shape in offline payload %s: %s", path, actions.shape)
                continue

        prompt = _get_episode_prompt(payload, fallback_prompt)
        base_chunk_cache: Dict[int, np.ndarray] = {}
        frame_cache_prefix = str(path)
        stats["files_loaded"] += 1

        for step_idx in range(actions.shape[0]):
            if max_transitions is not None and stats["inserted"] >= max_transitions:
                break

            stats["candidates"] += 1
            chunk_start = int((step_idx // chunk_horizon) * chunk_horizon)
            step_in_chunk = int(step_idx - chunk_start)

            try:
                obs_cache_key = (frame_cache_prefix, int(step_idx))
                obs_raw = _build_offline_frame_obs(payload, step_idx)
                expert_chunk = _get_expert_chunk(actions, chunk_start, chunk_horizon)
                base_chunk = _get_base_chunk_for_start(
                    payload,
                    chunk_start=chunk_start,
                    chunk_horizon=chunk_horizon,
                    prompt=prompt,
                    openpi_client=openpi_client,
                    cache=base_chunk_cache,
                    obs_cache=obs_cache,
                    obs_cache_key=(frame_cache_prefix, int(chunk_start)),
                )
                base_action = base_chunk[step_in_chunk]
                expert_action = expert_chunk[step_in_chunk]
                obs_input = _build_residual_step_obs_profiled(
                    profiler,
                    obs_raw,
                    base_action,
                    image_keys=image_keys,
                    stack_horizon=stack_horizon,
                    normalizer=normalizer,
                    obs_cache=obs_cache,
                    cache_key=obs_cache_key,
                )

                is_last_step = bool(step_idx >= (actions.shape[0] - 1))
                done = bool(dones[step_idx]) or is_last_step
                reward = float(rewards[step_idx]) if step_idx < rewards.shape[0] else float(done)

                if done:
                    next_obs_input = _zero_obs_like(obs_input)
                    mask = 0.0
                else:
                    next_obs_cache_key = (frame_cache_prefix, int(step_idx + 1))
                    next_obs_raw = _build_offline_frame_obs(payload, step_idx + 1)
                    next_chunk_start = int(((step_idx + 1) // chunk_horizon) * chunk_horizon)
                    next_step_in_chunk = int((step_idx + 1) - next_chunk_start)
                    if next_chunk_start == chunk_start:
                        next_base_chunk = base_chunk
                    else:
                        next_base_chunk = _get_base_chunk_for_start(
                            payload,
                            chunk_start=next_chunk_start,
                            chunk_horizon=chunk_horizon,
                            prompt=prompt,
                            openpi_client=openpi_client,
                            cache=base_chunk_cache,
                            obs_cache=obs_cache,
                            obs_cache_key=(frame_cache_prefix, int(next_chunk_start)),
                        )
                    next_obs_input = _build_residual_step_obs_profiled(
                        profiler,
                        next_obs_raw,
                        next_base_chunk[next_step_in_chunk],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                        cache_key=next_obs_cache_key,
                    )
                    mask = 1.0

                raw_residual = (expert_action[control_indices] - base_action[control_indices]) / denom
                stats["clipped_values"] += int(np.count_nonzero((raw_residual < -1.0) | (raw_residual > 1.0)))
                if clip_residual_to_unit:
                    raw_residual = np.clip(raw_residual, -1.0, 1.0)
                residual_step_action = np.asarray(raw_residual, dtype=np.float32).reshape(-1)
                if residual_step_action.shape[0] != action_dim:
                    raise ValueError(
                        f"offline residual action dim mismatch: {residual_step_action.shape[0]} != {action_dim}"
                    )

                offline_buffer.insert(
                    {
                        "observations": _clone_obs_dict(obs_input),
                        "actions": residual_step_action,
                        "next_observations": _clone_obs_dict(next_obs_input),
                        "rewards": np.float32(reward),
                        "masks": np.float32(mask),
                        "dones": bool(done),
                    }
                )
                stats["inserted"] += 1
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                stats["skipped"] += 1
                logger.warning("offline conversion failed file=%s step=%s: %s", path, step_idx, exc)
                continue
    return stats


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
        self._queue: "queue.Queue[_PreparedBatch]" = queue.Queue(maxsize=self.queue_size)
        self._exception: Optional[BaseException] = None

        self._use_cuda_stream = bool(
            self.to_device and self.device is not None and self.device.type == "cuda" and torch.cuda.is_available()
        )
        self._cuda_stream = (
            torch.cuda.Stream(device=self.device) if self._use_cuda_stream and self.device is not None else None
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="libero-batch-prefetch")
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
    online_buffer: ReplayBuffer,
    offline_buffer: Optional[ReplayBuffer],
    *,
    batch_size: int,
    offline_ratio: float,
    symmetric_replay: bool = False,
) -> Tuple[Dict[str, Any], int, int]:
    if offline_buffer is None or len(offline_buffer) == 0 or ((not symmetric_replay) and offline_ratio <= 0.0):
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

    online_batch = online_buffer.sample(batch_size=online_bs)
    offline_batch = offline_buffer.sample(batch_size=offline_bs)
    mixed_batch = concat_batches(offline_batch, online_batch, axis=0)
    return mixed_batch, int(online_bs), int(offline_bs)


class _AsyncLearner:
    """In-process async collection-learning coordinator."""

    def __init__(
        self,
        *,
        learner_agent: Any,
        actor_agent: Any,
        online_buffer: ReplayBuffer,
        offline_buffer: Optional[ReplayBuffer],
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
        self._thread = threading.Thread(target=self._run, daemon=True, name="libero-async-learner")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._prefetcher is not None:
            self._prefetcher.stop(timeout=min(timeout, 5.0))
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.sync_now()

    def sample_actor_action(self, obs_input: Dict[str, np.ndarray], action_dim: int) -> np.ndarray:
        start = time.perf_counter()
        with self.actor_lock:
            sampled = self.actor_agent.sample_actions(obs_input, deterministic=False)
        if self.profiler is not None:
            self.profiler.record_duration("agent_sample_actions", (time.perf_counter() - start) * 1000.0)
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
            if len(self.online_buffer) < self.training_starts:
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
                info["offline_fraction"] = float(offline_bs / max(1, online_bs + offline_bs))
                self.last_update_info = info
                self.update_steps += 1
                if self.update_steps % self.update_frequency == 0:
                    should_sync_actor = True

            if should_sync_actor:
                self._sync_actor()


def _pretrain_critic_with_calql(
    cfg: DictConfig,
    *,
    agent,
    offline_buffer: Optional[ReplayBuffer],
    logger: logging.Logger,
    tb_writer: Optional[SummaryWriter] = None,
) -> Dict[str, Any]:
    calql_cfg = cfg.training.get("calql_pretrain", None)
    if calql_cfg is None or (not bool(calql_cfg.get("enabled", False))):
        return {"enabled": 0, "steps": 0}

    warm_steps = int(calql_cfg.get("steps", 0))
    warm_batch_size = int(calql_cfg.get("batch_size", cfg.replay.batch_size))
    calql_alpha = float(calql_cfg.get("alpha", 0.0))
    calql_n_actions = int(calql_cfg.get("n_actions", cfg.sac.get("cql_n_actions", 10)))
    calql_temperature = float(calql_cfg.get("temperature", cfg.sac.get("cql_temperature", 1.0)))
    if warm_steps <= 0 or calql_alpha <= 0.0 or offline_buffer is None or len(offline_buffer) == 0:
        return {
            "enabled": 0,
            "steps": 0,
            "requested_steps": int(warm_steps),
            "offline_buffer_size": int(len(offline_buffer) if offline_buffer is not None else 0),
        }

    info_last: Dict[str, Any] = {}
    progress = range(warm_steps)
    if tqdm is not None:
        progress = tqdm(progress, desc="Cal-QL critic pretrain", unit="step", dynamic_ncols=True)

    for step in progress:
        batch = offline_buffer.sample(batch_size=warm_batch_size)
        agent, info_last = agent.update_critics_calql(
            batch,
            calql_alpha=calql_alpha,
            calql_n_actions=calql_n_actions,
            calql_temperature=calql_temperature,
        )
        if tqdm is not None and (step % 50 == 0 or step == warm_steps - 1):
            loss_str = f"loss={info_last.get('critic_loss', 0):.3f}"
            if "predicted_qs" in info_last:
                loss_str += f" Q={info_last['predicted_qs']:.2f}"
            progress.set_postfix_str(loss_str)
        if tb_writer is not None:
            _log_info_scalars(
                tb_writer,
                info_last,
                step,
                (
                    ("calql_pretrain/critic_loss", "critic_loss"),
                    ("calql_pretrain/critic_td_loss", "critic_td_loss"),
                    ("calql_pretrain/critic_cql_penalty", "critic_cql_penalty"),
                    ("calql_pretrain/predicted_qs", "predicted_qs"),
                    ("calql_pretrain/target_qs", "target_qs"),
                    ("calql_pretrain/predicted_q_min", "predicted_q_min"),
                    ("calql_pretrain/predicted_q_max", "predicted_q_max"),
                    ("calql_pretrain/predicted_q_std", "predicted_q_std"),
                    ("calql_pretrain/predicted_q_gap", "predicted_q_gap"),
                    ("calql_pretrain/critic_lr", "critic_lr"),
                ),
            )

    logger.info(
        (
            "Cal-QL critic pretrain done: steps=%s batch_size=%s offline_buffer=%s "
            "alpha=%.4f n_actions=%s temp=%.4f"
        ),
        warm_steps,
        warm_batch_size,
        len(offline_buffer),
        calql_alpha,
        calql_n_actions,
        calql_temperature,
    )
    return {
        "enabled": 1,
        "steps": int(warm_steps),
        "batch_size": int(warm_batch_size),
        "alpha": float(calql_alpha),
        "n_actions": int(calql_n_actions),
        "temperature": float(calql_temperature),
        "last_info": info_last,
    }


def _bootstrap_offline_with_base_success(
    cfg: DictConfig,
    *,
    env,
    openpi_client: OpenPIChunkClient,
    offline_buffer: ReplayBuffer,
    sample_obs_template: Dict[str, np.ndarray],
    action_dim: int,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    chunk_horizon: int,
    logger: logging.Logger,
    normalizer: Optional[StateActionNormalizer] = None,
    profiler: Optional[_RuntimeProfiler] = None,
) -> Dict[str, int]:
    stats = {
        "enabled": 0,
        "attempts": 0,
        "episodes_collected": 0,
        "success_episodes": 0,
        "inserted": 0,
        "seed_start": 0,
        "seed_next": 0,
    }

    bootstrap_cfg = cfg.offline.get("bootstrap_base", None)
    if bootstrap_cfg is None or (not bool(bootstrap_cfg.get("enabled", False))):
        return stats

    stats["enabled"] = 1
    target_success_episodes = int(bootstrap_cfg.get("success_episodes", 0))
    if target_success_episodes <= 0:
        logger.warning("offline.bootstrap_base.enabled=true but success_episodes<=0, skip bootstrap")
        return stats

    max_seed_attempts = int(bootstrap_cfg.get("max_seed_attempts", max(1000, target_success_episodes * 100)))
    seed_base_cfg = bootstrap_cfg.get("seed_base", None)
    seed_cursor = int(cfg.task.seed_base) + 1_000_000 if seed_base_cfg is None else int(seed_base_cfg)
    stats["seed_start"] = int(seed_cursor)
    max_ep_steps_override = bootstrap_cfg.get("max_env_steps_per_episode", None)
    only_success = bool(bootstrap_cfg.get("only_success", True))
    obs_cache = LiberoObservationCache()

    while stats["attempts"] < max_seed_attempts and stats["success_episodes"] < target_success_episodes:
        seed = int(seed_cursor)
        seed_cursor += 1
        stats["attempts"] += 1

        obs_cache.clear()
        obs_raw = _profile_call(profiler, "env_reset", env.reset, seed=seed, episode_id=-1)
        max_episode_steps = int(env.step_limit)
        if max_ep_steps_override is not None:
            max_episode_steps = min(max_episode_steps, int(max_ep_steps_override))

        episode_transitions: List[Dict[str, Any]] = []
        episode_steps = 0
        success = False
        episode_done = False
        cached_base_chunk = None

        while episode_steps < max_episode_steps and (not episode_done):
            if cached_base_chunk is None:
                openpi_chunk, _ = openpi_client.infer_chunk(
                    obs_raw,
                    env.current_instruction,
                    obs_cache=obs_cache,
                )
                base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
            else:
                base_chunk = cached_base_chunk
                cached_base_chunk = None

            next_obs_raw = obs_raw
            for chunk_step in range(chunk_horizon):
                if episode_steps >= max_episode_steps:
                    episode_done = True
                    break

                obs_input = _build_residual_step_obs_profiled(
                    profiler,
                    next_obs_raw,
                    base_chunk[chunk_step],
                    image_keys=image_keys,
                    stack_horizon=stack_horizon,
                    normalizer=normalizer,
                    obs_cache=obs_cache,
                )
                next_obs_raw, reward, env_done, _, info = _profile_call(
                    profiler,
                    "env_step",
                    env.step,
                    base_chunk[chunk_step],
                )
                episode_steps += 1
                success = bool(info["success"])
                timeout = bool(episode_steps >= max_episode_steps)
                done = bool(env_done or timeout)

                if done:
                    next_obs_input = _zero_obs_like(obs_input)
                    mask = 0.0
                elif chunk_step < (chunk_horizon - 1):
                    next_obs_input = _build_residual_step_obs_profiled(
                        profiler,
                        next_obs_raw,
                        base_chunk[chunk_step + 1],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                    )
                    mask = 1.0
                else:
                    next_openpi_chunk, _ = openpi_client.infer_chunk(
                        next_obs_raw,
                        env.current_instruction,
                        obs_cache=obs_cache,
                    )
                    next_base_chunk = select_action_chunk_window(next_openpi_chunk, horizon=chunk_horizon)
                    next_obs_input = _build_residual_step_obs_profiled(
                        profiler,
                        next_obs_raw,
                        next_base_chunk[0],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                    )
                    cached_base_chunk = next_base_chunk
                    mask = 1.0

                episode_transitions.append(
                    {
                        "observations": _clone_obs_dict(obs_input),
                        "actions": np.zeros((action_dim,), dtype=np.float32),
                        "next_observations": _clone_obs_dict(next_obs_input),
                        "rewards": np.float32(reward),
                        "masks": np.float32(mask),
                        "dones": bool(done),
                    }
                )

                if done:
                    episode_done = True
                    break

            obs_raw = next_obs_raw

        should_keep = bool(success or (not only_success))
        if should_keep:
            for transition in episode_transitions:
                offline_buffer.insert(transition)
            stats["inserted"] += int(len(episode_transitions))
            stats["episodes_collected"] += 1
            stats["success_episodes"] += int(success)

    stats["seed_next"] = int(seed_cursor)
    return stats


def _scheduled_residual_scale(cfg: DictConfig, phase_scale: float, global_policy_step: int) -> float:
    sched_cfg = cfg.training.get("residual_scale_scheduler", None)
    if sched_cfg is None or (not bool(sched_cfg.get("enabled", False))):
        return float(phase_scale)

    min_scale = float(sched_cfg.get("min_scale", phase_scale))
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    anneal_steps = int(sched_cfg.get("anneal_steps", 1))
    if global_policy_step < warmup_steps:
        return float(min_scale)
    if anneal_steps <= 0:
        return float(phase_scale)
    progress = min(1.0, max(0.0, (global_policy_step - warmup_steps) / float(anneal_steps)))
    return float(min_scale + (float(phase_scale) - min_scale) * progress)


def _scheduled_xi(cfg: DictConfig, base_xi: float, global_policy_step: int) -> float:
    sched_cfg = cfg.training.get("xi_scheduler", None)
    if sched_cfg is None or (not bool(sched_cfg.get("enabled", False))):
        return float(base_xi)

    min_xi = float(sched_cfg.get("min_xi", base_xi))
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    anneal_steps = int(sched_cfg.get("anneal_steps", 1))
    if global_policy_step < warmup_steps:
        return float(min_xi)
    if anneal_steps <= 0:
        return float(base_xi)
    progress = min(1.0, max(0.0, (global_policy_step - warmup_steps) / float(anneal_steps)))
    return float(min_xi + (float(base_xi) - min_xi) * progress)


def _create_env(cfg: DictConfig, logger: logging.Logger):
    env_backend = str(cfg.get("env", {}).get("backend", "remote")).lower()
    common_kwargs = dict(
        suite_name=str(cfg.task.suite_name),
        task_id=int(cfg.task.task_id),
        resolution=int(cfg.task.resolution),
        num_steps_wait=int(cfg.task.num_steps_wait),
        max_episode_steps=(
            int(cfg.task.max_episode_steps) if cfg.task.max_episode_steps is not None else None
        ),
        libero_root=cfg.get("libero_root", None),
        openpi_root=cfg.get("openpi_root", None),
        libero_config_dir=cfg.get("libero_config_dir", None),
        libero_datasets_root=cfg.get("libero_datasets_root", None),
        env_seed_mode=str(cfg.task.get("env_seed_mode", "per_episode")),
        fixed_env_seed=cfg.task.get("fixed_env_seed", None),
        init_state_index_mode=str(cfg.task.get("init_state_index_mode", "seed")),
        logger=logger,
    )
    if env_backend == "local":
        return LiberoTaskEnv(**common_kwargs)
    if env_backend == "remote":
        remote_cfg = cfg.get("env", {}).get("remote", {})
        return RemoteLiberoTaskEnv(
            host=str(remote_cfg.get("host", "127.0.0.1")),
            port=int(remote_cfg.get("port", 30000)),
            timeout_sec=float(remote_cfg.get("timeout_sec", 120.0)),
            **common_kwargs,
        )
    raise ValueError(f"env.backend must be 'local' or 'remote', got {env_backend}")


def _start_async_eval_watcher(
    cfg: DictConfig,
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    logger: logging.Logger,
) -> Tuple[Optional[subprocess.Popen], Optional[IO[str]], Optional[Path], Optional[Path]]:
    async_eval_cfg = cfg.training.get("async_eval", None)
    if async_eval_cfg is None or (not bool(async_eval_cfg.get("enabled", False))):
        return None, None, None, None

    watcher_path = Path(__file__).resolve().parent / "async_eval_watch.py"
    if not watcher_path.exists():
        logger.warning(
            "training.async_eval.enabled=true but watcher script is missing: %s",
            watcher_path,
        )
        return None, None, None, None

    train_cfg_path = run_dir / ".hydra" / "config.yaml"
    summary_jsonl_path = Path(str(async_eval_cfg.get("summary_jsonl", "async_eval_results.jsonl")))
    if not summary_jsonl_path.is_absolute():
        summary_jsonl_path = run_dir / summary_jsonl_path

    cmd = [
        sys.executable,
        str(watcher_path),
        "--train-run-dir",
        str(run_dir),
        "--train-config",
        str(train_cfg_path),
        "--checkpoints-dir",
        str(checkpoint_dir),
        "--summary-jsonl",
        str(summary_jsonl_path),
    ]

    log_path = run_dir / "async_eval_watch.log"
    log_fp = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )

    logger.info(
        "Async eval watcher started: pid=%s log=%s summary=%s",
        proc.pid,
        log_path,
        summary_jsonl_path,
    )
    return proc, log_fp, log_path, summary_jsonl_path


def _stop_async_eval_watcher(
    proc: Optional[subprocess.Popen],
    log_fp: Optional[IO[str]],
    *,
    logger: logging.Logger,
) -> Optional[int]:
    return_code: Optional[int] = None
    try:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    logger.warning("Async eval watcher did not exit in time; killing it")
                    proc.kill()
                    proc.wait(timeout=5.0)
            return_code = proc.returncode
    finally:
        if log_fp is not None:
            log_fp.close()
    return return_code


def _init_async_eval_tb_sync_state(summary_jsonl_path: Optional[Path]) -> Dict[str, Any]:
    processed_lines = 0
    if summary_jsonl_path is not None and summary_jsonl_path.exists():
        try:
            with open(summary_jsonl_path, "r", encoding="utf-8") as f:
                processed_lines = sum(1 for _ in f)
        except Exception:  # noqa: BLE001
            processed_lines = 0
    return {"processed_lines": int(processed_lines)}


def _sync_async_eval_results_to_tb(
    tb_writer: SummaryWriter,
    *,
    summary_jsonl_path: Optional[Path],
    sync_state: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    if summary_jsonl_path is None or (not summary_jsonl_path.exists()):
        return

    try:
        with open(summary_jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to read async eval summary jsonl: %s", exc)
        return

    processed_lines = int(sync_state.get("processed_lines", 0))
    if processed_lines < 0:
        processed_lines = 0
    if processed_lines > len(lines):
        processed_lines = 0

    new_lines = lines[processed_lines:]
    if not new_lines:
        return

    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        step_raw = payload.get("step", None)
        try:
            if step_raw is None:
                continue
            step = int(step_raw)
        except Exception:  # noqa: BLE001
            continue

        status = str(payload.get("status", "")).lower()
        duration_sec = payload.get("duration_sec", None)
        return_code = payload.get("return_code", None)

        if isinstance(duration_sec, (int, float)):
            tb_writer.add_scalar("async_eval/duration_sec", float(duration_sec), step)
        if isinstance(return_code, (int, float)):
            tb_writer.add_scalar("async_eval/return_code", float(return_code), step)

        if status == "ok":
            tb_writer.add_scalar("async_eval/status_ok", 1.0, step)
            tb_writer.add_scalar("async_eval/status_failed", 0.0, step)
            summary = payload.get("summary", None)
            if isinstance(summary, dict):
                success_rate = summary.get("success_rate", None)
                total_success = summary.get("total_success", None)
                episodes = summary.get("episodes", None)
                if isinstance(success_rate, (int, float)):
                    tb_writer.add_scalar("async_eval/success_rate", float(success_rate), step)
                if isinstance(total_success, (int, float)):
                    tb_writer.add_scalar("async_eval/total_success", float(total_success), step)
                if isinstance(episodes, (int, float)):
                    tb_writer.add_scalar("async_eval/eval_episodes", float(episodes), step)
        elif status == "failed":
            tb_writer.add_scalar("async_eval/status_ok", 0.0, step)
            tb_writer.add_scalar("async_eval/status_failed", 1.0, step)
        elif status == "aborted":
            tb_writer.add_scalar("async_eval/status_ok", 0.0, step)
            tb_writer.add_scalar("async_eval/status_failed", 1.0, step)

    sync_state["processed_lines"] = int(len(lines))
    tb_writer.flush()


@hydra.main(version_base=None, config_path="../conf", config_name="train_residual_sac")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("libero_train_residual_sac")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_global_seeds(int(cfg.seed))

    openpi_root = resolve_openpi_root(cfg.get("openpi_root", None))
    setup_openpi_client_pythonpath(openpi_root)
    logger.info("openpi root: %s", openpi_root)

    env = _create_env(cfg, logger)
    logger.info(
        "LIBERO task: suite=%s task_id=%s prompt=%s",
        cfg.task.suite_name,
        cfg.task.task_id,
        env.current_instruction,
    )

    normalizer: StateActionNormalizer | None = None
    norm_cfg = cfg.get("normalization", None)
    if norm_cfg is not None and bool(norm_cfg.get("enabled", False)):
        task_key = f"{cfg.task.suite_name}_task_{int(cfg.task.task_id)}"
        normalizer = load_normalizer(task_key, stats_dir=norm_cfg.get("stats_dir", None))
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)

    openpi_client = OpenPIChunkClient(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )

    image_keys = resolve_image_keys(cfg)
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    control_indices = resolve_control_indices_from_cfg(cfg)
    action_dim = int(len(control_indices))
    chunk_horizon = int(cfg.residual.chunk_horizon)
    residual_xi = float(cfg.residual.get("xi", 1.0))
    residual_limits = build_residual_limits(
        control_indices,
        arm_limit=float(cfg.residual.arm_delta_limit),
        gripper_limit=float(cfg.residual.gripper_delta_limit),
    )
    logger.info(
        "Residual config: image_keys=%s action_dim=%s action_indices=%s chunk_horizon=%s xi=%.4f",
        list(image_keys),
        action_dim,
        control_indices.tolist(),
        chunk_horizon,
        residual_xi,
    )

    offline_enabled = bool(cfg.offline.enabled)
    offline_ratio = float(cfg.offline.ratio)
    if not (0.0 <= offline_ratio <= 1.0):
        raise ValueError(f"offline.ratio must be in [0,1], got {offline_ratio}")
    symmetric_replay = bool(cfg.offline.get("symmetric_replay", False))
    async_cfg = cfg.training.get("async", None)
    async_enabled = bool(async_cfg.get("enabled", False)) if async_cfg is not None else False
    async_update_frequency = int(async_cfg.get("update_frequency", 1)) if async_cfg is not None else 1
    async_idle_sleep_sec = float(async_cfg.get("idle_sleep_sec", 0.002)) if async_cfg is not None else 0.002
    replay_prefetch_cfg = cfg.training.get("replay_prefetch", None)
    replay_prefetch_enabled = (
        bool(replay_prefetch_cfg.get("enabled", True)) if replay_prefetch_cfg is not None else True
    )
    replay_prefetch_queue_size = (
        int(replay_prefetch_cfg.get("queue_size", 2)) if replay_prefetch_cfg is not None else 2
    )
    replay_prefetch_pin_memory = (
        bool(replay_prefetch_cfg.get("pin_memory", True)) if replay_prefetch_cfg is not None else True
    )
    replay_prefetch_to_device = (
        bool(replay_prefetch_cfg.get("to_device", True)) if replay_prefetch_cfg is not None else True
    )
    profiling_cfg = cfg.training.get("profiling", None)
    profiling_enabled = bool(profiling_cfg.get("enabled", False)) if profiling_cfg is not None else False
    profiling_window_size = int(profiling_cfg.get("window_size", 2048)) if profiling_cfg is not None else 2048
    profiling_log_period_steps = (
        int(profiling_cfg.get("log_period_steps", 500)) if profiling_cfg is not None else 500
    )
    profiling_log_file = (
        str(profiling_cfg.get("log_file", "profiling_logs.jsonl"))
        if profiling_cfg is not None
        else "profiling_logs.jsonl"
    )
    if async_enabled and any((not bool(phase.get("train", True))) for phase in cfg.training.phases):
        logger.warning(
            "Detected non-train phase in training.phases; disable async mode to preserve phase semantics."
        )
        async_enabled = False
    logger.info(
        "Async collection-learning: enabled=%s update_frequency=%s idle_sleep_sec=%.4f",
        async_enabled,
        async_update_frequency,
        async_idle_sleep_sec,
    )
    logger.info(
        "Replay batch prefetch: enabled=%s queue_size=%s pin_memory=%s to_device=%s",
        replay_prefetch_enabled,
        replay_prefetch_queue_size,
        replay_prefetch_pin_memory,
        replay_prefetch_to_device,
    )
    logger.info(
        "Profiling: enabled=%s window_size=%s log_period_steps=%s log_file=%s",
        profiling_enabled,
        profiling_window_size,
        profiling_log_period_steps,
        profiling_log_file,
    )

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    agent = None
    learner_agent = None
    async_learner: Optional[_AsyncLearner] = None
    sync_replay_lock: Optional[threading.Lock] = None
    sync_replay_prefetcher: Optional[_MixedBatchPrefetcher] = None
    checkpoint_writer: Optional[_AsyncCheckpointWriter] = None
    openpi_prefetcher: Optional[_AsyncOpenPIChunkPrefetcher] = None
    replay_buffer = None
    offline_buffer = None
    offline_stats: Dict[str, Any] = {
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "clipped_values": 0,
        "errors": 0,
    }
    bootstrap_stats: Dict[str, Any] = {"enabled": 0, "inserted": 0}
    warmstart_info: Dict[str, Any] = {"enabled": 0, "steps": 0}

    checkpoint_dir = Path(str(cfg.training.checkpoint_dir))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async_eval_proc: Optional[subprocess.Popen] = None
    async_eval_log_fp: Optional[IO[str]] = None
    async_eval_log_path: Optional[Path] = None
    async_eval_summary_path: Optional[Path] = None
    async_eval_watcher_return_code: Optional[int] = None
    async_eval_dead_reported = False
    profiler = _RuntimeProfiler(enabled=profiling_enabled, window_size=profiling_window_size)
    checkpoint_writer = _AsyncCheckpointWriter(profiler=profiler)
    profiling_logger: Optional[JsonlLogger] = None
    profiling_last_flush_step = -1
    (
        async_eval_proc,
        async_eval_log_fp,
        async_eval_log_path,
        async_eval_summary_path,
    ) = _start_async_eval_watcher(
        cfg,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
    )

    step_logger = JsonlLogger(run_dir / str(cfg.logging.step_log_file))
    episode_logger = JsonlLogger(run_dir / str(cfg.logging.episode_log_file))
    openpi_prefetcher = _AsyncOpenPIChunkPrefetcher(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )
    if profiling_enabled:
        profiling_logger = JsonlLogger(run_dir / profiling_log_file)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    tb_step_period = int(cfg.logging.get("tb_step_period", 100))
    tb_histogram_period = max(
        tb_step_period,
        int(cfg.logging.get("tb_histogram_period", max(tb_step_period * 10, 1))),
    )
    step_metric_window = _new_tb_step_window()
    async_eval_tb_sync_state = _init_async_eval_tb_sync_state(async_eval_summary_path)
    obs_cache = LiberoObservationCache()

    sample_obs_raw = _profile_call(profiler, "env_reset", env.reset, seed=int(cfg.task.seed_base), episode_id=-1)
    sample_openpi_chunk, _ = openpi_client.infer_chunk(
        sample_obs_raw,
        env.current_instruction,
        obs_cache=obs_cache,
    )
    sample_base_chunk = select_action_chunk_window(sample_openpi_chunk, horizon=chunk_horizon)
    sample_obs = _build_residual_step_obs_profiled(
        profiler,
        sample_obs_raw,
        sample_base_chunk[0],
        image_keys=image_keys,
        stack_horizon=stack_horizon,
        normalizer=normalizer,
        obs_cache=obs_cache,
    )

    learner_agent = build_drq_agent(
        cfg,
        sample_obs=sample_obs,
        action_dim=action_dim,
        image_keys=image_keys,
    )
    agent = learner_agent
    replay_buffer = ReplayBuffer(
        observation_space=_obs_space_from_sample(sample_obs),
        action_space=action_space,
        capacity=int(cfg.replay.capacity),
    )
    if offline_enabled:
        offline_buffer = ReplayBuffer(
            observation_space=_obs_space_from_sample(sample_obs),
            action_space=action_space,
            capacity=int(cfg.offline.capacity),
        )
        bootstrap_stats = _bootstrap_offline_with_base_success(
            cfg,
            env=env,
            openpi_client=openpi_client,
            offline_buffer=offline_buffer,
            sample_obs_template=sample_obs,
            action_dim=action_dim,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            chunk_horizon=chunk_horizon,
            logger=logger,
            normalizer=normalizer,
            profiler=profiler,
        )
        offline_stats = _load_offline_residual_buffer(
            cfg,
            sample_obs_template=sample_obs,
            offline_buffer=offline_buffer,
            action_dim=action_dim,
            full_action_dim=action_dim,
            chunk_horizon=chunk_horizon,
            control_indices=control_indices,
            residual_limits=residual_limits,
            residual_xi=residual_xi,
            openpi_client=openpi_client,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            logger=logger,
            normalizer=normalizer,
            profiler=profiler,
        )
        logger.info(
            "offline bootstrap: success_episodes=%s collected=%s inserted=%s attempts=%s",
            bootstrap_stats.get("success_episodes", 0),
            bootstrap_stats.get("episodes_collected", 0),
            bootstrap_stats.get("inserted", 0),
            bootstrap_stats.get("attempts", 0),
        )
        logger.info(
            "offline preload: buffer=%s files_loaded=%s/%s candidates=%s inserted=%s skipped=%s clipped=%s errors=%s",
            len(offline_buffer),
            offline_stats.get("files_loaded", 0),
            offline_stats.get("files_total", 0),
            offline_stats.get("candidates", 0),
            offline_stats.get("inserted", 0),
            offline_stats.get("skipped", 0),
            offline_stats.get("clipped_values", 0),
            offline_stats.get("errors", 0),
        )
        warmstart_info = _pretrain_critic_with_calql(
            cfg,
            agent=learner_agent,
            offline_buffer=offline_buffer,
            logger=logger,
            tb_writer=tb_writer,
        )
    if async_enabled:
        agent = build_drq_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=action_dim,
            image_keys=image_keys,
        )
        _sync_agent_modules_inplace(agent, learner_agent)
        async_learner = _AsyncLearner(
            learner_agent=learner_agent,
            actor_agent=agent,
            online_buffer=replay_buffer,
            offline_buffer=offline_buffer if offline_enabled else None,
            batch_size=int(cfg.replay.batch_size),
            offline_ratio=offline_ratio,
            symmetric_replay=symmetric_replay,
            training_starts=int(cfg.training.training_starts),
            utd_ratio=int(cfg.sac.utd_ratio),
            update_frequency=async_update_frequency,
            idle_sleep_sec=async_idle_sleep_sec,
            replay_prefetch_enabled=replay_prefetch_enabled,
            replay_prefetch_queue_size=replay_prefetch_queue_size,
            replay_prefetch_pin_memory=replay_prefetch_pin_memory,
            replay_prefetch_to_device=replay_prefetch_to_device,
            checkpoint_writer=checkpoint_writer,
            profiler=profiler,
        )
        async_learner.start()
    elif replay_prefetch_enabled:
        sync_replay_lock = threading.Lock()

        def _sample_sync_prefetch_batch() -> Optional[Tuple[Dict[str, Any], int, int]]:
            assert replay_buffer is not None
            assert sync_replay_lock is not None
            with sync_replay_lock:
                if len(replay_buffer) < int(cfg.training.training_starts):
                    return None
                return _sample_mixed_batch(
                    replay_buffer,
                    offline_buffer if offline_enabled else None,
                    batch_size=int(cfg.replay.batch_size),
                    offline_ratio=offline_ratio,
                    symmetric_replay=symmetric_replay,
                )

        sync_replay_prefetcher = _MixedBatchPrefetcher(
            sample_fn=_sample_sync_prefetch_batch,
            queue_size=replay_prefetch_queue_size,
            idle_sleep_sec=async_idle_sleep_sec,
            device=learner_agent.device,
            pin_memory=replay_prefetch_pin_memory,
            to_device=replay_prefetch_to_device,
            profiler=profiler,
        )
        sync_replay_prefetcher.start()
    logger.info("Initialized DrQ agent, replay buffer, and offline pipeline")

    global_env_step = 0
    global_policy_step = 0
    episode_id = 0
    total_success = 0
    recent_successes: deque[int] = deque(maxlen=20)
    skipped_seeds = 0
    seed_cursor = int(cfg.task.seed_base)
    stopped_by_env_budget = False
    last_update_info: Dict[str, Any] = {}

    max_online_env_steps = int(cfg.training.get("max_online_env_steps", 0))
    warmup_base_episodes = int(cfg.training.get("warmup_base_episodes", 0))
    warmup_base_steps = int(cfg.training.get("warmup_base_steps", 0))

    assert agent is not None
    assert learner_agent is not None
    assert replay_buffer is not None

    try:
        for phase in cfg.training.phases:
            if max_online_env_steps > 0 and global_env_step >= max_online_env_steps:
                stopped_by_env_budget = True
                break
            phase_name = str(phase.name)
            phase_episodes = int(phase.episodes)
            phase_train = bool(phase.get("train", True))
            phase_residual_scale = float(phase.residual_scale)
            logger.info(
                "Start phase=%s episodes=%s train=%s residual_scale=%.4f",
                phase_name,
                phase_episodes,
                phase_train,
                phase_residual_scale,
            )

            phase_episode_count = 0
            while phase_episode_count < phase_episodes:
                if max_online_env_steps > 0 and global_env_step >= max_online_env_steps:
                    stopped_by_env_budget = True
                    break

                seed = int(seed_cursor)
                seed_cursor += 1

                if bool(cfg.training.get("expert_check", False)):
                    passed, _ = env.expert_precheck(seed=seed, episode_id=episode_id)
                    if not passed:
                        skipped_seeds += 1
                        logger.warning("skip seed=%s in phase=%s: expert precheck failed", seed, phase_name)
                        continue

                obs_cache.clear()
                obs_raw = _profile_call(profiler, "env_reset", env.reset, seed=seed, episode_id=episode_id)
                max_episode_steps = int(env.step_limit)
                if cfg.training.max_env_steps_per_episode is not None:
                    max_episode_steps = min(max_episode_steps, int(cfg.training.max_env_steps_per_episode))

                episode_success = False
                episode_return = 0.0
                episode_steps = 0
                episode_done = False
                cached_base_chunk = None
                cached_infer_info = None

                probing_steps_target = sample_probing_steps(cfg.training, episode_horizon=max_episode_steps)
                if probing_steps_target > 0:
                    probing_remaining = int(min(probing_steps_target, max_episode_steps - episode_steps))
                    probe_future: Optional[Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]] = None
                    while probing_remaining > 0 and episode_steps < max_episode_steps:
                        if probe_future is not None:
                            probe_chunk, probe_info = probe_future.result()
                            probe_future = None
                        else:
                            probe_chunk, probe_info = openpi_client.infer_chunk(
                                obs_raw,
                                env.current_instruction,
                                obs_cache=obs_cache,
                            )
                        probe_base_chunk = select_action_chunk_window(probe_chunk, horizon=chunk_horizon)
                        for probe_step in range(chunk_horizon):
                            if probing_remaining <= 0 or episode_steps >= max_episode_steps:
                                break
                            base_action = probe_base_chunk[probe_step]
                            next_obs_raw, reward, env_done, _, info = _profile_call(
                                profiler,
                                "env_step",
                                env.step,
                                base_action,
                            )
                            episode_steps += 1
                            global_env_step += 1
                            probing_remaining -= 1
                            episode_return += float(reward)
                            episode_success = bool(info["success"])
                            timeout = bool(episode_steps >= max_episode_steps)
                            budget_exhausted = bool(
                                max_online_env_steps > 0 and global_env_step >= max_online_env_steps
                            )
                            done = bool(env_done or timeout or budget_exhausted)
                            if (
                                (not done)
                                and probe_step == (chunk_horizon - 1)
                                and probing_remaining > 0
                                and openpi_prefetcher is not None
                            ):
                                probe_future = openpi_prefetcher.submit(
                                    next_obs_raw,
                                    env.current_instruction,
                                    obs_cache=obs_cache,
                                )
                            step_logger.write(
                                {
                                    "global_env_step": int(global_env_step),
                                    "global_policy_step": int(global_policy_step),
                                    "episode_id": episode_id,
                                    "phase": phase_name,
                                    "episode_step": episode_steps,
                                    "seed": int(env.last_seed if env.last_seed is not None else seed),
                                    "init_state_idx": (
                                        int(env.current_init_state_idx)
                                        if env.current_init_state_idx is not None
                                        else None
                                    ),
                                    "is_probing": True,
                                    "replan_point": bool(probe_step == 0),
                                    "chunk_step": int(probe_step),
                                    "chunk_horizon": int(chunk_horizon),
                                    "infer_e2e_ms": probe_info.get("e2e_ms") if probe_step == 0 else None,
                                    "infer_policy_ms": probe_info.get("policy_ms") if probe_step == 0 else None,
                                    "infer_server_ms": probe_info.get("server_ms") if probe_step == 0 else None,
                                    "a_base": base_action.tolist(),
                                    "a_res_policy": [0.0] * action_dim,
                                    "a_res": np.zeros_like(base_action, dtype=np.float32).tolist(),
                                    "a_final": base_action.tolist(),
                                    "residual_scale": 0.0,
                                    "reward": float(reward),
                                    "done": bool(done),
                                    "success": bool(episode_success),
                                }
                            )
                            obs_raw = next_obs_raw
                            if done:
                                episode_done = True
                                break
                        if episode_done:
                            break

                while (episode_steps < max_episode_steps) and (not episode_done):
                    if cached_base_chunk is None:
                        openpi_chunk, infer_info = openpi_client.infer_chunk(
                            obs_raw,
                            env.current_instruction,
                            obs_cache=obs_cache,
                        )
                        base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
                    else:
                        base_chunk = cached_base_chunk
                        infer_info = cached_infer_info or {
                            "e2e_ms": None,
                            "policy_ms": None,
                            "server_ms": None,
                        }
                        cached_base_chunk = None
                        cached_infer_info = None

                    next_obs_raw = obs_raw
                    for chunk_step in range(chunk_horizon):
                        if episode_steps >= max_episode_steps:
                            episode_done = True
                            break

                        obs_input = _build_residual_step_obs_profiled(
                            profiler,
                            next_obs_raw,
                            base_chunk[chunk_step],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            normalizer=normalizer,
                            obs_cache=obs_cache,
                        )

                        residual_scale_step = _scheduled_residual_scale(
                            cfg,
                            phase_scale=phase_residual_scale,
                            global_policy_step=global_policy_step,
                        )
                        xi_step = _scheduled_xi(
                            cfg,
                            base_xi=residual_xi,
                            global_policy_step=global_policy_step,
                        )

                        in_warmup_episode = bool(episode_id < warmup_base_episodes)
                        in_warmup_step = bool(
                            warmup_base_steps > 0 and global_policy_step < warmup_base_steps
                        )
                        if phase_train and (in_warmup_episode or in_warmup_step):
                            residual_step_action = np.zeros((action_dim,), dtype=np.float32)
                        elif residual_scale_step <= 0.0:
                            residual_step_action = np.zeros((action_dim,), dtype=np.float32)
                        elif (not phase_train) or (global_policy_step < int(cfg.training.random_steps)):
                            residual_step_action = np.random.uniform(
                                -1.0, 1.0, size=(action_dim,)
                            ).astype(np.float32)
                            residual_step_action *= float(cfg.training.random_action_scale)
                        else:
                            if async_learner is not None:
                                residual_step_action = async_learner.sample_actor_action(obs_input, action_dim)
                            else:
                                sample_actions_start = time.perf_counter()
                                sampled = agent.sample_actions(obs_input, deterministic=False)
                                profiler.record_duration(
                                    "agent_sample_actions",
                                    (time.perf_counter() - sample_actions_start) * 1000.0,
                                )
                                residual_step_action = as_numpy_action(sampled, action_dim)

                        delta_action, final_action = compose_residual_action(
                            base_action=base_chunk[chunk_step],
                            residual_action=residual_step_action,
                            indices=control_indices,
                            limits=residual_limits,
                            residual_scale=residual_scale_step,
                            xi=xi_step,
                            clip_gripper=bool(cfg.residual.clip_gripper),
                        )

                        next_obs_raw, reward, env_done, _, info = _profile_call(
                            profiler,
                            "env_step",
                            env.step,
                            final_action,
                        )
                        episode_steps += 1
                        global_env_step += 1
                        episode_return += float(reward)
                        episode_success = bool(info["success"])
                        timeout = bool(episode_steps >= max_episode_steps)
                        budget_exhausted = bool(
                            max_online_env_steps > 0 and global_env_step >= max_online_env_steps
                        )
                        done = bool(env_done or timeout or budget_exhausted)
                        next_chunk_future: Optional[Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]] = None
                        if (
                            (not done)
                            and chunk_step == (chunk_horizon - 1)
                            and openpi_prefetcher is not None
                        ):
                            next_chunk_future = openpi_prefetcher.submit(
                                next_obs_raw,
                                env.current_instruction,
                                obs_cache=obs_cache,
                            )

                        step_logger.write(
                            {
                                "global_env_step": int(global_env_step),
                                "global_policy_step": int(global_policy_step),
                                "episode_id": episode_id,
                                "phase": phase_name,
                                "episode_step": episode_steps,
                                "seed": int(env.last_seed if env.last_seed is not None else seed),
                                "init_state_idx": (
                                    int(env.current_init_state_idx)
                                    if env.current_init_state_idx is not None
                                    else None
                                ),
                                "is_probing": False,
                                "replan_point": bool(chunk_step == 0),
                                "chunk_step": int(chunk_step),
                                "chunk_horizon": int(chunk_horizon),
                                "infer_e2e_ms": infer_info.get("e2e_ms") if chunk_step == 0 else None,
                                "infer_policy_ms": infer_info.get("policy_ms") if chunk_step == 0 else None,
                                "infer_server_ms": infer_info.get("server_ms") if chunk_step == 0 else None,
                                "a_base": base_chunk[chunk_step].tolist(),
                                "a_res_policy": residual_step_action.tolist(),
                                "a_res": delta_action.tolist(),
                                "a_final": final_action.tolist(),
                                "residual_scale": float(residual_scale_step),
                                "xi": float(xi_step),
                                "reward": float(reward),
                                "done": bool(done),
                                "success": bool(episode_success),
                            }
                        )

                        _append_tb_step_window(
                            step_metric_window,
                            reward=float(reward),
                            residual_scale=float(residual_scale_step),
                            xi=float(xi_step),
                            residual_action=residual_step_action,
                            delta_action=delta_action,
                            base_action=base_chunk[chunk_step],
                            final_action=final_action,
                            infer_info=infer_info,
                            replan_point=bool(chunk_step == 0),
                        )
                        if global_env_step % tb_step_period == 0:
                            _flush_tb_step_window(
                                tb_writer,
                                step_window=step_metric_window,
                                global_env_step=global_env_step,
                                control_indices=control_indices,
                                histogram=bool(global_env_step % tb_histogram_period == 0),
                            )

                        if done:
                            next_obs_input = _zero_obs_like(obs_input)
                            mask = 0.0
                        elif chunk_step < (chunk_horizon - 1):
                            next_obs_input = _build_residual_step_obs_profiled(
                                profiler,
                                next_obs_raw,
                                base_chunk[chunk_step + 1],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                normalizer=normalizer,
                                obs_cache=obs_cache,
                            )
                            mask = 1.0
                        else:
                            if next_chunk_future is not None:
                                next_openpi_chunk, next_infer_info = next_chunk_future.result()
                            else:
                                next_openpi_chunk, next_infer_info = openpi_client.infer_chunk(
                                    next_obs_raw,
                                    env.current_instruction,
                                    obs_cache=obs_cache,
                                )
                            next_base_chunk = select_action_chunk_window(
                                next_openpi_chunk,
                                horizon=chunk_horizon,
                            )
                            next_obs_input = _build_residual_step_obs_profiled(
                                profiler,
                                next_obs_raw,
                                next_base_chunk[0],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                normalizer=normalizer,
                                obs_cache=obs_cache,
                            )
                            cached_base_chunk = next_base_chunk
                            cached_infer_info = next_infer_info
                            mask = 1.0

                        transition_payload = {
                            "observations": _clone_obs_dict(obs_input),
                            "actions": residual_step_action.astype(np.float32),
                            "next_observations": _clone_obs_dict(next_obs_input),
                            "rewards": np.float32(reward),
                            "masks": np.float32(mask),
                            "dones": bool(done),
                        }
                        if async_learner is not None:
                            with async_learner.replay_lock:
                                replay_buffer.insert(transition_payload)
                        elif sync_replay_lock is not None:
                            with sync_replay_lock:
                                replay_buffer.insert(transition_payload)
                        else:
                            replay_buffer.insert(transition_payload)

                        if async_learner is None:
                            if (
                                phase_train
                                and len(replay_buffer) >= int(cfg.training.training_starts)
                                and global_policy_step % int(cfg.training.update_every) == 0
                            ):
                                for _ in range(int(cfg.training.updates_per_step)):
                                    if sync_replay_prefetcher is not None:
                                        sampled_batch = sync_replay_prefetcher.get(timeout=async_idle_sleep_sec)
                                        if sampled_batch is None:
                                            continue
                                        batch, online_bs, offline_bs = sampled_batch
                                    else:
                                        replay_sample_start = time.perf_counter()
                                        sampled = _sample_mixed_batch(
                                            replay_buffer,
                                            offline_buffer if offline_enabled else None,
                                            batch_size=int(cfg.replay.batch_size),
                                            offline_ratio=offline_ratio,
                                            symmetric_replay=symmetric_replay,
                                        )
                                        profiler.record_duration(
                                            "replay_sample",
                                            (time.perf_counter() - replay_sample_start) * 1000.0,
                                        )
                                        replay_prepare_start = time.perf_counter()
                                        prepared = _prepare_replay_batch(
                                            sampled,
                                            device=learner_agent.device,
                                            pin_memory=replay_prefetch_pin_memory,
                                            to_device=replay_prefetch_to_device,
                                            profiler=profiler,
                                            cuda_stream=None,
                                        )
                                        batch, online_bs, offline_bs = _consume_prepared_replay_batch(
                                            prepared,
                                            device=learner_agent.device,
                                            profiler=profiler,
                                        )
                                        profiler.record_duration(
                                            "replay_prepare",
                                            (time.perf_counter() - replay_prepare_start) * 1000.0,
                                        )
                                    update_start = time.perf_counter()
                                    learner_agent, last_update_info = learner_agent.update_high_utd(
                                        batch,
                                        utd_ratio=int(cfg.sac.utd_ratio),
                                    )
                                    profiler.record_duration(
                                        "agent_update_high_utd",
                                        (time.perf_counter() - update_start) * 1000.0,
                                    )
                                    last_update_info["online_batch_size"] = int(online_bs)
                                    last_update_info["offline_batch_size"] = int(offline_bs)
                                    last_update_info["offline_fraction"] = float(
                                        offline_bs / max(1, online_bs + offline_bs)
                                    )
                                agent = learner_agent
                        else:
                            last_update_info = async_learner.get_last_update_info()

                        if global_env_step % tb_step_period == 0 and last_update_info:
                            _log_update_metrics(tb_writer, last_update_info, global_env_step)
                            tb_writer.add_scalar(
                                "system/online_buffer_size",
                                int(len(replay_buffer)),
                                global_env_step,
                            )
                            if offline_buffer is not None:
                                tb_writer.add_scalar(
                                    "system/offline_buffer_size",
                                    int(len(offline_buffer)),
                                    global_env_step,
                                )
                            tb_writer.add_scalar(
                                "system/global_policy_step",
                                int(global_policy_step),
                                global_env_step,
                            )
                            if async_learner is not None:
                                tb_writer.add_scalar(
                                    "system/learner_update_steps",
                                    int(async_learner.get_update_steps()),
                                    global_env_step,
                                )
                                tb_writer.add_scalar(
                                    "system/replay_prefetch_queue_size",
                                    int(async_learner.get_prefetch_queue_size()),
                                    global_env_step,
                                )
                            elif sync_replay_prefetcher is not None:
                                tb_writer.add_scalar(
                                    "system/replay_prefetch_queue_size",
                                    int(sync_replay_prefetcher.get_queue_size()),
                                    global_env_step,
                                )

                        global_policy_step += 1

                        if (
                            profiling_enabled
                            and profiling_log_period_steps > 0
                            and global_env_step > 0
                            and (global_env_step - profiling_last_flush_step) >= profiling_log_period_steps
                        ):
                            _emit_profiling_snapshot(
                                profiler,
                                profile_logger=profiling_logger,
                                tb_writer=tb_writer,
                                logger=logger,
                                global_env_step=global_env_step,
                                global_policy_step=global_policy_step,
                                episode_id=episode_id,
                                learner_update_steps=(
                                    int(async_learner.get_update_steps()) if async_learner is not None else 0
                                ),
                                replay_prefetch_queue_size=(
                                    int(async_learner.get_prefetch_queue_size())
                                    if async_learner is not None
                                    else int(sync_replay_prefetcher.get_queue_size())
                                    if sync_replay_prefetcher is not None
                                    else 0
                                ),
                            )
                            profiling_last_flush_step = int(global_env_step)

                        if (
                            phase_train
                            and int(cfg.training.checkpoint_period) > 0
                            and global_policy_step % int(cfg.training.checkpoint_period) == 0
                        ):
                            if async_learner is not None:
                                async_learner.save_checkpoint(
                                    str(checkpoint_dir),
                                    step=global_policy_step,
                                    keep=int(cfg.training.keep_checkpoints),
                                )
                            else:
                                checkpoint_payload = _snapshot_agent_checkpoint_payload(
                                    learner_agent,
                                    step=global_policy_step,
                                )
                                if checkpoint_writer is not None:
                                    checkpoint_writer.submit(
                                        _CheckpointTask(
                                            checkpoint_dir=str(checkpoint_dir),
                                            payload=checkpoint_payload,
                                            step=int(global_policy_step),
                                            keep=int(cfg.training.keep_checkpoints),
                                        )
                                    )
                                else:
                                    _write_checkpoint_payload(
                                        profiler,
                                        str(checkpoint_dir),
                                        checkpoint_payload,
                                        step=global_policy_step,
                                        keep=int(cfg.training.keep_checkpoints),
                                    )

                        if done:
                            episode_done = True
                            break

                    obs_raw = next_obs_raw
                    if episode_done:
                        break

                _flush_tb_step_window(
                    tb_writer,
                    step_window=step_metric_window,
                    global_env_step=global_env_step,
                    control_indices=control_indices,
                    histogram=bool(global_env_step > 0 and global_env_step % tb_histogram_period == 0),
                )
                total_success += int(episode_success)
                recent_successes.append(int(episode_success))
                running_success_rate = float(total_success) / float(episode_id + 1)
                recent_success_rate = float(sum(recent_successes)) / float(len(recent_successes))

                episode_logger.write(
                    {
                        "episode_id": episode_id,
                        "phase": phase_name,
                        "seed": int(env.last_seed if env.last_seed is not None else seed),
                        "init_state_idx": (
                            int(env.current_init_state_idx)
                            if env.current_init_state_idx is not None
                            else None
                        ),
                        "success": bool(episode_success),
                        "episode_steps": int(episode_steps),
                        "episode_return": float(episode_return),
                        "global_env_step": int(global_env_step),
                        "global_policy_step": int(global_policy_step),
                        "running_success_rate": running_success_rate,
                        "recent_success_rate": recent_success_rate,
                    }
                )
                tb_writer.add_scalar("episode/success", int(episode_success), episode_id)
                tb_writer.add_scalar("episode/return", float(episode_return), episode_id)
                tb_writer.add_scalar("episode/length", int(episode_steps), episode_id)
                tb_writer.add_scalar("episode/running_success_rate", running_success_rate, episode_id)
                tb_writer.add_scalar("episode/recent_success_rate_20", recent_success_rate, episode_id)
                tb_writer.add_scalar("system/online_buffer_size", int(len(replay_buffer)), global_env_step)
                if offline_buffer is not None:
                    tb_writer.add_scalar("system/offline_buffer_size", int(len(offline_buffer)), global_env_step)
                tb_writer.add_scalar("system/global_policy_step", int(global_policy_step), global_env_step)
                if async_learner is not None:
                    tb_writer.add_scalar(
                        "system/learner_update_steps",
                        int(async_learner.get_update_steps()),
                        global_env_step,
                    )
                    tb_writer.add_scalar(
                        "system/replay_prefetch_queue_size",
                        int(async_learner.get_prefetch_queue_size()),
                        global_env_step,
                    )
                elif sync_replay_prefetcher is not None:
                    tb_writer.add_scalar(
                        "system/replay_prefetch_queue_size",
                        int(sync_replay_prefetcher.get_queue_size()),
                        global_env_step,
                    )

                logger.info(
                    "phase=%s episode=%s success=%s steps=%s return=%.2f success_rate=%.3f recent=%.3f",
                    phase_name,
                    episode_id,
                    episode_success,
                    episode_steps,
                    episode_return,
                    running_success_rate,
                    recent_success_rate,
                )
                if async_eval_proc is not None and (not async_eval_dead_reported):
                    proc_rc = async_eval_proc.poll()
                    if proc_rc is not None:
                        async_eval_dead_reported = True
                        logger.warning(
                            "Async eval watcher exited early with returncode=%s; "
                            "see %s for details",
                            proc_rc,
                            async_eval_log_path,
                        )
                _sync_async_eval_results_to_tb(
                    tb_writer,
                    summary_jsonl_path=async_eval_summary_path,
                    sync_state=async_eval_tb_sync_state,
                    logger=logger,
                )

                episode_id += 1
                phase_episode_count += 1

            if stopped_by_env_budget:
                break

        if async_learner is not None:
            async_learner.stop()
            last_update_info = async_learner.get_last_update_info()
        if checkpoint_writer is not None:
            checkpoint_writer.close(wait=True)
            checkpoint_writer = None

        final_profiling_payload = _emit_profiling_snapshot(
            profiler,
            profile_logger=profiling_logger,
            tb_writer=tb_writer,
            logger=logger,
            global_env_step=global_env_step,
            global_policy_step=global_policy_step,
            episode_id=episode_id,
            learner_update_steps=int(async_learner.get_update_steps()) if async_learner is not None else 0,
            replay_prefetch_queue_size=(
                int(async_learner.get_prefetch_queue_size())
                if async_learner is not None
                else int(sync_replay_prefetcher.get_queue_size())
                if sync_replay_prefetcher is not None
                else 0
            ),
        )

        summary = {
            "episodes": int(episode_id),
            "global_env_step": int(global_env_step),
            "global_policy_step": int(global_policy_step),
            "total_success": int(total_success),
            "success_rate": float(total_success / max(1, int(episode_id))),
            "skipped_seeds": int(skipped_seeds),
            "seed_start": int(cfg.task.seed_base),
            "seed_next": int(seed_cursor),
            "stopped_by_env_budget": bool(stopped_by_env_budget),
            "max_online_env_steps": int(max_online_env_steps),
            "replay_size": int(len(replay_buffer) if replay_buffer is not None else 0),
            "offline_enabled": bool(offline_enabled),
            "offline_ratio": float(offline_ratio),
            "offline_symmetric_replay": bool(symmetric_replay),
            "offline_buffer_size": int(len(offline_buffer) if offline_buffer is not None else 0),
            "offline_stats": offline_stats,
            "bootstrap_stats": bootstrap_stats,
            "critic_pretrain": _to_jsonable(warmstart_info),
            "checkpoint_dir": str(checkpoint_dir),
            "last_update_info": _to_jsonable(last_update_info),
            "async_enabled": bool(async_enabled),
            "async_update_frequency": int(async_update_frequency),
            "learner_update_steps": int(async_learner.get_update_steps() if async_learner is not None else 0),
            "replay_prefetch_enabled": bool(replay_prefetch_enabled),
            "replay_prefetch_queue_size": int(replay_prefetch_queue_size),
            "replay_prefetch_pin_memory": bool(replay_prefetch_pin_memory),
            "replay_prefetch_to_device": bool(replay_prefetch_to_device),
            "profiling": {
                "enabled": bool(profiling_enabled),
                "window_size": int(profiling_window_size),
                "log_period_steps": int(profiling_log_period_steps),
                "log_file": str(run_dir / profiling_log_file) if profiling_enabled else None,
                "snapshot": (
                    _to_jsonable(final_profiling_payload.get("metrics", {}))
                    if final_profiling_payload is not None
                    else {}
                ),
            },
            "async_eval": {
                "enabled": bool(cfg.training.get("async_eval", {}).get("enabled", False)),
                "watcher_started": bool(async_eval_proc is not None),
                "watcher_log_path": str(async_eval_log_path) if async_eval_log_path is not None else None,
                "summary_jsonl_path": (
                    str(async_eval_summary_path) if async_eval_summary_path is not None else None
                ),
                "watcher_return_code": (
                    int(async_eval_proc.returncode)
                    if async_eval_proc is not None and async_eval_proc.returncode is not None
                    else None
                ),
            },
        }
        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("training done: %s", summary)

    finally:
        if async_learner is not None:
            async_learner.stop()
        if checkpoint_writer is not None:
            checkpoint_writer.close(wait=True)
        if sync_replay_prefetcher is not None:
            sync_replay_prefetcher.stop()
        if openpi_prefetcher is not None:
            openpi_prefetcher.close()
        async_eval_watcher_return_code = _stop_async_eval_watcher(
            async_eval_proc,
            async_eval_log_fp,
            logger=logger,
        )
        if async_eval_proc is not None:
            logger.info(
                "Async eval watcher stopped (returncode=%s, log=%s)",
                async_eval_watcher_return_code,
                async_eval_log_path,
            )
        _sync_async_eval_results_to_tb(
            tb_writer,
            summary_jsonl_path=async_eval_summary_path,
            sync_state=async_eval_tb_sync_state,
            logger=logger,
        )
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        step_logger.close()
        episode_logger.close()
        if profiling_logger is not None:
            profiling_logger.close()
        tb_writer.close()


if __name__ == "__main__":
    main()
