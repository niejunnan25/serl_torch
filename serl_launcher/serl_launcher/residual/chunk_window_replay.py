from __future__ import annotations

"""Chunk-window replay helpers for residual training pipelines."""

from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.data.replay_buffer import _to_torch


class BatchPrefetcher:
    """Single-worker prefetcher for replay batches.

    The caller consumes one result at a time. Immediately after a result is
    delivered, the next sample is submitted so CPU replay work can overlap with
    learner compute on the current batch.
    """

    def __init__(
        self,
        sample_fn: Callable[[], Any],
        *,
        thread_name_prefix: str = "replay-prefetch",
    ):
        self._sample_fn = sample_fn
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=str(thread_name_prefix),
        )
        self._future: Future | None = None
        self._closed = False

    def start(self) -> None:
        if self._closed:
            return
        if self._future is None:
            self._future = self._executor.submit(self._sample_fn)

    def get(self) -> Any:
        self.start()
        future = self._future
        if future is None:
            raise RuntimeError("BatchPrefetcher is closed")
        self._future = None
        try:
            result = future.result()
        except Exception:
            self.close()
            raise
        self.start()
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        future = self._future
        self._future = None
        if future is not None:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)


class ProfileAccumulator:
    """Accumulate numeric profile fields and report per-record averages."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._records = 0

    def record(self, profile: dict[str, float]) -> None:
        if not profile:
            return
        self._records += 1
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                self._totals[str(key)] = float(self._totals.get(str(key), 0.0)) + float(
                    value
                )

    def drain(self) -> dict[str, float | int]:
        if not self._totals:
            return {}
        totals = dict(self._totals)
        records = max(int(self._records), 1)
        self._totals.clear()
        self._records = 0
        result: dict[str, float | int] = {"profile_records": int(records)}
        for key, total in sorted(totals.items()):
            if key.endswith("_calls"):
                result[key] = int(total)
            else:
                result[key] = float(total) / float(records)
        return result


class PrefetchingMixedBatchSampler:
    """Prefetch mixed online/offline replay batches and aggregate sample profile."""

    def __init__(
        self,
        *,
        online_replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore,
        offline_replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore | None,
        batch_size: int,
        device: Any = None,
        pack_obs_and_next_obs: bool = True,
        prefer_device_concat: bool = True,
        thread_name_prefix: str = "replay-prefetch",
    ) -> None:
        self._online_replay_buffer = online_replay_buffer
        self._offline_replay_buffer = offline_replay_buffer
        self._batch_size = int(batch_size)
        self._device = device
        self._pack_obs_and_next_obs = bool(pack_obs_and_next_obs)
        self._prefer_device_concat = bool(prefer_device_concat)
        self._thread_name_prefix = str(thread_name_prefix)
        self._prefetcher: BatchPrefetcher | None = None
        self._prefetcher_offline_ratio: float | None = None
        self._sample_profile_totals: dict[str, float] = {}

    def next_batch(
        self, *, offline_ratio: float
    ) -> tuple[dict[str, Any], dict[str, int]]:
        prefetcher = self._get_prefetcher(offline_ratio=float(offline_ratio))
        batch, batch_mix, sample_profile = prefetcher.get()
        self._record_sample_profile(sample_profile)
        return batch, batch_mix

    def drain_sample_profile(self) -> dict[str, float | int]:
        if not self._sample_profile_totals:
            return {}
        totals = dict(self._sample_profile_totals)
        self._sample_profile_totals.clear()
        mixed_calls = max(float(totals.get("mixed_sample_calls", 0.0)), 1.0)
        online_calls = max(float(totals.get("online_sample_calls", 0.0)), 1.0)
        offline_calls = max(float(totals.get("offline_sample_calls", 0.0)), 1.0)
        result: dict[str, float | int] = {}
        for key, total in sorted(totals.items()):
            if key.endswith("_calls"):
                result[key] = int(total)
                continue
            if key.startswith("online_"):
                denom = online_calls
            elif key.startswith("offline_"):
                denom = offline_calls
            else:
                denom = mixed_calls
            result[key] = float(total) / float(denom)
        return result

    def close(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None
            self._prefetcher_offline_ratio = None

    def _sample(
        self,
        *,
        offline_ratio: float,
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, float]]:
        sample_profile: dict[str, float] = {}
        batch, batch_mix = sample_mixed_training_batch(
            online_replay_buffer=self._online_replay_buffer,
            offline_replay_buffer=self._offline_replay_buffer,
            batch_size=int(self._batch_size),
            offline_ratio=float(offline_ratio),
            profile=sample_profile,
            pack_obs_and_next_obs=bool(self._pack_obs_and_next_obs),
            device=self._device,
            prefer_device_concat=bool(self._prefer_device_concat),
        )
        return batch, batch_mix, sample_profile

    def _get_prefetcher(self, *, offline_ratio: float) -> BatchPrefetcher:
        resolved_ratio = float(offline_ratio)
        if (
            self._prefetcher is None
            or self._prefetcher_offline_ratio is None
            or float(self._prefetcher_offline_ratio) != float(resolved_ratio)
        ):
            self.close()
            self._prefetcher = BatchPrefetcher(
                lambda: self._sample(offline_ratio=resolved_ratio),
                thread_name_prefix=self._thread_name_prefix,
            )
            self._prefetcher_offline_ratio = float(resolved_ratio)
            self._prefetcher.start()
        return self._prefetcher

    def _record_sample_profile(self, profile: dict[str, float]) -> None:
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                self._sample_profile_totals[str(key)] = float(
                    self._sample_profile_totals.get(str(key), 0.0)
                ) + float(value)


def _profile_add(profile: Optional[Dict[str, float]], key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(value)


def _profile_count(
    profile: Optional[Dict[str, float]],
    key: str,
    value: int = 1,
) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + int(value)


def _merge_prefixed_profile(
    profile: Optional[Dict[str, float]],
    *,
    prefix: str,
    child_profile: Dict[str, float],
) -> None:
    if profile is None:
        return
    for key, value in child_profile.items():
        if isinstance(value, (int, float)):
            _profile_add(profile, f"{prefix}_{key}", float(value))


def _sample_with_profile(
    replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore,
    *,
    batch_size: int,
    prefix: str,
    profile: Optional[Dict[str, float]],
    pack_obs_and_next_obs: bool,
    convert_to_torch: bool,
    device: Any,
) -> dict[str, Any]:
    child_profile: Dict[str, float] = {}
    sample_start = time.perf_counter()
    try:
        batch = replay_buffer.sample(
            int(batch_size),
            profile=child_profile,
            pack_obs_and_next_obs=bool(pack_obs_and_next_obs),
        )
    except TypeError as exc:
        if "profile" not in str(exc) and "pack_obs_and_next_obs" not in str(exc):
            raise
        child_profile = {}
        batch = replay_buffer.sample(int(batch_size))
    _profile_add(profile, f"{prefix}_sample_sec", time.perf_counter() - sample_start)
    _profile_count(profile, f"{prefix}_sample_calls")
    _profile_add(profile, f"{prefix}_batch_size", int(batch_size))
    _merge_prefixed_profile(profile, prefix=prefix, child_profile=child_profile)
    if convert_to_torch:
        to_torch_start = time.perf_counter()
        batch = _to_torch(batch, device=device)
        _profile_add(
            profile, f"{prefix}_to_torch_sec", time.perf_counter() - to_torch_start
        )
    return batch


def create_chunk_replay_buffer(
    *,
    observation_space: gym.spaces.Dict,
    action_dim: int,
    chunk_horizon: int,
    discount: float,
    image_keys: tuple[str, ...],
    capacity: int,
) -> MemoryEfficientStepWindowReplayBufferDataStore:
    """Create a replay buffer configured for chunked residual training."""

    return MemoryEfficientStepWindowReplayBufferDataStore(
        observation_space=observation_space,
        action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        ),
        capacity=int(capacity),
        window_size=int(chunk_horizon),
        discount=float(discount),
        sample_stride=1,
        require_full_window=False,
        image_keys=image_keys,
    )


def reshape_chunk_batch_for_training(batch: dict[str, Any]) -> dict[str, Any]:
    """Flatten chunk action tensors into the shape expected by learner updates."""

    batch_out = dict(batch)
    if "actions" in batch_out:
        actions = batch_out["actions"]
        batch_out["actions"] = actions.reshape(int(actions.shape[0]), -1)
    if "action_mask" in batch_out:
        action_mask = batch_out["action_mask"]
        batch_out["action_mask"] = action_mask.reshape(int(action_mask.shape[0]), -1)
    return batch_out


def concat_batch_trees(values: list[Any]) -> Any:
    """Concatenate nested batch trees along the batch axis."""

    first = values[0]
    if isinstance(first, dict):
        return {
            key: concat_batch_trees([value[key] for value in values]) for key in first
        }
    if any(isinstance(value, torch.Tensor) for value in values):
        tensor_values = []
        device = next(
            (value.device for value in values if isinstance(value, torch.Tensor)),
            None,
        )
        for value in values:
            if isinstance(value, torch.Tensor):
                tensor_values.append(value)
            else:
                tensor_values.append(torch.as_tensor(value, device=device))
        return torch.cat(tensor_values, dim=0)
    arrays = [np.asarray(value) for value in values]
    return np.concatenate(arrays, axis=0)


def sample_mixed_training_batch(
    *,
    online_replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore,
    offline_replay_buffer: MemoryEfficientStepWindowReplayBufferDataStore | None,
    batch_size: int,
    offline_ratio: float,
    reshape_batch: bool = True,
    profile: Optional[Dict[str, float]] = None,
    pack_obs_and_next_obs: bool = False,
    device: Any = None,
    prefer_device_concat: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Sample a mixed online/offline training batch for chunked residual updates."""

    mixed_start = time.perf_counter()
    _profile_count(profile, "mixed_sample_calls")
    offline_size = (
        0 if offline_replay_buffer is None else int(len(offline_replay_buffer))
    )
    online_size = int(len(online_replay_buffer))

    if offline_replay_buffer is None or offline_size <= 0 or offline_ratio <= 0.0:
        batch = _sample_with_profile(
            online_replay_buffer,
            batch_size=int(batch_size),
            prefix="online",
            profile=profile,
            pack_obs_and_next_obs=pack_obs_and_next_obs,
            convert_to_torch=bool(prefer_device_concat),
            device=device,
        )
        batch_mix = {
            "online_batch_size": int(batch_size),
            "offline_batch_size": 0,
        }
        if reshape_batch:
            reshape_start = time.perf_counter()
            batch = reshape_chunk_batch_for_training(batch)
            _profile_add(profile, "reshape_sec", time.perf_counter() - reshape_start)
        _profile_add(profile, "mixed_total_sec", time.perf_counter() - mixed_start)
        return batch, batch_mix
    if online_size <= 0:
        batch = _sample_with_profile(
            offline_replay_buffer,
            batch_size=int(batch_size),
            prefix="offline",
            profile=profile,
            pack_obs_and_next_obs=pack_obs_and_next_obs,
            convert_to_torch=bool(prefer_device_concat),
            device=device,
        )
        batch_mix = {
            "online_batch_size": 0,
            "offline_batch_size": int(batch_size),
        }
        if reshape_batch:
            reshape_start = time.perf_counter()
            batch = reshape_chunk_batch_for_training(batch)
            _profile_add(profile, "reshape_sec", time.perf_counter() - reshape_start)
        _profile_add(profile, "mixed_total_sec", time.perf_counter() - mixed_start)
        return batch, batch_mix

    offline_batch_size = int(round(float(batch_size) * float(offline_ratio)))
    offline_batch_size = min(int(batch_size), max(0, int(offline_batch_size)))
    online_batch_size = int(batch_size) - int(offline_batch_size)

    if offline_batch_size <= 0:
        batch = _sample_with_profile(
            online_replay_buffer,
            batch_size=int(batch_size),
            prefix="online",
            profile=profile,
            pack_obs_and_next_obs=pack_obs_and_next_obs,
            convert_to_torch=bool(prefer_device_concat),
            device=device,
        )
        batch_mix = {
            "online_batch_size": int(batch_size),
            "offline_batch_size": 0,
        }
        if reshape_batch:
            reshape_start = time.perf_counter()
            batch = reshape_chunk_batch_for_training(batch)
            _profile_add(profile, "reshape_sec", time.perf_counter() - reshape_start)
        _profile_add(profile, "mixed_total_sec", time.perf_counter() - mixed_start)
        return batch, batch_mix
    if online_batch_size <= 0:
        batch = _sample_with_profile(
            offline_replay_buffer,
            batch_size=int(batch_size),
            prefix="offline",
            profile=profile,
            pack_obs_and_next_obs=pack_obs_and_next_obs,
            convert_to_torch=bool(prefer_device_concat),
            device=device,
        )
        batch_mix = {
            "online_batch_size": 0,
            "offline_batch_size": int(batch_size),
        }
        if reshape_batch:
            reshape_start = time.perf_counter()
            batch = reshape_chunk_batch_for_training(batch)
            _profile_add(profile, "reshape_sec", time.perf_counter() - reshape_start)
        _profile_add(profile, "mixed_total_sec", time.perf_counter() - mixed_start)
        return batch, batch_mix

    online_batch = _sample_with_profile(
        online_replay_buffer,
        batch_size=int(online_batch_size),
        prefix="online",
        profile=profile,
        pack_obs_and_next_obs=pack_obs_and_next_obs,
        convert_to_torch=bool(prefer_device_concat),
        device=device,
    )
    offline_batch = _sample_with_profile(
        offline_replay_buffer,
        batch_size=int(offline_batch_size),
        prefix="offline",
        profile=profile,
        pack_obs_and_next_obs=pack_obs_and_next_obs,
        convert_to_torch=bool(prefer_device_concat),
        device=device,
    )
    concat_start = time.perf_counter()
    batch = concat_batch_trees([online_batch, offline_batch])
    _profile_add(profile, "concat_sec", time.perf_counter() - concat_start)
    batch_mix = {
        "online_batch_size": int(online_batch_size),
        "offline_batch_size": int(offline_batch_size),
    }
    if reshape_batch:
        reshape_start = time.perf_counter()
        batch = reshape_chunk_batch_for_training(batch)
        _profile_add(profile, "reshape_sec", time.perf_counter() - reshape_start)
    _profile_add(profile, "mixed_total_sec", time.perf_counter() - mixed_start)
    return batch, batch_mix
