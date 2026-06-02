from threading import Lock
import time
from typing import Iterable, Mapping, Optional, TypeVar

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym
import numpy as np

from serl_launcher.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer
from serl_launcher.data.batch_ops import is_packed_batch
from serl_launcher.data.batch_ops import pack_transition_batch
from serl_launcher.data.batch_ops import packed_batch_size
from serl_launcher.data.batch_ops import packed_batch_slice
from serl_launcher.data.batch_ops import packed_batch_take
from serl_launcher.data.batch_ops import ring_write_batch
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.step_window_replay_buffer import (
    MemoryEfficientStepWindowReplayBuffer,
)
from serl_launcher.data.step_window_replay_buffer import StepWindowReplayBuffer

from agentlace.data.data_store import DataStoreBase

try:
    from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
except ImportError:
    print(
        "rlds logger is not installed, install it if required: "
        "https://github.com/rail-berkeley/oxe_envlogger "
    )
    RLDSLogger = TypeVar("RLDSLogger")


def _normalize_batch_data(batch_data):
    if isinstance(batch_data, list):
        if not batch_data:
            return []
        return pack_transition_batch(batch_data)
    if is_packed_batch(batch_data):
        return batch_data
    raise TypeError(
        "batch_insert expects a list[transition] or a packed nested batch dict"
    )


def _profile_add(profile: Optional[dict], key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(value)


def _profile_set(profile: Optional[dict], key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(value)


def _batch_storage_view(packed_batch, keys: Iterable[str]):
    return {key: packed_batch[key] for key in keys}


def _advance_step_type(current_step_type, transition):
    if current_step_type in {RLDSStepType.TERMINATION, RLDSStepType.TRUNCATION}:
        return RLDSStepType.RESTART
    if not transition["masks"]:
        return RLDSStepType.TERMINATION
    if transition["dones"]:
        return RLDSStepType.TRUNCATION
    return RLDSStepType.TRANSITION


def _log_transition_batch(logger, step_type, packed_batch):
    if logger is None:
        return step_type
    batch_count = int(packed_batch_size(packed_batch))
    current_step_type = step_type
    for batch_index in range(batch_count):
        transition = packed_batch_take(packed_batch, batch_index)
        current_step_type = _advance_step_type(current_step_type, transition)
        logger(
            action=transition["actions"],
            obs=transition["next_observations"],
            reward=transition["rewards"],
            step_type=current_step_type,
        )
    return current_step_type


def _batch_write_plan(*, insert_index: int, capacity: int, total_count: int) -> tuple[int, int]:
    keep_count = min(int(total_count), int(capacity))
    write_start = (
        int(insert_index)
        if int(total_count) <= int(capacity)
        else int((int(insert_index) + int(total_count) - keep_count) % int(capacity))
    )
    return int(write_start), int(keep_count)


def _refresh_candidate_windows_range(
    replay: StepWindowReplayBuffer,
    *,
    first_step_id: int,
    last_step_id: int,
) -> None:
    replay._cleanup_stale_candidates()
    start_step_id = max(
        replay._min_active_step_id(),
        int(first_step_id) - int(replay.window_size),
    )
    for step_id in range(int(start_step_id), int(last_step_id) + 1):
        if step_id in replay._candidate_start_step_set:
            continue
        if replay._transition_ready(step_id):
            replay._candidate_start_step_ids.append(int(step_id))
            replay._candidate_start_step_set.add(int(step_id))


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        rlds_logger: Optional[RLDSLogger] = None,
        extra_fields: Optional[Mapping[str, gym.Space]] = None,
    ):
        ReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            extra_fields=extra_fields,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    pack_transition_batch([data]),
                )

    def batch_insert(self, batch_data):
        packed_batch = _normalize_batch_data(batch_data)
        total_count = int(packed_batch_size(packed_batch))
        if total_count <= 0:
            return
        with self._lock:
            write_start, keep_count = _batch_write_plan(
                insert_index=int(self._insert_index),
                capacity=int(self._capacity),
                total_count=int(total_count),
            )
            packed_kept = (
                packed_batch
                if keep_count == total_count
                else packed_batch_slice(
                    packed_batch,
                    int(total_count - keep_count),
                    int(total_count),
                )
            )
            ring_write_batch(
                self.dataset_dict,
                packed_kept,
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            self._insert_index = (int(self._insert_index) + int(total_count)) % int(
                self._capacity
            )
            self._size = min(int(self._size) + int(total_count), int(self._capacity))
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    packed_batch,
                )

    def sample(self, *args, **kwargs):
        profile = kwargs.pop("profile", None)
        lock_wait_start = time.perf_counter()
        self._lock.acquire()
        _profile_add(profile, "lock_wait_sec", time.perf_counter() - lock_wait_start)
        lock_hold_start = time.perf_counter()
        try:
            return super().sample(*args, **kwargs)
        finally:
            _profile_add(
                profile,
                "lock_hold_sec",
                time.perf_counter() - lock_hold_start,
            )
            self._lock.release()

    def latest_data_id(self):
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            pixel_keys=image_keys,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    pack_transition_batch([data]),
                )

    def batch_insert(self, batch_data):
        packed_batch = _normalize_batch_data(batch_data)
        batch_count = int(packed_batch_size(packed_batch))
        if batch_count <= 0:
            return
        with self._lock:
            for batch_index in range(batch_count):
                super().insert(packed_batch_take(packed_batch, batch_index))
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    packed_batch,
                )

    def sample(self, *args, **kwargs):
        profile = kwargs.pop("profile", None)
        lock_wait_start = time.perf_counter()
        self._lock.acquire()
        _profile_add(profile, "lock_wait_sec", time.perf_counter() - lock_wait_start)
        lock_hold_start = time.perf_counter()
        try:
            return super().sample(*args, **kwargs)
        finally:
            _profile_add(
                profile,
                "lock_hold_sec",
                time.perf_counter() - lock_hold_start,
            )
            self._lock.release()

    def latest_data_id(self):
        return self._insert_index

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class StepWindowReplayBufferDataStore(StepWindowReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        window_size: int,
        discount: float,
        sample_stride: int = 1,
        require_full_window: bool = False,
        next_observation_space: Optional[gym.Space] = None,
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        StepWindowReplayBuffer.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            window_size=window_size,
            discount=discount,
            sample_stride=sample_stride,
            require_full_window=require_full_window,
            next_observation_space=next_observation_space,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    pack_transition_batch([data]),
                )

    def batch_insert(self, batch_data):
        packed_batch = _normalize_batch_data(batch_data)
        total_count = int(packed_batch_size(packed_batch))
        if total_count <= 0:
            return
        if "episode_id" not in packed_batch or "episode_step" not in packed_batch:
            raise KeyError(
                "step window replay batch_insert requires 'episode_id' and 'episode_step'"
            )
        with self._lock:
            first_step_id = int(self._insert_count)
            last_step_id = int(first_step_id + total_count - 1)
            write_start, keep_count = _batch_write_plan(
                insert_index=int(self._insert_index),
                capacity=int(self._capacity),
                total_count=int(total_count),
            )
            packed_kept = (
                packed_batch
                if keep_count == total_count
                else packed_batch_slice(
                    packed_batch,
                    int(total_count - keep_count),
                    int(total_count),
                )
            )
            record_batch = _batch_storage_view(packed_kept, self.dataset_dict.keys())
            ring_write_batch(
                self.dataset_dict,
                record_batch,
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            keep_offset = int(total_count - keep_count)
            step_ids = np.arange(
                int(first_step_id + keep_offset),
                int(first_step_id + total_count),
                dtype=np.int64,
            )
            ring_write_batch(
                self._episode_ids,
                np.asarray(packed_kept["episode_id"], dtype=np.int64),
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            ring_write_batch(
                self._episode_steps,
                np.asarray(packed_kept["episode_step"], dtype=np.int32),
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            ring_write_batch(
                self._step_ids,
                step_ids,
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )

            self._insert_index = (int(self._insert_index) + int(total_count)) % int(
                self._capacity
            )
            self._insert_count += int(total_count)
            self._size = min(int(self._size) + int(total_count), int(self._capacity))
            _refresh_candidate_windows_range(
                self,
                first_step_id=int(first_step_id),
                last_step_id=int(last_step_id),
            )
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    packed_batch,
                )

    def sample(self, *args, **kwargs):
        profile = kwargs.get("profile")
        lock_wait_start = time.perf_counter()
        self._lock.acquire()
        _profile_add(profile, "lock_wait_sec", time.perf_counter() - lock_wait_start)
        lock_hold_start = time.perf_counter()
        try:
            return super().sample(*args, **kwargs)
        finally:
            _profile_add(
                profile,
                "lock_hold_sec",
                time.perf_counter() - lock_hold_start,
            )
            self._lock.release()

    def latest_data_id(self):
        return self._insert_count

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


class MemoryEfficientStepWindowReplayBufferDataStore(
    MemoryEfficientStepWindowReplayBuffer,
    DataStoreBase,
):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        window_size: int,
        discount: float,
        sample_stride: int = 1,
        require_full_window: bool = False,
        next_observation_space: Optional[gym.Space] = None,
        image_keys: Iterable[str] = ("image",),
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        MemoryEfficientStepWindowReplayBuffer.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            window_size=window_size,
            discount=discount,
            sample_stride=sample_stride,
            require_full_window=require_full_window,
            next_observation_space=next_observation_space,
            pixel_keys=tuple(image_keys),
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION
            self._logger = rlds_logger

    def insert(self, data):
        with self._lock:
            super().insert(data)
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    pack_transition_batch([data]),
                )

    def batch_insert(self, batch_data):
        packed_batch = _normalize_batch_data(batch_data)
        total_count = int(packed_batch_size(packed_batch))
        if total_count <= 0:
            return
        if "episode_id" not in packed_batch or "episode_step" not in packed_batch:
            raise KeyError(
                "memory-efficient step window replay batch_insert requires "
                "'episode_id' and 'episode_step'"
            )
        with self._lock:
            first_step_id = int(self._insert_count)
            last_step_id = int(first_step_id + total_count - 1)
            write_start, keep_count = _batch_write_plan(
                insert_index=int(self._insert_index),
                capacity=int(self._capacity),
                total_count=int(total_count),
            )
            packed_kept = (
                packed_batch
                if keep_count == total_count
                else packed_batch_slice(
                    packed_batch,
                    int(total_count - keep_count),
                    int(total_count),
                )
            )
            record_batch = {
                key: value
                for key, value in packed_kept.items()
                if key in self.dataset_dict
            }
            record_batch["next_observations"] = {
                key: value
                for key, value in packed_kept["next_observations"].items()
                if key not in self.pixel_keys
            }
            ring_write_batch(
                self.dataset_dict,
                record_batch,
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            keep_offset = int(total_count - keep_count)
            step_ids = np.arange(
                int(first_step_id + keep_offset),
                int(first_step_id + total_count),
                dtype=np.int64,
            )
            ring_write_batch(
                self._episode_ids,
                np.asarray(packed_kept["episode_id"], dtype=np.int64),
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            ring_write_batch(
                self._episode_steps,
                np.asarray(packed_kept["episode_step"], dtype=np.int32),
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            ring_write_batch(
                self._step_ids,
                step_ids,
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )
            for pixel_key in self.pixel_keys:
                ring_write_batch(
                    self._explicit_next_pixels[pixel_key],
                    np.asarray(
                        packed_kept["next_observations"][pixel_key],
                        dtype=self._pixel_spaces[pixel_key].dtype,
                    ),
                    insert_index=int(write_start),
                    capacity=int(self._capacity),
                )
            ring_write_batch(
                self._has_explicit_next_pixels,
                np.ones((int(keep_count),), dtype=bool),
                insert_index=int(write_start),
                capacity=int(self._capacity),
            )

            self._insert_index = (int(self._insert_index) + int(total_count)) % int(
                self._capacity
            )
            self._insert_count += int(total_count)
            self._size = min(int(self._size) + int(total_count), int(self._capacity))

            for step_id in range(int(first_step_id), int(last_step_id) + 1):
                if not self._is_active_step_id(step_id):
                    continue
                previous_step_id = int(step_id - 1)
                if not self._is_active_step_id(previous_step_id):
                    continue
                current_index = self._buffer_index(step_id)
                previous_index = self._buffer_index(previous_step_id)
                if (
                    int(self._episode_ids[previous_index])
                    == int(self._episode_ids[current_index])
                    and int(self._episode_steps[previous_index]) + 1
                    == int(self._episode_steps[current_index])
                ):
                    self._has_explicit_next_pixels[previous_index] = False

            _refresh_candidate_windows_range(
                self,
                first_step_id=int(first_step_id),
                last_step_id=int(last_step_id),
            )
            if self._logger:
                self.step_type = _log_transition_batch(
                    self._logger,
                    self.step_type,
                    packed_batch,
                )

    def sample(self, *args, **kwargs):
        if len(args) > 3:
            raise TypeError(
                "sample accepts at most batch_size, keys, and indx as positional args"
            )
        batch_size = args[0] if len(args) >= 1 else kwargs.pop("batch_size")
        keys = args[1] if len(args) >= 2 else kwargs.pop("keys", None)
        indx = args[2] if len(args) >= 3 else kwargs.pop("indx", None)
        profile = kwargs.pop("profile", None)
        pack_obs_and_next_obs = bool(kwargs.pop("pack_obs_and_next_obs", False))
        max_retries = int(kwargs.pop("max_retries", 1))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected sample keyword argument(s): {unexpected}")

        sample_start = time.perf_counter()
        last_error: Exception | None = None
        for retry_count in range(max(0, max_retries) + 1):
            lock_wait_start = time.perf_counter()
            self._lock.acquire()
            _profile_add(profile, "lock_wait_sec", time.perf_counter() - lock_wait_start)
            lock_hold_start = time.perf_counter()
            try:
                metadata = self._sample_window_metadata(
                    batch_size=int(batch_size),
                    indx=indx,
                    profile=profile,
                )
            finally:
                _profile_add(
                    profile,
                    "lock_hold_sec",
                    time.perf_counter() - lock_hold_start,
                )
                self._lock.release()

            transition_build_start = time.perf_counter()
            try:
                batch = self._build_transition_batch_from_metadata(
                    metadata,
                    pack_obs_and_next_obs=pack_obs_and_next_obs,
                    profile=profile,
                )
            except RuntimeError as exc:
                last_error = exc
                _profile_add(
                    profile,
                    "transition_build_sec",
                    time.perf_counter() - transition_build_start,
                )
                continue
            _profile_add(
                profile,
                "transition_build_sec",
                time.perf_counter() - transition_build_start,
            )

            validate_wait_start = time.perf_counter()
            self._lock.acquire()
            _profile_add(
                profile,
                "lock_validate_wait_sec",
                time.perf_counter() - validate_wait_start,
            )
            validate_hold_start = time.perf_counter()
            try:
                is_valid = self._validate_window_metadata(metadata)
            finally:
                _profile_add(
                    profile,
                    "lock_validate_hold_sec",
                    time.perf_counter() - validate_hold_start,
                )
                self._lock.release()

            if not is_valid:
                continue

            _profile_set(profile, "retry_count", float(retry_count))
            _profile_set(profile, "batch_size", int(batch_size))
            _profile_add(profile, "stack_sec", 0.0)
            if keys is not None:
                select_start = time.perf_counter()
                batch = {key: batch[key] for key in list(keys)}
                _profile_add(profile, "select_keys_sec", time.perf_counter() - select_start)
            _profile_add(profile, "sample_total_sec", time.perf_counter() - sample_start)
            return batch

        lock_wait_start = time.perf_counter()
        self._lock.acquire()
        _profile_add(profile, "lock_wait_sec", time.perf_counter() - lock_wait_start)
        lock_hold_start = time.perf_counter()
        try:
            batch = MemoryEfficientStepWindowReplayBuffer.sample(
                self,
                int(batch_size),
                keys=keys,
                indx=indx,
                profile=profile,
                pack_obs_and_next_obs=pack_obs_and_next_obs,
            )
        finally:
            _profile_add(
                profile,
                "lock_hold_sec",
                time.perf_counter() - lock_hold_start,
            )
            self._lock.release()
        _profile_set(profile, "retry_count", float(max_retries + 1))
        if last_error is not None:
            _profile_set(profile, "retry_fallback_after_error", 1.0)
        return batch

    def latest_data_id(self):
        return self._insert_count

    def get_latest_data(self, from_id: int):
        raise NotImplementedError


def populate_data_store(data_store: DataStoreBase, demos_path: str):
    import pickle as pkl

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store


def populate_data_store_with_z_axis_only(data_store: DataStoreBase, demos_path: str):
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
