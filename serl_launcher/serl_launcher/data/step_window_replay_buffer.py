from __future__ import annotations

import collections
import copy
import time
from typing import Iterable, Mapping, Optional

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym
import numpy as np

from serl_launcher.data.dataset import Dataset, DatasetDict
from serl_launcher.data.replay_buffer import _init_extra_field_dict
from serl_launcher.data.replay_buffer import _init_replay_dict
from serl_launcher.data.replay_buffer import _insert_recursively
from serl_launcher.data.replay_buffer import _to_torch


def _profile_add(profile: Optional[dict], key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(value)


def _profile_set(profile: Optional[dict], key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(value)


def _copy_at(dataset_dict, index: int):
    if isinstance(dataset_dict, np.ndarray):
        return np.array(dataset_dict[index], copy=True)
    if isinstance(dataset_dict, dict):
        return {k: _copy_at(v, index) for k, v in dataset_dict.items()}
    raise TypeError("Unsupported dataset type")


def _stack_nested(items):
    first = items[0]
    if isinstance(first, dict):
        return {k: _stack_nested([item[k] for item in items]) for k in first}
    return np.stack(items, axis=0)


def _take_at(dataset_dict, indices: np.ndarray):
    if isinstance(dataset_dict, np.ndarray):
        return np.array(dataset_dict[indices], copy=True)
    if isinstance(dataset_dict, dict):
        return {k: _take_at(v, indices) for k, v in dataset_dict.items()}
    raise TypeError("Unsupported dataset type")


def _zero_for_space(space: gym.Space) -> np.ndarray:
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"extra replay fields must use Box spaces, got {type(space)}")
    return np.zeros(tuple(space.shape), dtype=space.dtype)


def _pack_obs_and_next_pixels(
    obs_pixels: np.ndarray,
    next_pixels: np.ndarray,
) -> np.ndarray:
    obs_pixels = np.asarray(obs_pixels)
    next_pixels = np.asarray(next_pixels)
    if (
        obs_pixels.ndim >= 5
        and next_pixels.ndim == obs_pixels.ndim
        and obs_pixels.shape[0] == next_pixels.shape[0]
        and obs_pixels.shape[2:] == next_pixels.shape[2:]
    ):
        if int(obs_pixels.shape[1]) != 1:
            raise ValueError(
                "pack_obs_and_next_obs for step-window pixels requires a single "
                "pixel frame per observation so _unpack can recover the terminal "
                "next observation without extra metadata"
            )
        return np.concatenate([obs_pixels, next_pixels[:, -1:, ...]], axis=1)
    return np.stack([obs_pixels, next_pixels], axis=1)


class _CandidateStartStepIds:
    """Append-only candidate ids with cheap stale-prefix cleanup and sampling."""

    def __init__(self):
        self._values: list[int] = []
        self._head = 0

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return int(len(self._values) - self._head)

    def __iter__(self):
        return iter(self._values[self._head :])

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [
                self._values[self._head + offset]
                for offset in range(start, stop, step)
            ]
        if int(index) < 0:
            index = len(self) + int(index)
        if int(index) < 0 or int(index) >= len(self):
            raise IndexError(index)
        return self._values[self._head + int(index)]

    def __array__(self, dtype=None):
        return np.asarray(list(self), dtype=dtype)

    def append(self, value: int) -> None:
        self._values.append(int(value))

    def extend(self, values) -> None:
        self._values.extend(int(value) for value in values)

    def clear(self) -> None:
        self._values.clear()
        self._head = 0

    def popleft(self) -> int:
        if not self:
            raise IndexError("pop from an empty candidate list")
        value = int(self._values[self._head])
        self._head += 1
        self._compact_if_needed()
        return value

    def sample(self, np_random, batch_size: int) -> np.ndarray:
        count = len(self)
        if count <= 0:
            raise RuntimeError("Cannot sample from an empty candidate list")
        if hasattr(np_random, "integers"):
            offsets = np_random.integers(count, size=int(batch_size))
        else:
            offsets = np_random.randint(count, size=int(batch_size))
        return np.fromiter(
            (self._values[self._head + int(offset)] for offset in offsets),
            dtype=np.int64,
            count=int(batch_size),
        )

    def _compact_if_needed(self) -> None:
        if self._head <= 4096 or self._head * 2 <= len(self._values):
            return
        del self._values[: self._head]
        self._head = 0


class StepWindowReplayBuffer(Dataset):
    """Store step transitions and sample contiguous step windows."""

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
        extra_fields: Optional[Mapping[str, gym.Space]] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        if int(capacity) <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if int(window_size) <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if int(sample_stride) <= 0:
            raise ValueError(f"sample_stride must be positive, got {sample_stride}")
        if not np.isfinite(float(discount)) or float(discount) < 0.0:
            raise ValueError(f"discount must be finite and >= 0.0, got {discount}")

        observation_data = _init_replay_dict(observation_space, int(capacity))
        next_observation_data = _init_replay_dict(
            next_observation_space, int(capacity)
        )
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty(
                (int(capacity), *action_space.shape), dtype=action_space.dtype
            ),
            rewards=np.empty((int(capacity),), dtype=np.float32),
            masks=np.empty((int(capacity),), dtype=np.float32),
            dones=np.empty((int(capacity),), dtype=bool),
        )
        resolved_extra_fields = dict(extra_fields or {})
        dataset_dict.update(_init_extra_field_dict(resolved_extra_fields, int(capacity)))
        super().__init__(dataset_dict)

        self._extra_field_keys = tuple(str(key) for key in resolved_extra_fields)
        self._extra_field_defaults = {
            str(key): _zero_for_space(space)
            for key, space in resolved_extra_fields.items()
        }

        self._capacity = int(capacity)
        self.window_size = int(window_size)
        self.discount = float(discount)
        self.sample_stride = int(sample_stride)
        self.require_full_window = bool(require_full_window)
        self._step_action_shape = tuple(action_space.shape)
        self._window_action_shape = (self.window_size, *self._step_action_shape)

        self._episode_ids = np.empty((self._capacity,), dtype=np.int64)
        self._episode_steps = np.empty((self._capacity,), dtype=np.int32)
        self._step_ids = np.full((self._capacity,), -1, dtype=np.int64)

        self._size = 0
        self._insert_index = 0
        self._insert_count = 0
        self._candidate_start_step_ids = _CandidateStartStepIds()
        self._candidate_start_step_set: set[int] = set()

    def __len__(self) -> int:
        return int(self._size)

    @property
    def num_steps(self) -> int:
        return int(self._size)

    @property
    def num_windows(self) -> int:
        self._cleanup_stale_candidates()
        return int(len(self._candidate_start_step_ids))

    @property
    def window_action_shape(self) -> tuple[int, ...]:
        return self._window_action_shape

    def _min_active_step_id(self) -> int:
        return int(max(0, self._insert_count - self._size))

    def _is_active_step_id(self, step_id: int) -> bool:
        step_id = int(step_id)
        if step_id < self._min_active_step_id() or step_id >= int(self._insert_count):
            return False
        idx = int(step_id % self._capacity)
        return int(self._step_ids[idx]) == step_id

    def _buffer_index(self, step_id: int) -> int:
        if not self._is_active_step_id(step_id):
            raise KeyError(f"Inactive step_id={step_id}")
        return int(step_id % self._capacity)

    def _cleanup_stale_candidates(self) -> None:
        min_active = self._min_active_step_id()
        while (
            self._candidate_start_step_ids
            and self._candidate_start_step_ids[0] < min_active
        ):
            stale = self._candidate_start_step_ids.popleft()
            self._candidate_start_step_set.discard(int(stale))

    def _collect_segment(self, start_step_id: int) -> tuple[list[int], bool, bool]:
        if not self._is_active_step_id(start_step_id):
            return [], False, False
        start_idx = self._buffer_index(start_step_id)
        episode_id = int(self._episode_ids[start_idx])
        episode_step = int(self._episode_steps[start_idx])
        collected: list[int] = []
        boundary = False
        for offset in range(self.window_size):
            step_id = int(start_step_id + offset)
            if not self._is_active_step_id(step_id):
                break
            idx = self._buffer_index(step_id)
            if int(self._episode_ids[idx]) != episode_id:
                break
            if int(self._episode_steps[idx]) != episode_step + offset:
                break
            collected.append(step_id)
            if bool(self.dataset_dict["dones"][idx]):
                boundary = True
                break
        return collected, boundary, bool(len(collected) == self.window_size)

    def _transition_ready(self, start_step_id: int) -> bool:
        if not self._is_active_step_id(start_step_id):
            return False
        start_idx = self._buffer_index(start_step_id)
        if int(self._episode_steps[start_idx]) % self.sample_stride != 0:
            return False

        step_ids, boundary, full = self._collect_segment(start_step_id)
        if not step_ids:
            return False
        if boundary:
            return (not self.require_full_window) or (len(step_ids) == self.window_size)
        return full

    def _refresh_candidate_window(self, last_inserted_step_id: int) -> None:
        self._cleanup_stale_candidates()
        start_step = max(
            self._min_active_step_id(),
            int(last_inserted_step_id) - int(self.window_size),
        )
        for step_id in range(int(start_step), int(last_inserted_step_id) + 1):
            if step_id in self._candidate_start_step_set:
                continue
            if self._transition_ready(step_id):
                self._candidate_start_step_ids.append(int(step_id))
                self._candidate_start_step_set.add(int(step_id))

    def _step_record_from_data(self, data_dict: DatasetDict) -> DatasetDict:
        step_record = {}
        for key in self.dataset_dict.keys():
            if key in data_dict:
                step_record[key] = data_dict[key]
            elif key in self._extra_field_defaults:
                step_record[key] = self._extra_field_defaults[key]
            else:
                raise KeyError(f"step window replay insert requires {key!r}")
        return step_record

    def _copy_extra_fields_batch(self, start_indices: np.ndarray) -> DatasetDict:
        return {
            key: _take_at(self.dataset_dict[key], start_indices)
            for key in self._extra_field_keys
        }

    def insert(self, data_dict: DatasetDict):
        if "episode_id" not in data_dict:
            raise KeyError("step window replay insert requires 'episode_id'")
        if "episode_step" not in data_dict:
            raise KeyError("step window replay insert requires 'episode_step'")

        step_record = self._step_record_from_data(data_dict)
        _insert_recursively(self.dataset_dict, step_record, self._insert_index)

        self._episode_ids[self._insert_index] = np.int64(data_dict["episode_id"])
        self._episode_steps[self._insert_index] = np.int32(data_dict["episode_step"])
        self._step_ids[self._insert_index] = np.int64(self._insert_count)

        current_step_id = int(self._insert_count)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._insert_count += 1
        self._size = min(self._size + 1, self._capacity)

        self._refresh_candidate_window(current_step_id)

    def _build_transition(self, start_step_id: int) -> DatasetDict:
        step_ids, boundary, _ = self._collect_segment(start_step_id)
        if not step_ids:
            raise RuntimeError(
                f"No window transition available for start_step_id={start_step_id}"
            )

        window_steps = int(len(step_ids))
        step_actions = self.dataset_dict["actions"]
        action_window = np.zeros(self._window_action_shape, dtype=step_actions.dtype)
        action_mask = np.zeros(self._window_action_shape, dtype=np.float32)
        discounted_reward = 0.0

        for offset, step_id in enumerate(step_ids):
            idx = self._buffer_index(step_id)
            action_window[offset] = np.array(step_actions[idx], copy=True)
            action_mask[offset] = np.ones(
                self._step_action_shape,
                dtype=np.float32,
            )
            discounted_reward += (self.discount**offset) * float(
                self.dataset_dict["rewards"][idx]
            )

        start_idx = self._buffer_index(start_step_id)
        last_idx = self._buffer_index(step_ids[-1])
        last_mask = float(self.dataset_dict["masks"][last_idx])
        transition = {
            "observations": _copy_at(self.dataset_dict["observations"], start_idx),
            "actions": action_window,
            "action_mask": action_mask,
            "next_observations": _copy_at(
                self.dataset_dict["next_observations"], last_idx
            ),
            "rewards": np.float32(discounted_reward),
            "masks": np.float32(
                (self.discount ** max(0, window_steps - 1)) * last_mask
            ),
            "dones": bool(boundary),
            "window_steps": np.int32(window_steps),
        }
        for key in self._extra_field_keys:
            transition[key] = _copy_at(self.dataset_dict[key], start_idx)
        return transition

    def _batch_window_indices(
        self,
        sampled_start_ids: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        sampled_start_ids = np.asarray(sampled_start_ids, dtype=np.int64).reshape(-1)
        offsets = np.arange(int(self.window_size), dtype=np.int64)
        step_ids = sampled_start_ids[:, None] + offsets[None, :]
        indices = np.mod(step_ids, int(self._capacity)).astype(np.int64, copy=False)

        start_indices = np.mod(sampled_start_ids, int(self._capacity)).astype(
            np.int64,
            copy=False,
        )
        start_episode_ids = self._episode_ids[start_indices]
        start_episode_steps = self._episode_steps[start_indices]

        min_active = int(self._min_active_step_id())
        valid = (step_ids >= min_active) & (step_ids < int(self._insert_count))
        valid &= self._step_ids[indices] == step_ids
        valid &= self._episode_ids[indices] == start_episode_ids[:, None]
        valid &= self._episode_steps[indices] == (
            start_episode_steps[:, None] + offsets[None, :]
        )

        done_flags = valid & self.dataset_dict["dones"][indices]
        has_boundary = np.any(done_flags, axis=1)
        first_done_offsets = np.argmax(done_flags, axis=1)
        before_or_at_boundary = offsets[None, :] <= first_done_offsets[:, None]
        valid &= ~has_boundary[:, None] | before_or_at_boundary

        window_steps = np.sum(valid, axis=1).astype(np.int32)
        if np.any(window_steps <= 0):
            bad_index = int(np.nonzero(window_steps <= 0)[0][0])
            bad_id = int(sampled_start_ids[bad_index])
            raise RuntimeError(f"No window transition available for start_step_id={bad_id}")

        last_step_ids = sampled_start_ids + window_steps.astype(np.int64) - 1
        last_indices = np.mod(last_step_ids, int(self._capacity)).astype(
            np.int64,
            copy=False,
        )
        return step_ids, indices, valid, window_steps, last_step_ids, last_indices

    def _sample_window_metadata(
        self,
        *,
        batch_size: int,
        indx: Optional[np.ndarray] = None,
        profile: Optional[dict] = None,
    ) -> dict[str, np.ndarray]:
        cleanup_start = time.perf_counter()
        self._cleanup_stale_candidates()
        _profile_add(profile, "cleanup_sec", time.perf_counter() - cleanup_start)
        if indx is None:
            if not self._candidate_start_step_ids:
                raise RuntimeError(
                    "StepWindowReplayBuffer has no eligible window starts. "
                    f"(num_steps={self.num_steps}, num_windows={self.num_windows}, "
                    f"sample_stride={self.sample_stride})"
                )
            candidate_choice_start = time.perf_counter()
            sampled_start_ids = self._candidate_start_step_ids.sample(
                self.np_random,
                int(batch_size),
            )
            _profile_add(
                profile,
                "candidate_choice_sec",
                time.perf_counter() - candidate_choice_start,
            )
            _profile_set(profile, "candidate_count", int(len(self._candidate_start_step_ids)))
        else:
            candidate_choice_start = time.perf_counter()
            sampled_start_ids = np.asarray(indx, dtype=np.int64).reshape(-1)
            if int(sampled_start_ids.shape[0]) != int(batch_size):
                raise ValueError(
                    "indx length must equal batch_size, got "
                    f"{sampled_start_ids.shape[0]} != {batch_size}"
                )
            _profile_add(
                profile,
                "candidate_choice_sec",
                time.perf_counter() - candidate_choice_start,
            )
            _profile_set(profile, "candidate_count", int(self.num_windows))

        metadata_start = time.perf_counter()
        start_indices = np.mod(sampled_start_ids, int(self._capacity)).astype(
            np.int64,
            copy=False,
        )
        (
            step_ids,
            window_indices,
            valid,
            window_steps,
            last_step_ids,
            last_indices,
        ) = self._batch_window_indices(sampled_start_ids)
        _profile_add(profile, "window_metadata_sec", time.perf_counter() - metadata_start)
        return {
            "sampled_start_ids": sampled_start_ids,
            "start_indices": start_indices,
            "step_ids": step_ids,
            "window_indices": window_indices,
            "valid": valid,
            "window_steps": window_steps,
            "last_step_ids": last_step_ids,
            "last_indices": last_indices,
        }

    def _validate_window_metadata(self, metadata: dict[str, np.ndarray]) -> bool:
        window_indices = metadata["window_indices"]
        step_ids = metadata["step_ids"]
        valid = metadata["valid"]
        if np.any(self._step_ids[window_indices[valid]] != step_ids[valid]):
            return False

        start_indices = metadata["start_indices"]
        sampled_start_ids = metadata["sampled_start_ids"]
        if np.any(self._step_ids[start_indices] != sampled_start_ids):
            return False

        last_indices = metadata["last_indices"]
        last_step_ids = metadata["last_step_ids"]
        if np.any(self._step_ids[last_indices] != last_step_ids):
            return False
        return True

    def _copy_observations_batch(
        self,
        *,
        start_indices: np.ndarray,
        last_step_ids: np.ndarray,
        last_indices: np.ndarray,
        pack_obs_and_next_obs: bool,
    ) -> DatasetDict:
        del last_step_ids, last_indices, pack_obs_and_next_obs
        return _take_at(self.dataset_dict["observations"], start_indices)

    def _copy_next_observations_batch(
        self,
        *,
        last_step_ids: np.ndarray,
        last_indices: np.ndarray,
        pack_obs_and_next_obs: bool = False,
    ) -> DatasetDict:
        del last_step_ids, pack_obs_and_next_obs
        return _take_at(self.dataset_dict["next_observations"], last_indices)

    def _build_transition_batch_from_metadata(
        self,
        metadata: dict[str, np.ndarray],
        *,
        pack_obs_and_next_obs: bool = False,
        profile: Optional[dict] = None,
    ) -> DatasetDict:
        start_indices = metadata["start_indices"]
        window_indices = metadata["window_indices"]
        valid = metadata["valid"]
        window_steps = metadata["window_steps"]
        last_step_ids = metadata["last_step_ids"]
        last_indices = metadata["last_indices"]

        action_window_start = time.perf_counter()
        valid_action_shape = valid.shape + (1,) * len(self._step_action_shape)
        valid_actions = valid.reshape(valid_action_shape)
        sampled_actions = self.dataset_dict["actions"][window_indices]
        action_window = np.array(
            np.where(
                valid_actions,
                sampled_actions,
                np.zeros((), dtype=sampled_actions.dtype),
            ),
            copy=True,
        )
        _profile_add(
            profile,
            "action_window_sec",
            time.perf_counter() - action_window_start,
        )

        action_mask_start = time.perf_counter()
        action_mask = np.broadcast_to(
            valid_actions,
            valid.shape + self._step_action_shape,
        ).astype(np.float32, copy=True)
        _profile_add(
            profile,
            "action_mask_sec",
            time.perf_counter() - action_mask_start,
        )

        reward_mask_start = time.perf_counter()
        offsets = np.arange(int(self.window_size), dtype=np.float32)
        discounts = np.power(np.float32(self.discount), offsets)
        rewards = self.dataset_dict["rewards"][window_indices]
        discounted_rewards = np.sum(
            rewards * valid.astype(np.float32) * discounts[None, :],
            axis=1,
        ).astype(np.float32)
        terminal_discounts = np.power(
            np.float32(self.discount),
            np.maximum(window_steps.astype(np.int64) - 1, 0).astype(np.float32),
        )
        masks = (
            terminal_discounts * self.dataset_dict["masks"][last_indices].astype(np.float32)
        ).astype(np.float32)
        dones = np.any(valid & self.dataset_dict["dones"][window_indices], axis=1)
        _profile_add(
            profile,
            "reward_mask_sec",
            time.perf_counter() - reward_mask_start,
        )

        obs_take_start = time.perf_counter()
        observations = self._copy_observations_batch(
            start_indices=start_indices,
            last_step_ids=last_step_ids,
            last_indices=last_indices,
            pack_obs_and_next_obs=pack_obs_and_next_obs,
        )
        _profile_add(profile, "obs_take_sec", time.perf_counter() - obs_take_start)

        next_obs_take_start = time.perf_counter()
        next_observations = self._copy_next_observations_batch(
            last_step_ids=last_step_ids,
            last_indices=last_indices,
            pack_obs_and_next_obs=pack_obs_and_next_obs,
        )
        _profile_add(
            profile,
            "next_obs_take_sec",
            time.perf_counter() - next_obs_take_start,
        )

        batch = {
            "observations": observations,
            "actions": action_window,
            "action_mask": action_mask,
            "next_observations": next_observations,
            "rewards": discounted_rewards,
            "masks": masks,
            "dones": dones,
            "window_steps": window_steps,
        }
        batch.update(self._copy_extra_fields_batch(start_indices))
        return batch

    def _build_transition_batch(
        self,
        sampled_start_ids: np.ndarray,
        *,
        pack_obs_and_next_obs: bool = False,
    ) -> DatasetDict:
        sampled_start_ids = np.asarray(sampled_start_ids, dtype=np.int64).reshape(-1)
        metadata = self._sample_window_metadata(
            batch_size=int(sampled_start_ids.shape[0]),
            indx=sampled_start_ids,
        )
        return self._build_transition_batch_from_metadata(
            metadata,
            pack_obs_and_next_obs=pack_obs_and_next_obs,
        )

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        profile: Optional[dict] = None,
        pack_obs_and_next_obs: bool = False,
    ) -> DatasetDict:
        sample_start = time.perf_counter()
        metadata = self._sample_window_metadata(
            batch_size=int(batch_size),
            indx=indx,
            profile=profile,
        )
        transition_build_start = time.perf_counter()
        batch = self._build_transition_batch_from_metadata(
            metadata,
            pack_obs_and_next_obs=bool(pack_obs_and_next_obs),
            profile=profile,
        )
        _profile_add(
            profile,
            "transition_build_sec",
            time.perf_counter() - transition_build_start,
        )
        _profile_add(profile, "stack_sec", 0.0)
        _profile_set(profile, "batch_size", int(batch_size))
        _profile_add(profile, "sample_total_sec", time.perf_counter() - sample_start)
        if keys is None:
            return batch
        select_start = time.perf_counter()
        selected_keys = list(keys)
        selected = {key: batch[key] for key in selected_keys}
        _profile_add(profile, "select_keys_sec", time.perf_counter() - select_start)
        return selected

    def get_iterator(self, queue_size: int = 2, sample_args: dict = None, device=None):
        sample_args = sample_args or {}
        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(_to_torch(data, device=device))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


class MemoryEfficientStepWindowReplayBuffer(StepWindowReplayBuffer):
    """Step-window replay that reconstructs next-observation pixels from the next step.

    Explicit next-observation pixels are only kept at step-stream boundaries where
    no contiguous successor step is available to reconstruct them.
    """

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
        pixel_keys: Iterable[str] = ("pixels",),
        extra_fields: Optional[Mapping[str, gym.Space]] = None,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError(
                "MemoryEfficientStepWindowReplayBuffer requires Dict observation_space"
            )
        if next_observation_space is None:
            next_observation_space = observation_space
        if not isinstance(next_observation_space, gym.spaces.Dict):
            raise TypeError(
                "MemoryEfficientStepWindowReplayBuffer requires Dict next_observation_space"
            )

        self.pixel_keys = tuple(pixel_keys)
        observation_space_copy = copy.deepcopy(observation_space)
        next_observation_space_dict = copy.deepcopy(next_observation_space.spaces)
        self._pixel_spaces: dict[str, gym.Space] = {}
        for pixel_key in self.pixel_keys:
            if pixel_key not in observation_space_copy.spaces:
                raise KeyError(
                    f"Missing pixel key {pixel_key!r} in observation_space"
                )
            self._pixel_spaces[pixel_key] = copy.deepcopy(
                observation_space_copy.spaces[pixel_key]
            )
            next_observation_space_dict.pop(pixel_key, None)

        super().__init__(
            observation_space=observation_space_copy,
            action_space=action_space,
            capacity=capacity,
            window_size=window_size,
            discount=discount,
            sample_stride=sample_stride,
            require_full_window=require_full_window,
            next_observation_space=gym.spaces.Dict(next_observation_space_dict),
            extra_fields=extra_fields,
        )

        self._explicit_next_pixels = {
            key: np.empty(
                (int(self._capacity), *self._pixel_spaces[key].shape),
                dtype=self._pixel_spaces[key].dtype,
            )
            for key in self.pixel_keys
        }
        self._has_explicit_next_pixels = np.zeros((int(self._capacity),), dtype=bool)

    def insert(self, data_dict: DatasetDict):
        if "observations" not in data_dict or "next_observations" not in data_dict:
            raise KeyError(
                "memory-efficient step window replay insert requires "
                "'observations' and 'next_observations'"
            )

        step_record = copy.deepcopy(data_dict)
        old_insert_index = int(self._insert_index)

        next_obs_pixels: dict[str, np.ndarray] = {}
        for pixel_key in self.pixel_keys:
            if pixel_key not in step_record["observations"]:
                raise KeyError(
                    f"Missing pixel key {pixel_key!r} in observations"
                )
            if pixel_key not in step_record["next_observations"]:
                raise KeyError(
                    f"Missing pixel key {pixel_key!r} in next_observations"
                )
            next_obs_pixels[pixel_key] = np.asarray(
                step_record["next_observations"].pop(pixel_key),
                dtype=self._pixel_spaces[pixel_key].dtype,
            )

        previous_step_id = int(self._insert_count - 1)
        previous_index: Optional[int] = None
        if self._is_active_step_id(previous_step_id):
            previous_index = self._buffer_index(previous_step_id)

        super().insert(step_record)

        self._has_explicit_next_pixels[old_insert_index] = True
        for pixel_key in self.pixel_keys:
            self._explicit_next_pixels[pixel_key][old_insert_index] = next_obs_pixels[
                pixel_key
            ]

        if previous_index is None:
            return

        current_episode_id = int(self._episode_ids[old_insert_index])
        current_episode_step = int(self._episode_steps[old_insert_index])
        previous_episode_id = int(self._episode_ids[previous_index])
        previous_episode_step = int(self._episode_steps[previous_index])
        if (
            previous_episode_id == current_episode_id
            and previous_episode_step + 1 == current_episode_step
        ):
            self._has_explicit_next_pixels[previous_index] = False

    def _copy_next_pixel(self, last_step_id: int, *, pixel_key: str) -> np.ndarray:
        last_idx = self._buffer_index(last_step_id)
        if self._has_explicit_next_pixels[last_idx]:
            return np.array(
                self._explicit_next_pixels[pixel_key][last_idx],
                copy=True,
            )

        next_step_id = int(last_step_id + 1)
        if not self._is_active_step_id(next_step_id):
            raise RuntimeError(
                "Cannot reconstruct next-observation pixels without an active "
                f"next step for step_id={last_step_id}"
            )

        next_idx = self._buffer_index(next_step_id)
        last_episode_id = int(self._episode_ids[last_idx])
        next_episode_id = int(self._episode_ids[next_idx])
        last_episode_step = int(self._episode_steps[last_idx])
        next_episode_step = int(self._episode_steps[next_idx])
        if next_episode_id != last_episode_id or next_episode_step != (last_episode_step + 1):
            raise RuntimeError(
                "Cannot reconstruct next-observation pixels across episode/discontinuous "
                f"boundary: step_id={last_step_id}, next_step_id={next_step_id}"
            )
        return np.array(
            self.dataset_dict["observations"][pixel_key][next_idx],
            copy=True,
        )

    def _copy_next_observations(self, last_step_id: int) -> DatasetDict:
        last_idx = self._buffer_index(last_step_id)
        next_obs = _copy_at(self.dataset_dict["next_observations"], last_idx)
        for pixel_key in self.pixel_keys:
            next_obs[pixel_key] = self._copy_next_pixel(
                last_step_id,
                pixel_key=pixel_key,
            )
        return next_obs

    def _copy_next_pixel_batch(
        self,
        last_step_ids: np.ndarray,
        last_indices: np.ndarray,
        *,
        pixel_key: str,
    ) -> np.ndarray:
        explicit = self._has_explicit_next_pixels[last_indices]
        pixels = np.empty(
            (int(last_indices.shape[0]), *self._pixel_spaces[pixel_key].shape),
            dtype=self._pixel_spaces[pixel_key].dtype,
        )

        if np.any(explicit):
            explicit_indices = last_indices[explicit]
            pixels[explicit] = self._explicit_next_pixels[pixel_key][explicit_indices]

        implicit = ~explicit
        if np.any(implicit):
            implicit_last_step_ids = last_step_ids[implicit]
            implicit_last_indices = last_indices[implicit]
            next_step_ids = implicit_last_step_ids + 1
            next_indices = np.mod(next_step_ids, int(self._capacity)).astype(
                np.int64,
                copy=False,
            )
            valid_next = self._step_ids[next_indices] == next_step_ids
            valid_next &= self._episode_ids[next_indices] == self._episode_ids[
                implicit_last_indices
            ]
            valid_next &= self._episode_steps[next_indices] == (
                self._episode_steps[implicit_last_indices] + 1
            )
            if not np.all(valid_next):
                bad_index = int(np.nonzero(~valid_next)[0][0])
                bad_step_id = int(implicit_last_step_ids[bad_index])
                raise RuntimeError(
                    "Cannot reconstruct next-observation pixels without an active "
                    f"contiguous next step for step_id={bad_step_id}"
                )
            pixels[implicit] = self.dataset_dict["observations"][pixel_key][
                next_indices
            ]

        return pixels

    def _copy_observations_batch(
        self,
        *,
        start_indices: np.ndarray,
        last_step_ids: np.ndarray,
        last_indices: np.ndarray,
        pack_obs_and_next_obs: bool,
    ) -> DatasetDict:
        observations = _take_at(self.dataset_dict["observations"], start_indices)
        if not pack_obs_and_next_obs:
            return observations

        for pixel_key in self.pixel_keys:
            observations[pixel_key] = _pack_obs_and_next_pixels(
                observations[pixel_key],
                self._copy_next_pixel_batch(
                    last_step_ids,
                    last_indices,
                    pixel_key=pixel_key,
                ),
            )
        return observations

    def _copy_next_observations_batch(
        self,
        *,
        last_step_ids: np.ndarray,
        last_indices: np.ndarray,
        pack_obs_and_next_obs: bool = False,
    ) -> DatasetDict:
        next_obs = _take_at(self.dataset_dict["next_observations"], last_indices)
        if pack_obs_and_next_obs:
            return next_obs
        for pixel_key in self.pixel_keys:
            next_obs[pixel_key] = self._copy_next_pixel_batch(
                last_step_ids,
                last_indices,
                pixel_key=pixel_key,
            )
        return next_obs

    def _build_transition(self, start_step_id: int) -> DatasetDict:
        step_ids, boundary, _ = self._collect_segment(start_step_id)
        if not step_ids:
            raise RuntimeError(
                f"No window transition available for start_step_id={start_step_id}"
            )

        window_steps = int(len(step_ids))
        step_actions = self.dataset_dict["actions"]
        action_window = np.zeros(self._window_action_shape, dtype=step_actions.dtype)
        action_mask = np.zeros(self._window_action_shape, dtype=np.float32)
        discounted_reward = 0.0

        for offset, step_id in enumerate(step_ids):
            idx = self._buffer_index(step_id)
            action_window[offset] = np.array(step_actions[idx], copy=True)
            action_mask[offset] = np.ones(
                self._step_action_shape,
                dtype=np.float32,
            )
            discounted_reward += (self.discount**offset) * float(
                self.dataset_dict["rewards"][idx]
            )

        start_idx = self._buffer_index(start_step_id)
        last_idx = self._buffer_index(step_ids[-1])
        last_mask = float(self.dataset_dict["masks"][last_idx])
        return {
            "observations": _copy_at(self.dataset_dict["observations"], start_idx),
            "actions": action_window,
            "action_mask": action_mask,
            "next_observations": self._copy_next_observations(step_ids[-1]),
            "rewards": np.float32(discounted_reward),
            "masks": np.float32(
                (self.discount ** max(0, window_steps - 1)) * last_mask
            ),
            "dones": bool(boundary),
            "window_steps": np.int32(window_steps),
        }
