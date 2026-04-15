from __future__ import annotations

import collections
import copy
from typing import Iterable, Optional

import gym
import numpy as np

from serl_launcher.data.dataset import Dataset, DatasetDict
from serl_launcher.data.replay_buffer import _init_replay_dict
from serl_launcher.data.replay_buffer import _insert_recursively
from serl_launcher.data.replay_buffer import _to_torch


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
        super().__init__(dataset_dict)

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
        self._candidate_start_step_ids = collections.deque()
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

    def insert(self, data_dict: DatasetDict):
        if "episode_id" not in data_dict:
            raise KeyError("step window replay insert requires 'episode_id'")
        if "episode_step" not in data_dict:
            raise KeyError("step window replay insert requires 'episode_step'")

        step_record = {key: data_dict[key] for key in self.dataset_dict.keys()}
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
        return {
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

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> DatasetDict:
        self._cleanup_stale_candidates()
        if indx is None:
            if not self._candidate_start_step_ids:
                raise RuntimeError(
                    "StepWindowReplayBuffer has no eligible window starts. "
                    f"(num_steps={self.num_steps}, num_windows={self.num_windows}, "
                    f"sample_stride={self.sample_stride})"
                )
            sampled_start_ids = self.np_random.choice(
                np.asarray(self._candidate_start_step_ids, dtype=np.int64),
                size=int(batch_size),
                replace=True,
            )
        else:
            sampled_start_ids = np.asarray(indx, dtype=np.int64).reshape(-1)
            if int(sampled_start_ids.shape[0]) != int(batch_size):
                raise ValueError(
                    "indx length must equal batch_size, got "
                    f"{sampled_start_ids.shape[0]} != {batch_size}"
                )

        transitions = [
            self._build_transition(int(step_id)) for step_id in sampled_start_ids
        ]
        batch = {
            key: _stack_nested([transition[key] for transition in transitions])
            for key in transitions[0]
        }
        if keys is None:
            return batch
        selected_keys = list(keys)
        return {key: batch[key] for key in selected_keys}

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
