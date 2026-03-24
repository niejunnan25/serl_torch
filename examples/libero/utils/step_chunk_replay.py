"""Step-stream chunk replay buffer aligned with RLT-style window sampling."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np


_OBS_SPECIAL_KEYS = {"state", "base_action", "base_action_chunk", "xi"}


def _strip_sample_axis(template: np.ndarray) -> np.ndarray:
    arr = np.asarray(template)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return np.array(arr[0], copy=True)
    return np.array(arr, copy=True)


class StepChunkReplayBuffer:
    """Store step-level rollout data and sample chunk windows with env-step stride."""

    def __init__(
        self,
        *,
        sample_observation_template: Dict[str, np.ndarray],
        state_core_dim: int,
        step_action_dim: int,
        chunk_horizon: int,
        discount: float,
        capacity: int,
        sample_stride: int = 1,
        require_full_horizon: bool = False,
        pad_action_to_horizon: bool = True,
    ) -> None:
        self.capacity = int(capacity)
        self.state_core_dim = int(state_core_dim)
        self.step_action_dim = int(step_action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.discount = float(discount)
        self.sample_stride = max(1, int(sample_stride))
        self.require_full_horizon = bool(require_full_horizon)
        self.pad_action_to_horizon = bool(pad_action_to_horizon)
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.state_core_dim <= 0:
            raise ValueError(f"state_core_dim must be positive, got {self.state_core_dim}")
        if self.step_action_dim <= 0:
            raise ValueError(f"step_action_dim must be positive, got {self.step_action_dim}")
        if self.chunk_horizon <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {self.chunk_horizon}")
        if not self.pad_action_to_horizon:
            raise ValueError("StepChunkReplayBuffer currently requires pad_action_to_horizon=true")

        self._sample_observation_template = {
            key: np.array(value, copy=True)
            for key, value in sample_observation_template.items()
        }
        stripped_template = {
            key: _strip_sample_axis(np.asarray(value))
            for key, value in sample_observation_template.items()
        }
        if "base_action_chunk" not in stripped_template:
            raise ValueError("sample_observation_template must include 'base_action_chunk' in chunk-step mode")
        self._image_keys = tuple(key for key in stripped_template if key not in _OBS_SPECIAL_KEYS)
        self._zero_obs_template = {
            key: np.zeros_like(value)
            for key, value in self._sample_observation_template.items()
        }

        self._images = {
            key: np.empty((self.capacity, *stripped_template[key].shape), dtype=stripped_template[key].dtype)
            for key in self._image_keys
        }
        self._state_core = np.empty((self.capacity, self.state_core_dim), dtype=np.float32)
        self._base_action = np.empty((self.capacity, self.step_action_dim), dtype=np.float32)
        self._base_action_norm = np.empty((self.capacity, self.step_action_dim), dtype=np.float32)
        self._final_action = np.empty((self.capacity, self.step_action_dim), dtype=np.float32)
        self._rewards = np.empty((self.capacity,), dtype=np.float32)
        self._dones = np.empty((self.capacity,), dtype=bool)
        self._xis = np.empty((self.capacity,), dtype=np.float32)
        self._episode_ids = np.empty((self.capacity,), dtype=np.int64)
        self._episode_steps = np.empty((self.capacity,), dtype=np.int32)
        self._step_ids = np.full((self.capacity,), -1, dtype=np.int64)

        self._insert_index = 0
        self._insert_count = 0
        self._size = 0
        self._candidate_start_step_ids: deque[int] = deque()
        self._candidate_start_step_set: set[int] = set()

    def __len__(self) -> int:
        return int(len(self._candidate_start_step_ids))

    @property
    def num_steps(self) -> int:
        return int(self._size)

    def _min_active_step_id(self) -> int:
        return int(max(0, self._insert_count - self._size))

    def _is_active_step_id(self, step_id: int) -> bool:
        step_id = int(step_id)
        if step_id < self._min_active_step_id() or step_id >= int(self._insert_count):
            return False
        idx = int(step_id % self.capacity)
        return int(self._step_ids[idx]) == step_id

    def _buffer_index(self, step_id: int) -> int:
        if not self._is_active_step_id(step_id):
            raise KeyError(f"Inactive step_id={step_id}")
        return int(step_id % self.capacity)

    def _cleanup_stale_candidates(self) -> None:
        min_active = self._min_active_step_id()
        while self._candidate_start_step_ids and self._candidate_start_step_ids[0] < min_active:
            stale = self._candidate_start_step_ids.popleft()
            self._candidate_start_step_set.discard(int(stale))

    def _collect_segment(self, start_step_id: int, max_len: int) -> Tuple[List[int], bool, bool]:
        if not self._is_active_step_id(start_step_id):
            return [], False, False
        start_idx = self._buffer_index(start_step_id)
        episode_id = int(self._episode_ids[start_idx])
        episode_step = int(self._episode_steps[start_idx])
        collected: List[int] = []
        terminal = False
        for offset in range(int(max_len)):
            step_id = int(start_step_id + offset)
            if not self._is_active_step_id(step_id):
                break
            idx = self._buffer_index(step_id)
            if int(self._episode_ids[idx]) != episode_id:
                break
            if int(self._episode_steps[idx]) != episode_step + offset:
                break
            collected.append(step_id)
            if bool(self._dones[idx]):
                terminal = True
                break
        return collected, terminal, bool(len(collected) == int(max_len))

    def _obs_window_ready(self, start_step_id: int) -> bool:
        step_ids, terminal, full = self._collect_segment(start_step_id, self.chunk_horizon)
        if not step_ids:
            return False
        if full:
            return True
        return bool(terminal and (not self.require_full_horizon))

    def _transition_ready(self, start_step_id: int) -> bool:
        if not self._is_active_step_id(start_step_id):
            return False
        start_idx = self._buffer_index(start_step_id)
        if int(self._episode_steps[start_idx]) % int(self.sample_stride) != 0:
            return False

        step_ids, terminal, full = self._collect_segment(start_step_id, self.chunk_horizon)
        if not step_ids:
            return False
        if terminal:
            if self.require_full_horizon and len(step_ids) < self.chunk_horizon:
                return False
            return True
        if not full:
            return False
        return self._obs_window_ready(start_step_id + len(step_ids))

    def _refresh_candidate_window(self, last_inserted_step_id: int) -> None:
        self._cleanup_stale_candidates()
        lookback = max(1, 2 * int(self.chunk_horizon))
        start_step = max(self._min_active_step_id(), int(last_inserted_step_id) - lookback)
        for step_id in range(int(start_step), int(last_inserted_step_id) + 1):
            if step_id in self._candidate_start_step_set:
                continue
            if self._transition_ready(step_id):
                self._candidate_start_step_ids.append(int(step_id))
                self._candidate_start_step_set.add(int(step_id))

    def insert(self, data_dict: Dict[str, Any]) -> None:
        obs_core = data_dict["obs_core"]
        if "state_core" not in obs_core:
            raise KeyError("obs_core must include 'state_core'")
        state_core = np.asarray(obs_core["state_core"], dtype=np.float32).reshape(-1)
        if state_core.shape[0] != self.state_core_dim:
            raise ValueError(
                "state_core dim mismatch during replay insertion: "
                f"{state_core.shape[0]} != {self.state_core_dim}"
            )

        base_action = np.asarray(data_dict["base_action"], dtype=np.float32).reshape(-1)
        base_action_norm = np.asarray(data_dict["base_action_norm"], dtype=np.float32).reshape(-1)
        final_action = np.asarray(data_dict["actions"], dtype=np.float32).reshape(-1)
        for name, arr in (
            ("base_action", base_action),
            ("base_action_norm", base_action_norm),
            ("actions", final_action),
        ):
            if arr.shape[0] != self.step_action_dim:
                raise ValueError(
                    f"{name} dim mismatch during replay insertion: {arr.shape[0]} != {self.step_action_dim}"
                )

        insert_index = int(self._insert_index)
        for key in self._image_keys:
            if key not in obs_core:
                raise KeyError(f"obs_core is missing image key '{key}'")
            self._images[key][insert_index] = np.asarray(obs_core[key], dtype=self._images[key].dtype)
        self._state_core[insert_index] = state_core
        self._base_action[insert_index] = base_action
        self._base_action_norm[insert_index] = base_action_norm
        self._final_action[insert_index] = final_action
        self._rewards[insert_index] = np.float32(data_dict["rewards"])
        self._dones[insert_index] = bool(data_dict["dones"])
        self._xis[insert_index] = np.float32(data_dict["xi"])
        self._episode_ids[insert_index] = np.int64(data_dict["episode_id"])
        self._episode_steps[insert_index] = np.int32(data_dict["episode_step"])
        self._step_ids[insert_index] = np.int64(self._insert_count)

        current_step_id = int(self._insert_count)
        self._insert_index = (insert_index + 1) % self.capacity
        self._insert_count += 1
        self._size = min(self._size + 1, self.capacity)

        self._refresh_candidate_window(current_step_id)

    def _assemble_obs(self, start_step_id: int) -> Dict[str, np.ndarray]:
        step_ids, _, _ = self._collect_segment(start_step_id, self.chunk_horizon)
        if not step_ids:
            raise RuntimeError(f"Unable to assemble observation for step_id={start_step_id}")
        start_idx = self._buffer_index(start_step_id)
        base_window = np.zeros((self.chunk_horizon, self.step_action_dim), dtype=np.float32)
        base_window_norm = np.zeros((self.chunk_horizon, self.step_action_dim), dtype=np.float32)
        for offset, step_id in enumerate(step_ids[: self.chunk_horizon]):
            idx = self._buffer_index(step_id)
            base_window[offset] = self._base_action[idx]
            base_window_norm[offset] = self._base_action_norm[idx]
        fused_state = np.concatenate(
            (
                self._state_core[start_idx],
                self._base_action_norm[start_idx],
                base_window_norm.reshape(-1),
                np.asarray([self._xis[start_idx]], dtype=np.float32),
            ),
            axis=0,
        ).astype(np.float32)
        obs = {
            "state": np.expand_dims(fused_state, axis=0),
            "base_action": np.expand_dims(np.array(self._base_action[start_idx], copy=True), axis=0),
            "xi": np.asarray([[self._xis[start_idx]]], dtype=np.float32),
            "base_action_chunk": np.expand_dims(np.array(base_window, copy=True), axis=0),
        }
        for key in self._image_keys:
            obs[key] = np.expand_dims(np.array(self._images[key][start_idx], copy=True), axis=0)
        return obs

    def _zero_obs(self) -> Dict[str, np.ndarray]:
        return {
            key: np.array(value, copy=True)
            for key, value in self._zero_obs_template.items()
        }

    def _build_transition(self, start_step_id: int) -> Dict[str, Any]:
        step_ids, terminal, _ = self._collect_segment(start_step_id, self.chunk_horizon)
        if not step_ids:
            raise RuntimeError(f"No chunk transition available for start_step_id={start_step_id}")
        chunk_steps = int(len(step_ids))
        action_chunk = np.zeros((self.chunk_horizon, self.step_action_dim), dtype=np.float32)
        action_mask = np.zeros((self.chunk_horizon, self.step_action_dim), dtype=np.float32)
        discounted_reward = 0.0
        for offset, step_id in enumerate(step_ids):
            idx = self._buffer_index(step_id)
            action_chunk[offset] = self._final_action[idx]
            action_mask[offset] = 1.0
            discounted_reward += (self.discount ** offset) * float(self._rewards[idx])

        obs = self._assemble_obs(start_step_id)
        if terminal:
            next_obs = self._zero_obs()
        else:
            next_obs = self._assemble_obs(start_step_id + chunk_steps)

        return {
            "observations": obs,
            "actions": action_chunk.reshape(-1),
            "action_mask": action_mask.reshape(-1),
            "next_observations": next_obs,
            "rewards": np.float32(discounted_reward),
            "masks": np.float32(0.0 if terminal else (self.discount ** max(0, chunk_steps - 1))),
            "dones": bool(terminal),
            "chunk_steps": np.int32(chunk_steps),
        }

    def sample(self, batch_size: int) -> Dict[str, Any]:
        self._cleanup_stale_candidates()
        if not self._candidate_start_step_ids:
            raise RuntimeError(
                "StepChunkReplayBuffer has no eligible chunk starts. "
                f"(num_steps={self.num_steps}, sample_stride={self.sample_stride})"
            )
        sampled_start_ids = np.random.choice(
            np.asarray(self._candidate_start_step_ids, dtype=np.int64),
            size=int(batch_size),
            replace=True,
        )
        transitions = [self._build_transition(int(step_id)) for step_id in sampled_start_ids]
        obs_keys = tuple(transitions[0]["observations"].keys())
        next_obs_keys = tuple(transitions[0]["next_observations"].keys())
        return {
            "observations": {
                key: np.stack([transition["observations"][key] for transition in transitions], axis=0)
                for key in obs_keys
            },
            "actions": np.stack([transition["actions"] for transition in transitions], axis=0),
            "action_mask": np.stack([transition["action_mask"] for transition in transitions], axis=0),
            "next_observations": {
                key: np.stack([transition["next_observations"][key] for transition in transitions], axis=0)
                for key in next_obs_keys
            },
            "rewards": np.asarray([transition["rewards"] for transition in transitions], dtype=np.float32),
            "masks": np.asarray([transition["masks"] for transition in transitions], dtype=np.float32),
            "dones": np.asarray([transition["dones"] for transition in transitions], dtype=bool),
            "chunk_steps": np.asarray([transition["chunk_steps"] for transition in transitions], dtype=np.int32),
        }


ChunkReplayBuffer = StepChunkReplayBuffer
DecisionChunkReplayBuffer = StepChunkReplayBuffer
