from __future__ import annotations

import unittest

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.residual.chunk_window_replay import BatchPrefetcher
from serl_launcher.residual.chunk_window_replay import create_chunk_replay_buffer
from serl_launcher.residual.chunk_window_replay import (
    PreparedStepWindowReplayBufferSampler,
)
from serl_launcher.residual.chunk_window_replay import reshape_chunk_batch_for_training
from serl_launcher.residual.chunk_window_replay import sample_mixed_training_batch


class _FakeReplayBuffer:
    def __init__(self, *, size: int, fill_value: float):
        self._size = int(size)
        self._fill_value = float(fill_value)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> dict[str, object]:
        return {
            "observations": {
                "state": np.full((batch_size, 2), self._fill_value, dtype=np.float32),
            },
            "actions": np.full((batch_size, 2, 3), self._fill_value, dtype=np.float32),
            "action_mask": np.ones((batch_size, 2, 3), dtype=np.float32),
        }


class ChunkReplayTest(unittest.TestCase):
    def _make_step_replay(self):
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                )
            }
        )
        replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=2,
            chunk_horizon=3,
            discount=0.99,
            image_keys=tuple(),
            capacity=32,
        )
        for step in range(10):
            replay_buffer.insert(
                {
                    "episode_id": 0,
                    "episode_step": step,
                    "observations": {
                        "state": np.asarray([step, step + 1], dtype=np.float32),
                    },
                    "actions": np.asarray([step, -step], dtype=np.float32),
                    "next_observations": {
                        "state": np.asarray([step + 1, step + 2], dtype=np.float32),
                    },
                    "rewards": np.float32(1.0),
                    "masks": np.float32(1.0),
                    "dones": False,
                }
            )
        return replay_buffer

    def test_batch_prefetcher_returns_ordered_results(self) -> None:
        calls: list[int] = []

        def _sample() -> int:
            calls.append(len(calls) + 1)
            return calls[-1]

        prefetcher = BatchPrefetcher(_sample)
        try:
            self.assertEqual(prefetcher.get(), 1)
            self.assertEqual(prefetcher.get(), 2)
        finally:
            prefetcher.close()

        self.assertGreaterEqual(len(calls), 2)

    def test_batch_prefetcher_propagates_sample_errors(self) -> None:
        def _sample() -> int:
            raise RuntimeError("sample failed")

        prefetcher = BatchPrefetcher(_sample)
        with self.assertRaisesRegex(RuntimeError, "sample failed"):
            prefetcher.get()

    def test_create_chunk_replay_buffer_smoke(self) -> None:
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                )
            }
        )
        replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=6,
            chunk_horizon=5,
            discount=0.99,
            image_keys=tuple(),
            capacity=128,
        )

        self.assertIsInstance(
            replay_buffer,
            MemoryEfficientStepWindowReplayBufferDataStore,
        )
        self.assertEqual(replay_buffer.capacity, 128)
        self.assertEqual(replay_buffer.window_size, 5)
        self.assertEqual(replay_buffer._step_action_shape, (6,))

    def test_create_chunk_replay_buffer_forwards_sample_stride(self) -> None:
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                )
            }
        )
        replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=6,
            chunk_horizon=5,
            discount=0.99,
            image_keys=tuple(),
            capacity=128,
            sample_stride=2,
        )

        self.assertEqual(replay_buffer.sample_stride, 2)

    def test_mc_returns_are_sampled_from_window_start(self) -> None:
        observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                )
            }
        )
        replay_buffer = create_chunk_replay_buffer(
            observation_space=observation_space,
            action_dim=2,
            chunk_horizon=3,
            discount=0.99,
            image_keys=tuple(),
            capacity=32,
        )
        for step in range(6):
            replay_buffer.insert(
                {
                    "episode_id": 0,
                    "episode_step": step,
                    "observations": {
                        "state": np.asarray([step, step + 1], dtype=np.float32),
                    },
                    "actions": np.asarray([step, -step], dtype=np.float32),
                    "next_observations": {
                        "state": np.asarray([step + 1, step + 2], dtype=np.float32),
                    },
                    "rewards": np.float32(1.0),
                    "masks": np.float32(1.0),
                    "dones": False,
                    "mc_returns": np.float32(10.0 + step),
                    "mc_returns_valid": bool(step % 2 == 0),
                }
            )

        indices = np.asarray([0, 1, 2], dtype=np.int64)
        dynamic = replay_buffer.sample(len(indices), indx=indices)
        prepared = PreparedStepWindowReplayBufferSampler(replay_buffer).sample(
            len(indices),
            indx=indices,
        )

        expected_returns = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
        expected_valid = np.asarray([True, False, True])
        self.assertTrue(np.allclose(dynamic["mc_returns"], expected_returns))
        self.assertTrue(np.array_equal(dynamic["mc_returns_valid"], expected_valid))
        self.assertTrue(np.allclose(prepared["mc_returns"], expected_returns))
        self.assertTrue(np.array_equal(prepared["mc_returns_valid"], expected_valid))

    def test_missing_mc_returns_default_to_invalid_zero(self) -> None:
        replay_buffer = self._make_step_replay()
        batch = replay_buffer.sample(4, indx=np.asarray([0, 1, 2, 3], dtype=np.int64))

        self.assertTrue(np.allclose(batch["mc_returns"], 0.0))
        self.assertFalse(np.any(batch["mc_returns_valid"]))

    def test_reshape_chunk_batch_for_training_flattens_actions(self) -> None:
        batch = {
            "actions": np.zeros((4, 2, 3), dtype=np.float32),
            "action_mask": np.ones((4, 2, 3), dtype=np.float32),
        }

        reshaped = reshape_chunk_batch_for_training(batch)
        self.assertEqual(reshaped["actions"].shape, (4, 6))
        self.assertEqual(reshaped["action_mask"].shape, (4, 6))

    def test_sample_mixed_training_batch_online_only(self) -> None:
        batch, batch_mix = sample_mixed_training_batch(
            online_replay_buffer=_FakeReplayBuffer(size=100, fill_value=1.0),
            offline_replay_buffer=None,
            batch_size=8,
            offline_ratio=0.5,
        )

        self.assertEqual(batch_mix, {"online_batch_size": 8, "offline_batch_size": 0})
        self.assertEqual(batch["actions"].shape, (8, 6))
        self.assertTrue(np.allclose(batch["actions"], 1.0))

    def test_sample_mixed_training_batch_offline_only(self) -> None:
        batch, batch_mix = sample_mixed_training_batch(
            online_replay_buffer=_FakeReplayBuffer(size=0, fill_value=1.0),
            offline_replay_buffer=_FakeReplayBuffer(size=100, fill_value=2.0),
            batch_size=6,
            offline_ratio=0.5,
        )

        self.assertEqual(batch_mix, {"online_batch_size": 0, "offline_batch_size": 6})
        self.assertEqual(batch["actions"].shape, (6, 6))
        self.assertTrue(np.allclose(batch["actions"], 2.0))

    def test_sample_mixed_training_batch_mixed_ratio(self) -> None:
        batch, batch_mix = sample_mixed_training_batch(
            online_replay_buffer=_FakeReplayBuffer(size=100, fill_value=1.0),
            offline_replay_buffer=_FakeReplayBuffer(size=100, fill_value=2.0),
            batch_size=8,
            offline_ratio=0.25,
        )

        self.assertEqual(batch_mix, {"online_batch_size": 6, "offline_batch_size": 2})
        self.assertEqual(batch["actions"].shape, (8, 6))
        self.assertTrue(np.allclose(batch["actions"][:6], 1.0))
        self.assertTrue(np.allclose(batch["actions"][6:], 2.0))

    def test_sample_mixed_training_batch_can_skip_reshape(self) -> None:
        batch, batch_mix = sample_mixed_training_batch(
            online_replay_buffer=_FakeReplayBuffer(size=100, fill_value=1.0),
            offline_replay_buffer=_FakeReplayBuffer(size=100, fill_value=2.0),
            batch_size=4,
            offline_ratio=0.5,
            reshape_batch=False,
        )

        self.assertEqual(batch_mix, {"online_batch_size": 2, "offline_batch_size": 2})
        self.assertEqual(batch["actions"].shape, (4, 2, 3))

    def test_prepared_sampler_matches_dynamic_explicit_windows(self) -> None:
        replay_buffer = self._make_step_replay()
        prepared = PreparedStepWindowReplayBufferSampler(replay_buffer)
        indices = np.asarray([0, 1, 2, 3], dtype=np.int64)

        dynamic_profile: dict[str, float] = {}
        prepared_profile: dict[str, float] = {}
        dynamic = replay_buffer.sample(
            len(indices),
            indx=indices,
            profile=dynamic_profile,
        )
        cached = prepared.sample(
            len(indices),
            indx=indices,
            profile=prepared_profile,
        )

        self.assertEqual(prepared.num_windows, replay_buffer.num_windows)
        self.assertGreater(prepared.prepare_profile["prepare_window_sec"], 0.0)
        self.assertIn("prepared_cache_take_sec", prepared_profile)
        for key in ("actions", "action_mask", "rewards", "masks", "dones"):
            self.assertTrue(np.allclose(dynamic[key], cached[key]), key)
        self.assertTrue(
            np.allclose(
                dynamic["observations"]["state"],
                cached["observations"]["state"],
            )
        )
        self.assertTrue(
            np.allclose(
                dynamic["next_observations"]["state"],
                cached["next_observations"]["state"],
            )
        )


if __name__ == "__main__":
    unittest.main()
