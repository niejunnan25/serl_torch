from __future__ import annotations

import unittest

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from serl_launcher.data.data_store import MemoryEfficientStepWindowReplayBufferDataStore
from serl_launcher.residual.chunk_replay import create_chunk_replay_buffer
from serl_launcher.residual.chunk_replay import reshape_chunk_batch_for_training
from serl_launcher.residual.chunk_replay import sample_mixed_training_batch


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


if __name__ == "__main__":
    unittest.main()
