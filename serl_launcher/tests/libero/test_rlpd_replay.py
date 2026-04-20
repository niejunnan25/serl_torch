from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_torch.examples.libero.rlpd.replay import create_rlpd_replay_buffer
from serl_torch.examples.libero.rlpd.replay import sample_mixed_training_batch


class _FakeReplayBuffer:
    def __init__(self, *, size: int, fill_value: float):
        self._size = int(size)
        self._fill_value = float(fill_value)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> dict[str, object]:
        return {
            "observations": {
                "robot_proprio": np.full((batch_size, 1, 8), self._fill_value, dtype=np.float32),
            },
            "actions": np.full((batch_size, 7), self._fill_value, dtype=np.float32),
        }


class LiberoRLPDReplayTest(unittest.TestCase):
    def test_create_rlpd_replay_buffer_smoke(self) -> None:
        observation_space = gym.spaces.Dict(
            {
                "robot_proprio": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1, 8),
                    dtype=np.float32,
                ),
                "image_rgb_0": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, 224, 224, 3),
                    dtype=np.uint8,
                ),
            }
        )
        try:
            replay_buffer = create_rlpd_replay_buffer(
                observation_space=observation_space,
                action_dim=7,
                image_keys=("image_rgb_0",),
                capacity=128,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"replay buffer runtime dependencies unavailable: {exc}")

        self.assertEqual(
            replay_buffer.__class__.__name__,
            "MemoryEfficientReplayBufferDataStore",
        )
        self.assertEqual(replay_buffer.capacity, 128)

    def test_sample_mixed_training_batch_online_only(self) -> None:
        batch, batch_mix = sample_mixed_training_batch(
            online_replay_buffer=_FakeReplayBuffer(size=100, fill_value=1.0),
            offline_replay_buffer=None,
            batch_size=8,
            offline_ratio=0.5,
        )

        self.assertEqual(batch_mix, {"online_batch_size": 8, "offline_batch_size": 0})
        self.assertEqual(batch["actions"].shape, (8, 7))
        self.assertTrue(np.allclose(batch["actions"], 1.0))

    def test_sample_mixed_training_batch_mixed_ratio(self) -> None:
        batch, batch_mix = sample_mixed_training_batch(
            online_replay_buffer=_FakeReplayBuffer(size=100, fill_value=1.0),
            offline_replay_buffer=_FakeReplayBuffer(size=100, fill_value=2.0),
            batch_size=8,
            offline_ratio=0.25,
        )

        self.assertEqual(batch_mix, {"online_batch_size": 6, "offline_batch_size": 2})
        self.assertEqual(batch["actions"].shape, (8, 7))
        self.assertTrue(np.allclose(batch["actions"][:6], 1.0))
        self.assertTrue(np.allclose(batch["actions"][6:], 2.0))


if __name__ == "__main__":
    unittest.main()
