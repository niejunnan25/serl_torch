from __future__ import annotations

import unittest

import numpy as np

from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.residual.observation import build_chunk_residual_observation_space
from serl_launcher.residual.observation import build_chunk_residual_sample_obs
from serl_launcher.residual.observation import prepare_base_actions_chunk


class ResidualObservationHelpersTest(unittest.TestCase):
    def test_prepare_base_actions_chunk_clips_to_horizon(self) -> None:
        base_actions = np.arange(15, dtype=np.float32).reshape(5, 3)
        prepared = prepare_base_actions_chunk(
            base_actions=base_actions,
            chunk_horizon=3,
        )
        self.assertEqual(prepared.shape, (3, 3))
        np.testing.assert_array_equal(prepared, base_actions[:3])

    def test_build_chunk_residual_obs_assembles_schema(self) -> None:
        residual_obs = build_chunk_residual_obs(
            robot_state=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            images={
                "image_rgb_0": np.full((4, 5, 3), 7, dtype=np.uint8),
                "image_rgb_1": np.full((4, 5, 3), 9, dtype=np.uint8),
            },
            image_keys=("image_rgb_0", "image_rgb_1"),
            base_actions=np.asarray(
                [[0.1, 0.2], [0.3, 0.4]],
                dtype=np.float32,
            ),
            residual_alpha=0.25,
        )
        self.assertEqual(residual_obs["robot_proprio"].shape, (1, 3))
        self.assertEqual(residual_obs["base_action"].shape, (1, 2))
        self.assertEqual(residual_obs["base_action_chunk"].shape, (1, 2, 2))
        self.assertEqual(residual_obs["alpha"].shape, (1, 1))
        self.assertEqual(residual_obs["image_rgb_0"].shape, (1, 4, 5, 3))
        self.assertEqual(residual_obs["image_rgb_1"].shape, (1, 4, 5, 3))
        self.assertEqual(residual_obs["image_rgb_0"].dtype, np.uint8)
        self.assertEqual(residual_obs["robot_proprio"].dtype, np.float32)

    def test_sample_obs_and_observation_space_match(self) -> None:
        sample_obs = build_chunk_residual_sample_obs(
            state_dim=6,
            action_dim=4,
            chunk_horizon=3,
            image_keys=("image_rgb_0", "image_rgb_1"),
            image_height=8,
            image_width=10,
        )
        observation_space = build_chunk_residual_observation_space(
            sample_obs=sample_obs,
            image_keys=("image_rgb_0", "image_rgb_1"),
        )
        self.assertEqual(sample_obs["robot_proprio"].shape, (1, 6))
        self.assertEqual(sample_obs["base_action_chunk"].shape, (1, 3, 4))
        self.assertEqual(sample_obs["image_rgb_0"].shape, (1, 8, 10, 3))
        self.assertEqual(observation_space["image_rgb_0"].dtype, np.uint8)
        self.assertEqual(observation_space["robot_proprio"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
