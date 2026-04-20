from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[3] / "serl_launcher"
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_torch.examples.libero.rlpd.observation import build_rlpd_obs
from serl_torch.examples.libero.rlpd.observation import build_rlpd_observation_space
from serl_torch.examples.libero.rlpd.observation import build_rlpd_sample_obs


class LiberoRLPDObservationTest(unittest.TestCase):
    def test_build_rlpd_obs_contains_only_direct_keys(self) -> None:
        raw_obs = {
            "agentview_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "eye_in_hand_rgb": np.ones((8, 8, 3), dtype=np.uint8),
            "ee_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "ee_ori": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
            "gripper_states": np.asarray([0.7, 0.8], dtype=np.float32),
        }

        obs = build_rlpd_obs(obs=raw_obs, image_keys=("image", "wrist_image"))
        self.assertEqual(set(obs), {"robot_proprio", "image_rgb_0", "image_rgb_1"})
        self.assertEqual(obs["robot_proprio"].shape, (1, 8))
        self.assertEqual(obs["image_rgb_0"].shape, (1, 224, 224, 3))
        self.assertEqual(obs["image_rgb_0"].dtype, np.uint8)

    def test_build_rlpd_observation_space_matches_sample_obs(self) -> None:
        sample_obs = build_rlpd_sample_obs(image_keys=("image", "wrist_image"))
        observation_space = build_rlpd_observation_space(
            sample_obs=sample_obs,
            image_keys=("image", "wrist_image"),
        )

        self.assertEqual(
            set(observation_space.spaces),
            {"robot_proprio", "image_rgb_0", "image_rgb_1"},
        )
        self.assertEqual(observation_space["robot_proprio"].shape, (1, 8))
        self.assertEqual(observation_space["image_rgb_0"].dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
