from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.env.fake_task_env import AgiBotFakeTaskEnv


def _make_fake_env(*, arm_layout: str, action_dim: int) -> AgiBotFakeTaskEnv:
    return AgiBotFakeTaskEnv(
        task_name="fake_task",
        prompt="fake prompt",
        arm_layout=arm_layout,
        action_dim=action_dim,
        robot_action_dim=14,
        max_episode_steps=10,
        controller={"enabled": False},
        reset_hook=None,
        success_hook=None,
        fake_image_hw=(8, 8),
    )


class AgiBotFakeTaskEnvTest(unittest.TestCase):
    def test_right_arm_step_chunk_accepts_7d_and_holds_left_arm(self) -> None:
        env = _make_fake_env(arm_layout="right_arm", action_dim=7)
        try:
            obs = env.reset()
            self.assertEqual(obs["state/pose"].shape, (14,))

            right_action = np.arange(7, dtype=np.float32) + 10.0
            result = env.step_chunk(np.stack([right_action, right_action + 1.0]))

            self.assertEqual(result["num_steps"], 2)
            pose = np.asarray(result["obs"]["state/pose"], dtype=np.float32)
            np.testing.assert_allclose(
                pose[:7],
                np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
            )
            np.testing.assert_allclose(pose[7:], right_action + 1.0)
        finally:
            env.close()

    def test_left_arm_step_chunk_accepts_7d_and_holds_right_arm(self) -> None:
        env = _make_fake_env(arm_layout="left_arm", action_dim=7)
        try:
            env.reset()
            left_action = np.arange(7, dtype=np.float32) + 20.0

            result = env.step_chunk(left_action)

            pose = np.asarray(result["obs"]["state/pose"], dtype=np.float32)
            np.testing.assert_allclose(pose[:7], left_action)
            np.testing.assert_allclose(
                pose[7:],
                np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
            )
        finally:
            env.close()

    def test_dual_arm_step_chunk_accepts_14d(self) -> None:
        env = _make_fake_env(arm_layout="dual_arm", action_dim=14)
        try:
            env.reset()
            action = np.arange(14, dtype=np.float32)

            result = env.step_chunk(action)

            np.testing.assert_allclose(result["obs"]["state/pose"], action)
        finally:
            env.close()

    def test_dual_arm_rejects_7d_action(self) -> None:
        env = _make_fake_env(arm_layout="dual_arm", action_dim=14)
        try:
            env.reset()

            with self.assertRaisesRegex(ValueError, "Flat action chunk size"):
                env.step_chunk(np.ones(7, dtype=np.float32))
        finally:
            env.close()

    def test_single_arm_rejects_14d_chunk(self) -> None:
        env = _make_fake_env(arm_layout="right_arm", action_dim=7)
        try:
            env.reset()

            with self.assertRaisesRegex(ValueError, "Unexpected action chunk shape"):
                env.step_chunk(np.ones((1, 14), dtype=np.float32))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
