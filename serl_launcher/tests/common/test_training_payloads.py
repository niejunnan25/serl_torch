from __future__ import annotations

import unittest

import numpy as np

from serl_launcher.common.training_payloads import build_rollout_payload
from serl_launcher.common.training_payloads import build_rollout_stats_payload
from serl_launcher.common.training_payloads import parse_rollout_stats_payload


class TrainingPayloadsTest(unittest.TestCase):
    def test_build_and_parse_rollout_stats_payload(self) -> None:
        rollout = build_rollout_payload(
            episode_id=7,
            episode_steps=31,
            episode_return=4.5,
            init_episode_idx=2,
            success=True,
            cumulative_success_rate=0.4,
            recent_success_rate_20=0.6,
        )
        payload = build_rollout_stats_payload(
            env_steps=123,
            rollout=rollout,
            env_info={
                "success": True,
                "reward_trace": np.asarray([1.0, 2.0], dtype=np.float32),
            },
        )

        self.assertEqual(payload["env_steps"], 123)
        self.assertEqual(payload["rollout"]["episode_id"], 7)
        self.assertEqual(payload["env_info"]["reward_trace"], [1.0, 2.0])

        parsed = parse_rollout_stats_payload(payload)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["env_steps"], 123)
        self.assertEqual(parsed["rollout"]["episode_steps"], 31)
        self.assertTrue(parsed["rollout"]["success"])
        self.assertEqual(parsed["env_info"]["reward_trace"], [1.0, 2.0])

    def test_parse_invalid_rollout_payload_returns_none(self) -> None:
        payload = {
            "env_steps": 10,
            "rollout": {
                "episode_id": "bad",
            },
        }
        self.assertIsNone(parse_rollout_stats_payload(payload))


if __name__ == "__main__":
    unittest.main()
