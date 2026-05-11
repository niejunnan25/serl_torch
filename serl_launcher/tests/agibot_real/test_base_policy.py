from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(SERL_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERL_LAUNCHER_ROOT))

from serl_torch.examples.agibot_real.env.base_policy import AgiBotBasePolicy
from serl_torch.examples.agibot_real.env.base_policy import POLICY_ACTION_LAYOUT_RIGHT_ARM
from serl_torch.examples.agibot_real.env.observation import AGIBOT_RIGHT_ARM_STATE_SLICE
from serl_torch.examples.agibot_real.env.policy_input import build_agibot_policy_input


def _make_raw_obs() -> dict[str, object]:
    return {
        "state/pose": [float(value) for value in range(14)],
        "image/head": [[[0, 0, 0]] * 8 for _ in range(8)],
        "image/left_wrist": [[[1, 1, 1]] * 8 for _ in range(8)],
        "image/right_wrist": [[[2, 2, 2]] * 8 for _ in range(8)],
    }


class _FakeRightArmClient:
    def __init__(self, raw_actions: np.ndarray) -> None:
        self.raw_actions = np.asarray(raw_actions, dtype=np.float32)
        self.seen_state: np.ndarray | None = None
        self.seen_states: list[np.ndarray] = []

    def infer(self, policy_input):
        self.seen_state = np.asarray(policy_input.state, dtype=np.float32)
        return self.raw_actions, {"fake": True}

    def infer_many(self, policy_inputs):
        self.seen_states = [
            np.asarray(policy_input.state, dtype=np.float32)
            for policy_input in policy_inputs
        ]
        return [self.raw_actions for _policy_input in policy_inputs], {"fake_batch": True}


class AgiBotBasePolicyTest(unittest.TestCase):
    def test_build_policy_input_can_slice_right_arm_state(self) -> None:
        policy_input = build_agibot_policy_input(
            _make_raw_obs(),
            "pick",
            state_slice=AGIBOT_RIGHT_ARM_STATE_SLICE,
        )

        np.testing.assert_array_equal(
            np.asarray(policy_input.state),
            np.asarray([7, 8, 9, 10, 11, 12, 13], dtype=np.float32),
        )

    def test_right_arm_openpi_actions_are_embedded_in_canonical_14d_chunk(self) -> None:
        raw_right_actions = np.asarray(
            [
                [70, 71, 72, 73, 74, 75, 76],
                [80, 81, 82, 83, 84, 85, 86],
            ],
            dtype=np.float32,
        )
        client = _FakeRightArmClient(raw_right_actions)
        policy = AgiBotBasePolicy(
            _client=client,
            _backend_type="openpi",
            _description="openpi:right-arm",
            _action_dim=14,
            _chunk_horizon=2,
            _action_layout=POLICY_ACTION_LAYOUT_RIGHT_ARM,
        )

        chunk, info = policy.infer(_make_raw_obs(), prompt="pick")

        expected = np.asarray(
            [
                [0, 1, 2, 3, 4, 5, 6, 70, 71, 72, 73, 74, 75, 76],
                [0, 1, 2, 3, 4, 5, 6, 80, 81, 82, 83, 84, 85, 86],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(chunk, expected)
        np.testing.assert_array_equal(
            client.seen_state,
            np.asarray([7, 8, 9, 10, 11, 12, 13], dtype=np.float32),
        )
        self.assertEqual(info["action_layout"], POLICY_ACTION_LAYOUT_RIGHT_ARM)
        self.assertEqual(info["canonical_action_dim"], 14)
        self.assertEqual(info["raw_action_dim"], 7)

    def test_right_arm_openpi_infer_many_embeds_each_chunk(self) -> None:
        raw_right_actions = np.asarray(
            [
                [70, 71, 72, 73, 74, 75, 76],
                [80, 81, 82, 83, 84, 85, 86],
            ],
            dtype=np.float32,
        )
        client = _FakeRightArmClient(raw_right_actions)
        policy = AgiBotBasePolicy(
            _client=client,
            _backend_type="openpi",
            _description="openpi:right-arm",
            _action_dim=14,
            _chunk_horizon=2,
            _action_layout=POLICY_ACTION_LAYOUT_RIGHT_ARM,
        )

        chunks, info = policy.infer_many(
            [_make_raw_obs(), _make_raw_obs()],
            prompt=["pick", "place"],
        )

        expected = np.asarray(
            [
                [0, 1, 2, 3, 4, 5, 6, 70, 71, 72, 73, 74, 75, 76],
                [0, 1, 2, 3, 4, 5, 6, 80, 81, 82, 83, 84, 85, 86],
            ],
            dtype=np.float32,
        )
        self.assertEqual(len(chunks), 2)
        np.testing.assert_array_equal(chunks[0], expected)
        np.testing.assert_array_equal(chunks[1], expected)
        self.assertEqual(len(client.seen_states), 2)
        for seen_state in client.seen_states:
            np.testing.assert_array_equal(
                seen_state,
                np.asarray([7, 8, 9, 10, 11, 12, 13], dtype=np.float32),
            )
        self.assertEqual(info["action_layout"], POLICY_ACTION_LAYOUT_RIGHT_ARM)
        self.assertEqual(info["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
