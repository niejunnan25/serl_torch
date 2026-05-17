from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.env.arm_layout import embed_logical_action
from serl_torch.examples.agibot_real.env.arm_layout import normalize_arm_layout
from serl_torch.examples.agibot_real.env.arm_layout import project_vector_to_layout


class AgiBotArmLayoutTest(unittest.TestCase):
    def test_projects_14d_state_by_layout(self) -> None:
        vector = np.arange(14, dtype=np.float32)

        np.testing.assert_array_equal(
            project_vector_to_layout(vector, "left_arm"),
            np.arange(0, 7, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            project_vector_to_layout(vector, "right_arm"),
            np.arange(7, 14, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            project_vector_to_layout(vector, "dual_arm"),
            vector,
        )

    def test_projects_30d_joyra_vector_from_last_14d(self) -> None:
        vector = np.arange(30, dtype=np.float32)

        np.testing.assert_array_equal(
            project_vector_to_layout(vector, "right_arm"),
            np.arange(23, 30, dtype=np.float32),
        )

    def test_rejects_7d_dual_arm_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "dual_arm"):
            project_vector_to_layout(np.zeros(7, dtype=np.float32), "dual_arm")

    def test_embeds_single_arm_action_with_current_state_hold(self) -> None:
        current_state = np.arange(14, dtype=np.float32)
        action = np.asarray([70, 71, 72, 73, 74, 75, 76], dtype=np.float32)

        right_physical = embed_logical_action(action, current_state, "right_arm")
        np.testing.assert_array_equal(
            right_physical,
            np.asarray([0, 1, 2, 3, 4, 5, 6, 70, 71, 72, 73, 74, 75, 76], dtype=np.float32),
        )

        left_physical = embed_logical_action(action, current_state, "left_arm")
        np.testing.assert_array_equal(
            left_physical,
            np.asarray([70, 71, 72, 73, 74, 75, 76, 7, 8, 9, 10, 11, 12, 13], dtype=np.float32),
        )

    def test_normalizes_legacy_full_alias_to_dual_arm(self) -> None:
        self.assertEqual(normalize_arm_layout("full"), "dual_arm")


if __name__ == "__main__":
    unittest.main()
