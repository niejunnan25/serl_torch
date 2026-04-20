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

from serl_torch.examples.libero.rlpd.runtime import require_eval_checkpoint
from serl_torch.examples.libero.rlpd.runtime import sample_actor_action


class LiberoRLPDRuntimeTest(unittest.TestCase):
    def test_sample_actor_action_uses_random_warmup_without_calling_policy(self) -> None:
        policy_calls = 0

        def _policy_action() -> np.ndarray:
            nonlocal policy_calls
            policy_calls += 1
            return np.asarray([9.0, 9.0], dtype=np.float32)

        action, used_random_action = sample_actor_action(
            policy_action_fn=_policy_action,
            env_steps=5,
            random_steps=10,
            action_dim=7,
        )

        self.assertTrue(used_random_action)
        self.assertEqual(policy_calls, 0)
        self.assertEqual(action.shape, (7,))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_sample_actor_action_switches_to_policy_after_random_warmup(self) -> None:
        policy_calls = 0

        def _policy_action() -> np.ndarray:
            nonlocal policy_calls
            policy_calls += 1
            return np.asarray([0.25, -0.5, 0.75], dtype=np.float32)

        action, used_random_action = sample_actor_action(
            policy_action_fn=_policy_action,
            env_steps=10,
            random_steps=10,
            action_dim=3,
        )

        self.assertFalse(used_random_action)
        self.assertEqual(policy_calls, 1)
        np.testing.assert_allclose(
            action,
            np.asarray([0.25, -0.5, 0.75], dtype=np.float32),
        )

    def test_require_eval_checkpoint_rejects_missing_checkpoint_by_default(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires eval.checkpoint_path",
        ):
            require_eval_checkpoint(
                checkpoint_file=None,
                allow_random_policy=False,
            )

    def test_require_eval_checkpoint_allows_explicit_random_policy_debug(self) -> None:
        require_eval_checkpoint(
            checkpoint_file=None,
            allow_random_policy=True,
        )


if __name__ == "__main__":
    unittest.main()
