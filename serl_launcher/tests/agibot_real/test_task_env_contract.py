from __future__ import annotations

import sys
from pathlib import Path
import unittest

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.env.controller import TERMINAL_FAIL
from serl_torch.examples.agibot_real.env.controller import TERMINAL_HOOK
from serl_torch.examples.agibot_real.env.controller import TERMINAL_RESET
from serl_torch.examples.agibot_real.env.controller import TERMINAL_SUCCESS
from serl_torch.examples.agibot_real.env.controller import TERMINAL_TIMEOUT
from serl_torch.examples.agibot_real.env.task_env import AgiBotTaskEnv


class _FakeController:
    def __init__(self, *, terminal_signal: str | None, terminal_info: dict[str, object]):
        self._terminal_signal = terminal_signal
        self._terminal_info = dict(terminal_info)

    def get_meta(self) -> dict[str, object]:
        return {
            "terminal_signal": self._terminal_signal,
            "terminal_info": dict(self._terminal_info),
        }


def _make_env(*, terminal_signal: str | None = None) -> AgiBotTaskEnv:
    env = AgiBotTaskEnv.__new__(AgiBotTaskEnv)
    env._controller = _FakeController(
        terminal_signal=terminal_signal,
        terminal_info={"operator_note": "test"},
    )
    env._controller_enabled = True
    env._task_description = "test task"
    env.task_name = "agibot_real_default"
    env._take_action_cnt = 0
    env._step_limit = 5
    env._controller_cfg = {"poll_interval_sec": 0.05}
    env._last_obs = {"dummy": 1}
    env._success_hook_spec = "fake:success_hook"
    return env


class AgiBotTaskEnvContractTest(unittest.TestCase):
    def test_controller_success_drives_success_semantics(self) -> None:
        env = _make_env(terminal_signal=TERMINAL_SUCCESS)

        result = env._resolve_step_result(obs={}, action=[], controller_mode=True)

        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["done"])
        self.assertFalse(result["truncated"])
        self.assertTrue(result["success"])
        self.assertEqual(result["info"]["controller_terminal_signal"], TERMINAL_SUCCESS)

    def test_controller_fail_drives_failure_semantics(self) -> None:
        env = _make_env(terminal_signal=TERMINAL_FAIL)

        result = env._resolve_step_result(obs={}, action=[], controller_mode=True)

        self.assertEqual(result["reward"], 0.0)
        self.assertTrue(result["done"])
        self.assertFalse(result["truncated"])
        self.assertFalse(result["success"])
        self.assertEqual(result["info"]["controller_terminal_signal"], TERMINAL_FAIL)

    def test_controller_reset_drives_truncated_semantics(self) -> None:
        env = _make_env(terminal_signal=TERMINAL_RESET)

        result = env._resolve_step_result(obs={}, action=[], controller_mode=True)

        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(result["done"])
        self.assertTrue(result["truncated"])
        self.assertFalse(result["success"])
        self.assertEqual(result["info"]["controller_terminal_signal"], TERMINAL_RESET)

    def test_step_limit_timeout_is_only_non_controller_terminal(self) -> None:
        env = _make_env(terminal_signal=None)
        env._controller = None
        env._take_action_cnt = env._step_limit

        result = env._resolve_step_result(obs={}, action=[], controller_mode=False)

        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(result["done"])
        self.assertTrue(result["truncated"])
        self.assertFalse(result["success"])
        self.assertEqual(result["info"]["controller_terminal_signal"], TERMINAL_TIMEOUT)
        self.assertTrue(result["info"]["time_limit_reached"])

    def test_success_hook_spec_does_not_affect_canonical_result(self) -> None:
        env = _make_env(terminal_signal=None)
        env._controller = None
        env._take_action_cnt = 0

        result = env._resolve_step_result(obs={}, action=[], controller_mode=False)

        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(result["done"])
        self.assertFalse(result["truncated"])
        self.assertFalse(result["success"])

    def test_late_terminal_override_keeps_controller_terminal_signal(self) -> None:
        env = _make_env(terminal_signal=None)
        transition = {
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "info": {"success": False},
        }

        overridden = env._override_transition_for_terminal_meta(
            transition,
            {
                "terminal_signal": TERMINAL_SUCCESS,
                "terminal_info": {"operator_note": "late success"},
            },
        )
        self.assertEqual(
            overridden["info"]["controller_terminal_signal"],
            TERMINAL_SUCCESS,
        )
        self.assertTrue(overridden["info"]["human_success"])
        self.assertTrue(overridden["info"]["success"])


if __name__ == "__main__":
    unittest.main()
