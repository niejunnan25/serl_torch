from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch.nn as nn


REPO_PARENT = Path(__file__).resolve().parents[3]
SERL_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_PARENT, SERL_LAUNCHER_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serl_launcher.common.agent_acceleration import apply_torch_compile


def _compile_cfg(*, enabled: bool, target: str) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=bool(enabled),
        target=str(target),
        backend="inductor",
        mode="default",
        fullgraph=True,
        dynamic=False,
    )


def _fake_agent() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            modules={
                "actor": nn.Linear(2, 2),
                "critic": nn.Linear(2, 2),
            },
            target_modules={
                "critic": nn.Linear(2, 2),
            },
        )
    )


class AgentAccelerationTest(unittest.TestCase):
    def test_disabled_compile_keeps_agent_unchanged(self) -> None:
        agent = _fake_agent()
        original_actor = agent.state.modules["actor"]
        original_critic = agent.state.modules["critic"]
        original_target_critic = agent.state.target_modules["critic"]

        with mock.patch(
            "serl_launcher.common.agent_acceleration.torch.compile",
            create=True,
        ) as compile_mock:
            compiled = apply_torch_compile(
                agent,
                compile_cfg=_compile_cfg(enabled=False, target="actor_critic"),
            )

        self.assertIs(compiled, agent)
        self.assertIs(agent.state.modules["actor"], original_actor)
        self.assertIs(agent.state.modules["critic"], original_critic)
        self.assertIs(agent.state.target_modules["critic"], original_target_critic)
        compile_mock.assert_not_called()

    def test_actor_critic_target_compiles_actor_and_both_critics(self) -> None:
        agent = _fake_agent()

        def _compile_side_effect(module, **kwargs):
            return {
                "module": module,
                "kwargs": dict(kwargs),
            }

        with mock.patch(
            "serl_launcher.common.agent_acceleration.torch.compile",
            side_effect=_compile_side_effect,
            create=True,
        ) as compile_mock:
            apply_torch_compile(
                agent,
                compile_cfg=_compile_cfg(enabled=True, target="actor_critic"),
            )

        self.assertEqual(compile_mock.call_count, 3)
        self.assertEqual(agent.state.modules["critic"]["kwargs"]["backend"], "inductor")
        self.assertEqual(agent.state.modules["actor"]["kwargs"]["mode"], "default")
        self.assertTrue(agent.state.target_modules["critic"]["kwargs"]["fullgraph"])

    def test_critic_target_skips_actor_compile(self) -> None:
        agent = _fake_agent()
        original_actor = agent.state.modules["actor"]

        with mock.patch(
            "serl_launcher.common.agent_acceleration.torch.compile",
            side_effect=lambda module, **kwargs: ("compiled", module, dict(kwargs)),
            create=True,
        ) as compile_mock:
            apply_torch_compile(
                agent,
                compile_cfg=_compile_cfg(enabled=True, target="critic"),
            )

        self.assertEqual(compile_mock.call_count, 2)
        self.assertIs(agent.state.modules["actor"], original_actor)
        self.assertEqual(agent.state.modules["critic"][0], "compiled")
        self.assertEqual(agent.state.target_modules["critic"][0], "compiled")

    def test_invalid_target_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            apply_torch_compile(
                _fake_agent(),
                compile_cfg=_compile_cfg(enabled=True, target="actor"),
            )


if __name__ == "__main__":
    unittest.main()
