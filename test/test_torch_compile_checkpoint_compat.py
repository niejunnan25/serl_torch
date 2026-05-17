from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.common.checkpoint_codec import snapshot_actor_network_payload
from serl_launcher.common.checkpoint_codec import snapshot_agent_checkpoint_payload
from serl_launcher.common.common import TorchRLTrainState
from serl_launcher.common.torch_module_compat import module_state_dict


def _build_module(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    module = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(
                parameter
            )
            parameter.copy_(values + float(seed) + float(index) * 0.01)
    return module


def _make_agent(
    *,
    compile_actor: bool = False,
    compile_critic: bool = False,
    compile_target_critic: bool = False,
) -> SimpleNamespace:
    state = TorchRLTrainState(
        modules={
            "actor": _build_module(seed=11),
            "critic": _build_module(seed=22),
        },
        optimizers={},
        target_modules={
            "critic": _build_module(seed=33),
        },
    )
    if compile_actor:
        state.modules["actor"] = torch.compile(state.modules["actor"], backend="eager")
    if compile_critic:
        state.modules["critic"] = torch.compile(state.modules["critic"], backend="eager")
    if compile_target_critic:
        state.target_modules["critic"] = torch.compile(
            state.target_modules["critic"],
            backend="eager",
        )
    return SimpleNamespace(state=state)


def _state_dict_equals(lhs: nn.Module, rhs: nn.Module) -> bool:
    lhs_state = module_state_dict(lhs)
    rhs_state = module_state_dict(rhs)
    if list(lhs_state.keys()) != list(rhs_state.keys()):
        return False
    return all(torch.equal(lhs_state[key], rhs_state[key]) for key in lhs_state.keys())


@unittest.skipIf(not hasattr(torch, "compile"), "torch.compile is unavailable")
class TorchCompileCheckpointCompatTest(unittest.TestCase):
    def test_actor_snapshot_from_compiled_agent_uses_plain_keys(self) -> None:
        compiled_agent = _make_agent(compile_actor=True)
        plain_agent = _make_agent()

        payload = snapshot_actor_network_payload(compiled_agent, step=7)

        self.assertEqual(int(payload["step"]), 7)
        self.assertTrue(
            all(not key.startswith("_orig_mod.") for key in payload["params"]["actor"])
        )

        apply_checkpoint_payload_to_agent(plain_agent, payload, load_optimizers=False)

        self.assertTrue(
            _state_dict_equals(
                compiled_agent.state.modules["actor"],
                plain_agent.state.modules["actor"],
            )
        )

    def test_full_checkpoint_from_compiled_agent_loads_into_plain_agent(self) -> None:
        compiled_agent = _make_agent(
            compile_actor=True,
            compile_critic=True,
            compile_target_critic=True,
        )
        plain_agent = _make_agent()

        payload = snapshot_agent_checkpoint_payload(compiled_agent, step=13)

        self.assertTrue(
            all(
                not key.startswith("_orig_mod.")
                for state_dict in payload["params"].values()
                for key in state_dict.keys()
            )
        )
        self.assertTrue(
            all(
                not key.startswith("_orig_mod.")
                for state_dict in payload["target_params"].values()
                for key in state_dict.keys()
            )
        )

        apply_checkpoint_payload_to_agent(plain_agent, payload, load_optimizers=False)

        self.assertTrue(
            _state_dict_equals(
                compiled_agent.state.modules["actor"],
                plain_agent.state.modules["actor"],
            )
        )
        self.assertTrue(
            _state_dict_equals(
                compiled_agent.state.modules["critic"],
                plain_agent.state.modules["critic"],
            )
        )
        self.assertTrue(
            _state_dict_equals(
                compiled_agent.state.target_modules["critic"],
                plain_agent.state.target_modules["critic"],
            )
        )

    def test_plain_checkpoint_loads_into_compiled_agent(self) -> None:
        plain_agent = _make_agent()
        compiled_agent = _make_agent(
            compile_actor=True,
            compile_critic=True,
            compile_target_critic=True,
        )

        payload = snapshot_agent_checkpoint_payload(plain_agent, step=29)
        apply_checkpoint_payload_to_agent(compiled_agent, payload, load_optimizers=False)

        self.assertTrue(
            _state_dict_equals(
                plain_agent.state.modules["actor"],
                compiled_agent.state.modules["actor"],
            )
        )
        self.assertTrue(
            _state_dict_equals(
                plain_agent.state.modules["critic"],
                compiled_agent.state.modules["critic"],
            )
        )
        self.assertTrue(
            _state_dict_equals(
                plain_agent.state.target_modules["critic"],
                compiled_agent.state.target_modules["critic"],
            )
        )

    def test_legacy_prefixed_payload_loads_into_plain_agent(self) -> None:
        compiled_agent = _make_agent(compile_actor=True)
        plain_agent = _make_agent()

        legacy_payload = {
            "step": 41,
            "params": {
                "actor": compiled_agent.state.modules["actor"].state_dict(),
            },
        }
        self.assertTrue(
            any(
                key.startswith("_orig_mod.")
                for key in legacy_payload["params"]["actor"].keys()
            )
        )

        apply_checkpoint_payload_to_agent(
            plain_agent,
            legacy_payload,
            load_optimizers=False,
        )

        self.assertTrue(
            _state_dict_equals(
                compiled_agent.state.modules["actor"],
                plain_agent.state.modules["actor"],
            )
        )

    def test_train_state_params_round_trip_handles_compiled_modules(self) -> None:
        plain_agent = _make_agent()
        compiled_agent = _make_agent(
            compile_actor=True,
            compile_critic=True,
            compile_target_critic=True,
        )

        params = plain_agent.state.params
        target_params = plain_agent.state.target_params

        self.assertTrue(
            all(
                not key.startswith("_orig_mod.")
                for state_dict in params.values()
                for key in state_dict.keys()
            )
        )
        self.assertTrue(
            all(
                not key.startswith("_orig_mod.")
                for state_dict in target_params.values()
                for key in state_dict.keys()
            )
        )

        compiled_agent.state.params = params
        compiled_agent.state.target_params = target_params

        self.assertTrue(
            _state_dict_equals(
                plain_agent.state.modules["actor"],
                compiled_agent.state.modules["actor"],
            )
        )
        self.assertTrue(
            _state_dict_equals(
                plain_agent.state.modules["critic"],
                compiled_agent.state.modules["critic"],
            )
        )
        self.assertTrue(
            _state_dict_equals(
                plain_agent.state.target_modules["critic"],
                compiled_agent.state.target_modules["critic"],
            )
        )


if __name__ == "__main__":
    unittest.main()
