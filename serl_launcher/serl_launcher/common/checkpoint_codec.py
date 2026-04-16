"""Checkpoint payload codec helpers for Torch train states and agents."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from serl_launcher.common.torch_module_compat import load_module_state_dict
from serl_launcher.common.torch_module_compat import module_state_dict


def _clone_to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu_tree(item) for item in value)
    return value


def snapshot_torch_train_state_payload(
    train_state: Any,
    *,
    step: int,
) -> Dict[str, Any]:
    return {
        "step": int(step),
        "params": {
            name: _clone_to_cpu_tree(module_state_dict(module))
            for name, module in train_state.modules.items()
        },
        "target_params": {
            name: _clone_to_cpu_tree(module_state_dict(module))
            for name, module in train_state.target_modules.items()
        },
        "optimizer": {
            name: _clone_to_cpu_tree(optimizer.state_dict())
            for name, optimizer in train_state.optimizers.items()
        },
    }


def apply_checkpoint_payload_to_train_state(
    train_state: Any,
    payload: Mapping[str, Any],
    *,
    load_optimizers: bool = False,
) -> None:
    for name, state_dict in dict(payload.get("params", {})).items():
        if name in train_state.modules:
            load_module_state_dict(train_state.modules[name], state_dict, strict=True)
    for name, state_dict in dict(payload.get("target_params", {})).items():
        if name in train_state.target_modules:
            load_module_state_dict(
                train_state.target_modules[name],
                state_dict,
                strict=True,
            )
    if load_optimizers:
        for name, optimizer_state in dict(payload.get("optimizer", {})).items():
            if name in train_state.optimizers:
                train_state.optimizers[name].load_state_dict(optimizer_state)
    if "step" in payload:
        train_state.step = int(payload["step"])


def snapshot_agent_checkpoint_payload(
    agent: Any,
    *,
    step: int,
) -> Dict[str, Any]:
    return snapshot_torch_train_state_payload(agent.state, step=int(step))


def snapshot_actor_network_payload(
    agent: Any,
    *,
    step: int,
) -> Dict[str, Any]:
    """Build a lightweight learner->actor payload containing actor weights only."""

    actor_module = agent.state.modules.get("actor")
    if actor_module is None:
        raise KeyError("Cannot snapshot actor network payload: missing 'actor' module")
    return {
        "step": int(step),
        "params": {
            "actor": _clone_to_cpu_tree(module_state_dict(actor_module)),
        },
    }


def apply_checkpoint_payload_to_agent(
    agent: Any,
    payload: Mapping[str, Any],
    *,
    load_optimizers: bool = False,
) -> None:
    apply_checkpoint_payload_to_train_state(
        agent.state,
        payload,
        load_optimizers=bool(load_optimizers),
    )
