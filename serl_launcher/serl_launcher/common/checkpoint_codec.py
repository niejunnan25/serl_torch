"""Checkpoint payload codec helpers for Torch train states and agents."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch


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
            name: _clone_to_cpu_tree(module.state_dict())
            for name, module in train_state.modules.items()
        },
        "target_params": {
            name: _clone_to_cpu_tree(module.state_dict())
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
            train_state.modules[name].load_state_dict(state_dict, strict=True)
    for name, state_dict in dict(payload.get("target_params", {})).items():
        if name in train_state.target_modules:
            train_state.target_modules[name].load_state_dict(state_dict, strict=True)
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
