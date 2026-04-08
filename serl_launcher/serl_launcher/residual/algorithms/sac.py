"""SAC-backed residual algorithm implementation."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from serl_launcher.training.checkpoint import _snapshot_agent_checkpoint_payload
from serl_launcher.residual.runtime.config_utils import build_drq_agent


class ResidualSACAlgorithm:
    name = "sac"

    def _build_agent(
        self,
        cfg: Any,
        *,
        sample_obs: Dict[str, Any],
        action_dim: int,
        image_keys: tuple[str, ...],
        critic_action_dim: Optional[int] = None,
        action_transform: Optional[Dict[str, Any]] = None,
        device: Any = None,
    ) -> Any:
        return build_drq_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=int(action_dim),
            image_keys=tuple(image_keys),
            critic_action_dim=critic_action_dim,
            action_transform=action_transform,
            device=device,
        )

    def build_actor_agent(
        self,
        cfg: Any,
        *,
        sample_obs: Dict[str, Any],
        action_dim: int,
        image_keys: tuple[str, ...],
        critic_action_dim: Optional[int] = None,
        action_transform: Optional[Dict[str, Any]] = None,
        device: Any = None,
    ) -> Any:
        return self._build_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=action_dim,
            image_keys=image_keys,
            critic_action_dim=critic_action_dim,
            action_transform=action_transform,
            device=device,
        )

    def build_learner_agent(
        self,
        cfg: Any,
        *,
        sample_obs: Dict[str, Any],
        action_dim: int,
        image_keys: tuple[str, ...],
        critic_action_dim: Optional[int] = None,
        action_transform: Optional[Dict[str, Any]] = None,
        device: Any = None,
    ) -> Any:
        return self._build_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=action_dim,
            image_keys=image_keys,
            critic_action_dim=critic_action_dim,
            action_transform=action_transform,
            device=device,
        )

    def sample_actions(
        self,
        agent: Any,
        obs_input: Dict[str, Any],
        *,
        deterministic: bool = False,
    ) -> Any:
        return agent.sample_actions(obs_input, deterministic=bool(deterministic))

    def update_high_utd(
        self,
        agent: Any,
        batch: Dict[str, Any],
        *,
        utd_ratio: int,
    ) -> tuple[Any, Dict[str, Any]]:
        return agent.update_high_utd(batch, utd_ratio=int(utd_ratio))

    @torch.no_grad()
    def sync_modules(self, target_agent: Any, source_agent: Any) -> None:
        for name, source_module in source_agent.state.modules.items():
            if name in target_agent.state.modules:
                target_agent.state.modules[name].load_state_dict(
                    source_module.state_dict(), strict=True
                )
        for name, source_module in source_agent.state.target_modules.items():
            if name in target_agent.state.target_modules:
                target_agent.state.target_modules[name].load_state_dict(
                    source_module.state_dict(), strict=True
                )
        target_agent.state.step = int(source_agent.state.step)

    @torch.no_grad()
    def apply_snapshot_payload(
        self,
        target_agent: Any,
        payload: Dict[str, Any],
        *,
        load_optimizers: bool = False,
    ) -> None:
        for name, state_dict in payload.get("params", {}).items():
            if name in target_agent.state.modules:
                target_agent.state.modules[name].load_state_dict(
                    state_dict, strict=True
                )
        for name, state_dict in payload.get("target_params", {}).items():
            if name in target_agent.state.target_modules:
                target_agent.state.target_modules[name].load_state_dict(
                    state_dict, strict=True
                )
        if load_optimizers:
            for name, opt_state in payload.get("optimizer", {}).items():
                if name in target_agent.state.optimizers:
                    target_agent.state.optimizers[name].load_state_dict(opt_state)
        target_agent.state.step = int(payload.get("step", target_agent.state.step))

    def snapshot_checkpoint_payload(
        self,
        agent: Any,
        *,
        step: int,
    ) -> Dict[str, Any]:
        return _snapshot_agent_checkpoint_payload(agent, step=int(step))
