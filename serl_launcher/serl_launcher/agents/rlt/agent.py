"""RLTAgent: bridges the openpi RLT algorithm to serl_torch's agent interface.

This agent wraps the RLT actor and critic ensemble, providing the
`sample_action`, `update_critics`, and `update_high_utd` methods expected
by the serl_torch training loop and checkpoint codec.

The RLT encoder is frozen and NOT part of this agent — it lives on the
actor/eval side and pre-computes z_rl before storing into the replay buffer.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from serl_launcher.agents.rlt.modeling import MLP, RLTActor


class RLTCritic(nn.Module):
    """Q-function over (state, action_chunk) pairs."""

    def __init__(self, state_dim: int, action_chunk_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(state_dim + action_chunk_dim, hidden_dims, output_dim=1)

    def forward(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_chunk], dim=-1)
        return self.net(x)


# ── Train State (checkpoint codec compatible) ────────────────────────────


@dataclass
class RLTTrainState:
    """Minimal train state compatible with serl_torch checkpoint codec.

    The codec expects:
      - state.modules: dict[str, nn.Module]
      - state.target_modules: dict[str, nn.Module]
      - state.optimizers: dict[str, Optimizer]
      - state.step: int
    """

    modules: dict[str, nn.Module] = field(default_factory=dict)
    target_modules: dict[str, nn.Module] = field(default_factory=dict)
    optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    step: int = 0


# ── Agent ────────────────────────────────────────────────────────────────


class RLTAgent:
    """RLT Stage 2 agent for serl_torch async training.

    Owns the actor, critic ensemble, target critics, and optimizers.
    Provides the interface expected by the serl_torch training loop.
    """

    def __init__(
        self,
        *,
        z_rl_dim: int = 2048,
        proprio_dim: int = 8,
        action_dim: int = 7,
        chunk_size: int = 10,
        execute_horizon: int = 5,
        actor_hidden_dims: tuple[int, ...] = (512, 512, 512),
        critic_hidden_dims: tuple[int, ...] = (512, 512, 512),
        actor_std: float = 0.01,
        num_critics: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        bc_reg_coeff: float = 4.0,
        ref_dropout: float = 0.5,
        clip_grad_norm: float = 10.0,
        policy_update_freq: int = 2,
        device: str = "cuda",
    ):
        if execute_horizon <= 0 or execute_horizon > chunk_size:
            raise ValueError(
                "execute_horizon must be in [1, chunk_size], "
                f"got execute_horizon={execute_horizon}, chunk_size={chunk_size}"
            )
        self._device = torch.device(device)
        self._discount = discount
        self._tau = tau
        self._bc_reg_coeff = bc_reg_coeff
        self._ref_dropout = ref_dropout
        self._clip_grad_norm = clip_grad_norm
        self._policy_update_freq = policy_update_freq
        self._chunk_size = chunk_size
        self._execute_horizon = execute_horizon
        self._action_dim = action_dim
        self._optimization_step = 0
        self._critic_step_count = 0

        # Match the source RLT Stage 2 implementation: proprio is present in the
        # replay schema but the actor/critic state uses the RL token only.
        state_dim = z_rl_dim
        action_chunk_dim = chunk_size * action_dim

        self.actor = RLTActor(
            state_dim=state_dim,
            action_chunk_dim=action_chunk_dim,
            hidden_dims=list(actor_hidden_dims),
            std=actor_std,
        ).to(self._device)

        self.critics = nn.ModuleList(
            [
                RLTCritic(state_dim, action_chunk_dim, list(critic_hidden_dims))
                for _ in range(num_critics)
            ]
        ).to(self._device)

        self.critic_targets = nn.ModuleList(
            [copy.deepcopy(c) for c in self.critics]
        ).to(self._device)
        for ct in self.critic_targets:
            ct.requires_grad_(False)

        self._actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self._critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)

        # Checkpoint-codec-compatible state
        self.state = RLTTrainState(
            modules={
                "actor": self.actor,
                "critics": self.critics,
            },
            target_modules={
                "critics": self.critic_targets,
            },
            optimizers={
                "actor": self._actor_optimizer,
                "critic": self._critic_optimizer,
            },
        )

        # Config dict (for compatibility with serl_torch patterns)
        self.config = {"image_keys": ()}

    @torch.no_grad()
    def sample_action(
        self,
        obs: dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action from observation dict.

        Args:
            obs: dict with keys "z_rl", "proprio", "reference_action" (no batch dim)
            deterministic: if True, return mean action (no noise)

        Returns:
            Flattened action chunk as numpy array (chunk_size * action_dim,)
        """
        self.actor.eval()

        z_rl = torch.from_numpy(obs["z_rl"]).float().unsqueeze(0).to(self._device)
        ref_action = torch.from_numpy(obs["reference_action"]).float().unsqueeze(0).to(self._device)

        state = torch.cat([z_rl], dim=-1)

        if deterministic:
            action = self.actor(state, ref_action)
        else:
            action, _ = self.actor.sample(state, ref_action)

        return action.squeeze(0).cpu().numpy()

    def update_critics(self, batch: dict[str, Any]) -> tuple["RLTAgent", dict[str, float]]:
        """Single critic update step."""
        fb = self._convert_batch(batch)
        critic_loss, target_q_mean, predicted_q_mean = self._critic_step(fb)
        self._update_target_networks()
        return self, {
            "loss_critic": critic_loss,
            "target_q_mean": target_q_mean,
            "predicted_q_mean": predicted_q_mean,
        }

    def update_high_utd(self, batch, utd_ratio=5):
        info: dict[str, float] = {}
        fb = self._convert_batch(batch)
        for i in range(utd_ratio):
            critic_loss, target_q_mean, predicted_q_mean = self._critic_step(fb)
            info["loss_critic"] = critic_loss
            info["target_q_mean"] = target_q_mean
            info["predicted_q_mean"] = predicted_q_mean

            self._update_target_networks()

            if self._critic_step_count % self._policy_update_freq == 0:
                actor_info = self._actor_step(fb)
                info.update(actor_info)
            self._critic_step_count += 1
        return self, info

    def _convert_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert serl_torch replay batch to RLT forward batch format."""
        device = self._device

        z_rl = torch.as_tensor(batch["observations"]["z_rl"], dtype=torch.float32).to(device)
        state = torch.cat([z_rl], dim=-1)

        next_z_rl = torch.as_tensor(batch["next_observations"]["z_rl"], dtype=torch.float32).to(device)
        next_state = torch.cat([next_z_rl], dim=-1)

        ref_action = torch.as_tensor(batch["observations"]["reference_action"], dtype=torch.float32).to(device)
        next_ref_action = torch.as_tensor(batch["next_observations"]["reference_action"], dtype=torch.float32).to(device)

        action = torch.as_tensor(batch["actions"], dtype=torch.float32).to(device)
        reward = torch.as_tensor(batch["rewards"], dtype=torch.float32).to(device)
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        done = torch.as_tensor(batch["dones"], dtype=torch.float32).to(device)
        if done.dim() == 1:
            done = done.unsqueeze(-1)
        if "discounts" in batch:
            discount = torch.as_tensor(batch["discounts"], dtype=torch.float32).to(device)
        else:
            discount = torch.full_like(reward, float(self._discount ** self._execute_horizon))
        if discount.dim() == 1:
            discount = discount.unsqueeze(-1)

        return {
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done,
            "discount": discount,
            "reference_action": ref_action,
            "next_reference_action": next_ref_action,
        }

    def _critic_step(self, fb: dict[str, torch.Tensor]) -> tuple[float, float, float]:
        """Chunked TD with clipped double-Q target (Paper Eq. 3)."""
        state = fb["state"]
        next_state = fb["next_state"]
        action = fb["action"]
        reward = fb["reward"]
        done = fb["done"]
        discount = fb["discount"]

        with torch.no_grad():
            next_ref = fb["next_reference_action"]
            next_action = self.actor(next_state, next_ref)

            target_qs = [ct(next_state, next_action) for ct in self.critic_targets]
            min_target_q = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True).values

            td_target = reward + (1.0 - done) * discount * min_target_q

        q_preds = [c(state, action) for c in self.critics]
        loss = sum(F.mse_loss(q, td_target) for q in q_preds)

        self._critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=self._clip_grad_norm)
        self._critic_optimizer.step()
        target_q_mean = td_target.mean().item()
        predicted_q_mean = torch.stack([q.mean() for q in q_preds]).mean().item()
        return loss.item(), target_q_mean, predicted_q_mean

    def _actor_step(self, fb: dict[str, torch.Tensor]) -> tuple[float, float, float]:
        """Maximize Q while staying near VLA reference (Paper Eq. 5)."""
        state = fb["state"]
        ref = fb["reference_action"]

        mask = (torch.rand(ref.shape[0], 1, device=self._device) > self._ref_dropout).float()
        ref_input = ref * mask

        action = self.actor(state, ref_input)
        q_value = self.critics[0](state, action)
        bc_loss = F.mse_loss(action, ref)

        loss = -q_value.mean() + self._bc_reg_coeff * bc_loss

        self._actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self._clip_grad_norm)
        self._actor_optimizer.step()

        return {"loss_actor": loss.item(), "bc_loss": bc_loss.item(), "q_value_mean": q_value.mean().item()}

    def _update_target_networks(self) -> None:
        """Soft update target critics with tau."""
        for critic, target in zip(self.critics, self.critic_targets):
            for p, tp in zip(critic.parameters(), target.parameters()):
                tp.data.mul_(1 - self._tau).add_(p.data, alpha=self._tau)


def create_rlt_agent_from_cfg(cfg: Any) -> RLTAgent:
    """Factory: create RLTAgent from a LiberoRLTTrainConfig."""
    rlt = cfg.rlt
    return RLTAgent(
        z_rl_dim=rlt.z_rl_dim,
        proprio_dim=rlt.proprio_dim,
        action_dim=rlt.action_dim,
        chunk_size=rlt.chunk_size,
        execute_horizon=rlt.execute_horizon,
        actor_hidden_dims=rlt.actor_hidden_dims,
        critic_hidden_dims=rlt.critic_hidden_dims,
        actor_std=rlt.actor_std,
        num_critics=rlt.num_critics,
        actor_lr=rlt.actor_lr,
        critic_lr=rlt.critic_lr,
        discount=rlt.discount,
        tau=rlt.tau,
        bc_reg_coeff=rlt.bc_reg_coeff,
        ref_dropout=rlt.ref_dropout,
        clip_grad_norm=rlt.clip_grad_norm,
        policy_update_freq=rlt.policy_update_freq,
        device=rlt.device,
    )
