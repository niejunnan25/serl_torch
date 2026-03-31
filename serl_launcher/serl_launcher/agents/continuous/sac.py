import copy
from typing import FrozenSet, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from serl_launcher.common.common import TorchRLTrainState, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data
from serl_launcher.networks.actor_critic_nets import Critic, CriticEnsemble, Policy
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP


def _to_torch(data, device: torch.device):
    if isinstance(data, dict):
        return {k: _to_torch(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        tensor = data.to(device)
    else:
        tensor = torch.as_tensor(data, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def _tree_mean(values):
    out = {}
    for key in values[0].keys():
        valid = [v[key] for v in values if key in v]
        if not valid:
            continue
        out[key] = float(np.mean(valid))
    return out


def _split_batch(batch: Batch, utd_ratio: int):
    def _split(x):
        b = x.shape[0]
        if b % utd_ratio != 0:
            raise ValueError(
                f"Batch size {b} must be divisible by utd_ratio {utd_ratio}"
            )
        mini_b = b // utd_ratio
        return x.reshape(utd_ratio, mini_b, *x.shape[1:])

    if isinstance(batch, dict):
        return {k: _split_batch(v, utd_ratio) for k, v in batch.items()}
    return _split(batch)


def _index_batch(batch, idx: int):
    if isinstance(batch, dict):
        return {k: _index_batch(v, idx) for k, v in batch.items()}
    return batch[idx]


class SACAgent:
    state: TorchRLTrainState
    config: dict = nonpytree_field(default_factory=dict)

    def __init__(self, state: TorchRLTrainState, config: dict):
        self.state = state
        self.config = config

    @property
    def device(self):
        return self.state.device

    def replace(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def forward_critic(
        self, observations: Data, actions: torch.Tensor, train: bool = True
    ):
        critic = self.state.modules["critic"]
        return critic(observations, actions, train=train)

    def forward_target_critic(self, observations: Data, actions: torch.Tensor):
        critic = self.state.target_modules["critic"]
        return critic(observations, actions, train=False)

    def forward_policy(self, observations: Data, train: bool = True):
        actor = self.state.modules["actor"]
        return actor(observations, train=train)

    def forward_temperature(self):
        temperature_module = self.state.modules["temperature"]
        return temperature_module()

    @staticmethod
    def _extract_aux_tensor(observations: Data, key: str) -> torch.Tensor:
        if not isinstance(observations, dict) or key not in observations:
            raise KeyError(
                f"Missing observation key required for action transform: {key}"
            )
        value = observations[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Observation key '{key}' must be a torch.Tensor, got {type(value)}"
            )
        if value.ndim >= 3 and value.shape[1] == 1:
            value = value[:, 0]
        return value

    @staticmethod
    def _apply_action_mask(
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_mask is None:
            return actions
        mask = action_mask.to(device=actions.device, dtype=actions.dtype)
        if actions.ndim == mask.ndim:
            return actions * mask
        if actions.ndim == (mask.ndim + 1):
            return actions * mask.unsqueeze(1)
        raise ValueError(
            f"Unsupported action/action_mask ranks: actions={tuple(actions.shape)} mask={tuple(mask.shape)}"
        )

    def _project_critic_action_mask_to_policy_space(
        self,
        action_mask: Optional[torch.Tensor],
        *,
        policy_action_dim: int,
    ) -> Optional[torch.Tensor]:
        if action_mask is None:
            return None

        if action_mask.shape[-1] == int(policy_action_dim):
            return action_mask

        transform_cfg = self.config.get("action_transform", None)
        if (
            transform_cfg is None
            or transform_cfg.get("type", None) != "residual_combined"
        ):
            raise ValueError(
                "action_mask dim does not match policy action dim and no residual action transform is configured: "
                f"mask_dim={action_mask.shape[-1]} policy_dim={policy_action_dim}"
            )

        control_indices = torch.as_tensor(
            transform_cfg["control_indices"],
            device=action_mask.device,
            dtype=torch.long,
        )
        full_action_dim = int(transform_cfg["full_action_dim"])
        chunk_horizon = int(transform_cfg.get("chunk_horizon", 1))
        chunk_step_enabled = bool(transform_cfg.get("chunk_step_enabled", False))

        if chunk_step_enabled:
            expected_critic_dim = int(chunk_horizon * full_action_dim)
            if int(action_mask.shape[-1]) != expected_critic_dim:
                raise ValueError(
                    "Unexpected chunk critic action_mask dim: "
                    f"{action_mask.shape[-1]} != {expected_critic_dim}"
                )
            projected = action_mask.reshape(
                *action_mask.shape[:-1], chunk_horizon, full_action_dim
            )
            projected = projected.index_select(dim=-1, index=control_indices)
            projected = projected.reshape(*projected.shape[:-2], -1)
        else:
            expected_critic_dim = int(full_action_dim)
            if int(action_mask.shape[-1]) != expected_critic_dim:
                raise ValueError(
                    "Unexpected step critic action_mask dim: "
                    f"{action_mask.shape[-1]} != {expected_critic_dim}"
                )
            projected = action_mask.index_select(dim=-1, index=control_indices)

        if int(projected.shape[-1]) != int(policy_action_dim):
            raise ValueError(
                "Projected policy action_mask dim mismatch: "
                f"{projected.shape[-1]} != {policy_action_dim}"
            )
        return projected

    @staticmethod
    def _reduce_log_prob_with_mask(
        log_prob_per_dim: torch.Tensor,
        policy_action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if policy_action_mask is None:
            return log_prob_per_dim.sum(dim=-1)

        mask = policy_action_mask.to(
            device=log_prob_per_dim.device, dtype=log_prob_per_dim.dtype
        )
        if log_prob_per_dim.ndim == mask.ndim + 1:
            mask = mask.unsqueeze(1)
        elif log_prob_per_dim.ndim != mask.ndim:
            raise ValueError(
                f"Unsupported log_prob/mask ranks: log_prob={tuple(log_prob_per_dim.shape)} "
                f"mask={tuple(mask.shape)}"
            )
        return (log_prob_per_dim * mask).sum(dim=-1)

    @staticmethod
    def _clip_gripper_last_dim(actions: torch.Tensor) -> torch.Tensor:
        """Clip only the last action dim without in-place writes."""
        if actions.shape[-1] <= 0:
            return actions
        clipped_last = torch.clamp(actions[..., -1:], -1.0, 1.0)
        if actions.shape[-1] == 1:
            return clipped_last
        return torch.cat([actions[..., :-1], clipped_last], dim=-1)

    def _transform_policy_actions_for_critic(
        self,
        observations: Data,
        policy_actions: torch.Tensor,
    ) -> torch.Tensor:
        transform_cfg = self.config.get("action_transform", None)
        if (
            transform_cfg is None
            or transform_cfg.get("type", None) != "residual_combined"
        ):
            return policy_actions

        control_indices = torch.as_tensor(
            transform_cfg["control_indices"],
            device=policy_actions.device,
            dtype=torch.long,
        )
        limits = torch.as_tensor(
            transform_cfg["limits"],
            device=policy_actions.device,
            dtype=policy_actions.dtype,
        )
        full_action_dim = int(transform_cfg["full_action_dim"])
        chunk_horizon = int(transform_cfg.get("chunk_horizon", 1))
        chunk_step_enabled = bool(transform_cfg.get("chunk_step_enabled", False))
        clip_gripper = bool(transform_cfg.get("clip_gripper", True))
        scale_key = str(transform_cfg.get("scale_key", "alpha"))
        raw_scale = self._extract_aux_tensor(observations, scale_key)

        scale = raw_scale.to(
            device=policy_actions.device,
            dtype=policy_actions.dtype,
        )
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1)
        if torch.any(scale < 0.0):
            min_scale = float(scale.min().detach().cpu().item())
            raise ValueError(
                f"Observation residual scale '{scale_key}' must be >= 0.0, got min={min_scale}"
            )

        if not chunk_step_enabled:
            base_key = str(transform_cfg.get("base_action_key", "base_action"))
            base_action = self._extract_aux_tensor(observations, base_key).to(
                device=policy_actions.device,
                dtype=policy_actions.dtype,
            )
            if base_action.shape[-1] != full_action_dim:
                raise ValueError(
                    f"Unexpected base action dim: {base_action.shape[-1]} != {full_action_dim}"
                )

            clipped = torch.clamp(policy_actions, -1.0, 1.0)
            if clipped.ndim == 2:
                delta = clipped * scale * limits.view(1, -1)
                final_action = base_action.clone()
                final_action[:, control_indices] = (
                    final_action[:, control_indices] + delta
                )
            elif clipped.ndim == 3:
                delta = clipped * scale.unsqueeze(1) * limits.view(1, 1, -1)
                final_action = (
                    base_action.unsqueeze(1).expand(-1, clipped.shape[1], -1).clone()
                )
                final_action[:, :, control_indices] = (
                    final_action[:, :, control_indices] + delta
                )
            else:
                raise ValueError(
                    f"Unsupported policy action rank for step transform: {clipped.shape}"
                )

            if clip_gripper and final_action.shape[-1] > 0:
                final_action = self._clip_gripper_last_dim(final_action)
            return final_action

        base_chunk_key = str(
            transform_cfg.get("base_action_chunk_key", "base_action_chunk")
        )
        base_chunk = self._extract_aux_tensor(observations, base_chunk_key).to(
            device=policy_actions.device,
            dtype=policy_actions.dtype,
        )
        if base_chunk.ndim != 3:
            raise ValueError(f"Unexpected base action chunk shape: {base_chunk.shape}")
        if (
            base_chunk.shape[1] != chunk_horizon
            or base_chunk.shape[2] != full_action_dim
        ):
            raise ValueError(
                "Unexpected base action chunk dims: "
                f"{tuple(base_chunk.shape)} vs (*, {chunk_horizon}, {full_action_dim})"
            )

        residual_action_dim = int(control_indices.numel())
        clipped = torch.clamp(policy_actions, -1.0, 1.0)
        if clipped.ndim == 2:
            residual_chunk = clipped.reshape(-1, chunk_horizon, residual_action_dim)
            delta = residual_chunk * scale.unsqueeze(1) * limits.view(1, 1, -1)
            final_chunk = base_chunk.clone()
            final_chunk[:, :, control_indices] = (
                final_chunk[:, :, control_indices] + delta
            )
            if clip_gripper and final_chunk.shape[-1] > 0:
                final_chunk = self._clip_gripper_last_dim(final_chunk)
            return final_chunk.reshape(-1, chunk_horizon * full_action_dim)

        if clipped.ndim == 3:
            residual_chunk = clipped.reshape(
                -1, clipped.shape[1], chunk_horizon, residual_action_dim
            )
            delta = (
                residual_chunk
                * scale.unsqueeze(1).unsqueeze(1)
                * limits.view(1, 1, 1, -1)
            )
            final_chunk = (
                base_chunk.unsqueeze(1).expand(-1, clipped.shape[1], -1, -1).clone()
            )
            final_chunk[:, :, :, control_indices] = (
                final_chunk[:, :, :, control_indices] + delta
            )
            if clip_gripper and final_chunk.shape[-1] > 0:
                final_chunk = self._clip_gripper_last_dim(final_chunk)
            return final_chunk.reshape(
                -1, clipped.shape[1], chunk_horizon * full_action_dim
            )

        raise ValueError(
            f"Unsupported policy action rank for chunk transform: {clipped.shape}"
        )

    def temperature_lagrange_penalty(self, entropy: torch.Tensor):
        return self.state.modules["temperature"](
            lhs=entropy,
            rhs=torch.as_tensor(
                self.config["target_entropy"], device=self.device, dtype=torch.float32
            ),
        )

    def _sample_policy_actions_for_critic(
        self,
        observations,
        *,
        num_samples: int = 1,
        train: bool = True,
        with_log_prob: bool = True,
    ):
        """
        从 residual actor 采样动作，并转换为 critic 语义下的动作。

        - num_samples=1: 返回 (B, A_critic), (B,) 或 None
        - num_samples>1: 返回 (B, K, A_critic), (B, K) 或 None
        """
        action_distributions = self.forward_policy(observations, train=train)
        k = max(1, int(num_samples))
        if k == 1:
            if with_log_prob:
                policy_actions, log_probs = action_distributions.sample_and_log_prob()
            else:
                policy_actions = action_distributions.sample()
                log_probs = None
            critic_actions = self._transform_policy_actions_for_critic(
                observations, policy_actions
            )
            return critic_actions, log_probs

        action_list = []
        log_probs_list = []
        for _ in range(k):
            if with_log_prob:
                (
                    sampled_actions,
                    sampled_log_probs,
                ) = action_distributions.sample_and_log_prob()
                log_probs_list.append(sampled_log_probs)
            else:
                sampled_actions = action_distributions.sample()
            action_list.append(sampled_actions)
        policy_actions = torch.stack(action_list, dim=1)
        critic_actions = self._transform_policy_actions_for_critic(
            observations, policy_actions
        )
        if with_log_prob:
            return critic_actions, torch.stack(log_probs_list, dim=1)
        return critic_actions, None

    def critic_loss_fn(
        self,
        batch,
        *,
        calql_alpha: float = 0.0,
        calql_n_actions: Optional[int] = None,
        calql_temperature: Optional[float] = None,
    ):
        batch_size = batch["rewards"].shape[0]
        action_mask = (
            batch.get("action_mask", None) if isinstance(batch, dict) else None
        )

        with torch.no_grad():
            otf_num_samples = max(1, int(self.config.get("otf_num_samples", 1)))
            if otf_num_samples == 1:
                (
                    next_actions,
                    next_actions_log_probs,
                ) = self._sample_policy_actions_for_critic(
                    batch["next_observations"],
                    num_samples=1,
                    train=True,
                    with_log_prob=True,
                )
                target_next_qs = self.forward_target_critic(
                    batch["next_observations"], next_actions
                )
                if target_next_qs.ndim == 1:
                    target_next_qs = target_next_qs.unsqueeze(0)

                if self.config["critic_subsample_size"] is not None:
                    num_q = target_next_qs.shape[0]
                    subsample = torch.randint(
                        low=0,
                        high=num_q,
                        size=(self.config["critic_subsample_size"],),
                        device=target_next_qs.device,
                    )
                    target_next_qs = target_next_qs[subsample]

                target_next_min_q = target_next_qs.min(dim=0).values
            else:
                # OTF: 对 next action 多次采样，使用最优 bootstrap Q 提升前期样本效率。
                (
                    next_actions_k,
                    next_actions_log_probs_k,
                ) = self._sample_policy_actions_for_critic(
                    batch["next_observations"],
                    num_samples=otf_num_samples,
                    train=True,
                    with_log_prob=True,
                )  # (B,K,A), (B,K)
                target_next_qs = self.forward_target_critic(
                    batch["next_observations"],
                    next_actions_k,
                )  # (Q,B,K)
                if target_next_qs.ndim == 2:
                    target_next_qs = target_next_qs.unsqueeze(0)

                if self.config["critic_subsample_size"] is not None:
                    num_q = target_next_qs.shape[0]
                    subsample = torch.randint(
                        low=0,
                        high=num_q,
                        size=(self.config["critic_subsample_size"],),
                        device=target_next_qs.device,
                    )
                    target_next_qs = target_next_qs[subsample]

                q_min_bk = target_next_qs.min(dim=0).values  # (B,K)
                best_idx = torch.argmax(q_min_bk, dim=-1)  # (B,)
                target_next_min_q = q_min_bk.gather(-1, best_idx.unsqueeze(-1)).squeeze(
                    -1
                )
                next_actions_log_probs = next_actions_log_probs_k.gather(
                    -1,
                    best_idx.unsqueeze(-1),
                ).squeeze(-1)

            if self.config["backup_entropy"]:
                temperature = self.forward_temperature().detach()
                target_next_min_q = (
                    target_next_min_q - temperature * next_actions_log_probs
                )
            target_q = (
                batch["rewards"]
                + self.config["discount"] * batch["masks"] * target_next_min_q
            )

        critic_actions = self._apply_action_mask(batch["actions"], action_mask)
        predicted_qs = self.forward_critic(
            batch["observations"], critic_actions, train=True
        )
        if predicted_qs.ndim == 1:
            predicted_qs = predicted_qs.unsqueeze(0)

        target_qs = target_q.unsqueeze(0).expand_as(predicted_qs)
        td_critic_loss = torch.mean((predicted_qs - target_qs) ** 2)
        critic_loss = td_critic_loss

        predicted_q_min = predicted_qs.min(dim=0).values
        predicted_q_max = predicted_qs.max(dim=0).values
        if predicted_qs.shape[0] > 1:
            predicted_q_std = predicted_qs.std(dim=0, unbiased=False)
        else:
            predicted_q_std = torch.zeros_like(predicted_q_min)
        predicted_q_gap = predicted_q_max - predicted_q_min

        calql_alpha = float(calql_alpha)
        cql_penalty = torch.as_tensor(0.0, device=predicted_qs.device)
        if calql_alpha > 0.0:
            # Cal-QL/CQL-style conservative regularization:
            # 约束 Q(s, a_data) 不应明显低于采样动作集合上的 log-sum-exp 估计。
            n_actions = (
                int(calql_n_actions)
                if calql_n_actions is not None
                else int(self.config.get("cql_n_actions", 10))
            )
            n_actions = max(1, n_actions)
            cql_temp = (
                float(calql_temperature)
                if calql_temperature is not None
                else float(self.config.get("cql_temperature", 1.0))
            )
            cql_temp = max(cql_temp, 1e-6)

            action_dim = int(batch["actions"].shape[-1])
            random_actions = torch.empty(
                (batch_size, n_actions, action_dim),
                device=predicted_qs.device,
                dtype=predicted_qs.dtype,
            ).uniform_(-1.0, 1.0)
            random_actions = self._apply_action_mask(random_actions, action_mask)

            policy_actions, _ = self._sample_policy_actions_for_critic(
                batch["observations"],
                num_samples=n_actions,
                train=True,
                with_log_prob=False,
            )
            policy_actions = self._apply_action_mask(policy_actions, action_mask)

            q_rand = self.forward_critic(
                batch["observations"], random_actions, train=True
            )  # (Q,B,N)
            q_pi = self.forward_critic(
                batch["observations"], policy_actions, train=True
            )  # (Q,B,N)
            q_cat = torch.cat([q_rand, q_pi], dim=-1)  # (Q,B,2N)

            lse_q = torch.logsumexp(q_cat / cql_temp, dim=-1) * cql_temp  # (Q,B)
            cql_penalty = torch.mean(lse_q - predicted_qs)
            critic_loss = td_critic_loss + calql_alpha * cql_penalty

        info = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_td_loss": float(td_critic_loss.detach().cpu()),
            "critic_cql_penalty": float(cql_penalty.detach().cpu()),
            "predicted_qs": float(predicted_qs.mean().detach().cpu()),
            "target_qs": float(target_qs.mean().detach().cpu()),
            "predicted_q_min": float(predicted_q_min.mean().detach().cpu()),
            "predicted_q_max": float(predicted_q_max.mean().detach().cpu()),
            "predicted_q_std": float(predicted_q_std.mean().detach().cpu()),
            "predicted_q_gap": float(predicted_q_gap.mean().detach().cpu()),
            "batch_size": int(batch_size),
            "otf_num_samples": int(self.config.get("otf_num_samples", 1)),
            "calql_alpha": float(calql_alpha),
        }

        return critic_loss, info

    def policy_loss_fn(self, batch):
        temperature = self.forward_temperature().detach()
        action_distributions = self.forward_policy(batch["observations"], train=True)
        (
            policy_actions,
            log_prob_per_dim,
        ) = action_distributions.sample_and_log_prob_per_dim()
        actions = self._transform_policy_actions_for_critic(
            batch["observations"],
            policy_actions,
        )
        action_mask = (
            batch.get("action_mask", None) if isinstance(batch, dict) else None
        )
        actions = self._apply_action_mask(actions, action_mask)
        policy_action_mask = self._project_critic_action_mask_to_policy_space(
            action_mask,
            policy_action_dim=int(policy_actions.shape[-1]),
        )
        log_probs = self._reduce_log_prob_with_mask(
            log_prob_per_dim, policy_action_mask
        )

        predicted_qs = self.forward_critic(batch["observations"], actions, train=True)
        if predicted_qs.ndim == 1:
            predicted_q = predicted_qs
            predicted_q_min = predicted_qs
            predicted_q_std = torch.zeros_like(predicted_qs)
        else:
            predicted_q = predicted_qs.mean(dim=0)
            predicted_q_min = predicted_qs.min(dim=0).values
            predicted_q_std = predicted_qs.std(dim=0, unbiased=False)

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -torch.mean(actor_objective)

        info = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "temperature": float(temperature.detach().cpu()),
            "entropy": float((-log_probs.mean()).detach().cpu()),
            "log_prob": float(log_probs.mean().detach().cpu()),
            "policy_active_dims": float(
                policy_action_mask.to(dtype=torch.float32)
                .sum(dim=-1)
                .mean()
                .detach()
                .cpu()
            )
            if policy_action_mask is not None
            else float(policy_actions.shape[-1]),
            "actor_predicted_q": float(predicted_q.mean().detach().cpu()),
            "actor_predicted_q_min": float(predicted_q_min.mean().detach().cpu()),
            "actor_predicted_q_std": float(predicted_q_std.mean().detach().cpu()),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch):
        with torch.no_grad():
            action_distributions = self.forward_policy(
                batch["observations"], train=True
            )
            (
                policy_actions,
                log_prob_per_dim,
            ) = action_distributions.sample_and_log_prob_per_dim()
            action_mask = (
                batch.get("action_mask", None) if isinstance(batch, dict) else None
            )
            policy_action_mask = self._project_critic_action_mask_to_policy_space(
                action_mask,
                policy_action_dim=int(policy_actions.shape[-1]),
            )
            actions_log_probs = self._reduce_log_prob_with_mask(
                log_prob_per_dim,
                policy_action_mask,
            )
        entropy = -actions_log_probs.mean()
        target_entropy = torch.as_tensor(
            self.config["target_entropy"],
            device=self.device,
            dtype=torch.float32,
        )
        target_entropy_abs = torch.abs(target_entropy)
        penalty = self.temperature_lagrange_penalty(entropy)
        temperature_loss = penalty.mean() if penalty.ndim > 0 else penalty
        return temperature_loss, {
            "temperature_loss": float(temperature_loss.detach().cpu()),
            "temperature_entropy": float(entropy.detach().cpu()),
            "target_entropy": float(target_entropy.detach().cpu()),
            "target_entropy_abs": float(target_entropy_abs.detach().cpu()),
            "target_entropy_gap": float((entropy - target_entropy_abs).detach().cpu()),
            "temperature_constraint_gap": float(
                (entropy - target_entropy).detach().cpu()
            ),
            "temperature_policy_active_dims": float(
                policy_action_mask.to(dtype=torch.float32)
                .sum(dim=-1)
                .mean()
                .detach()
                .cpu()
            )
            if policy_action_mask is not None
            else float(policy_actions.shape[-1]),
        }

    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
    ) -> Tuple["SACAgent", dict]:
        del pmap_axis
        batch = _to_torch(batch, self.device)
        info = {}

        if "critic" in networks_to_update:
            self.state.zero_grad(["critic"])
            critic_loss, critic_info = self.critic_loss_fn(batch)
            critic_loss.backward()
            self.state.optimizer_step("critic")
            self.state.target_update(self.config["soft_target_update_rate"])
            info.update(critic_info)

        if "actor" in networks_to_update:
            self.state.zero_grad(["actor"])
            actor_loss, actor_info = self.policy_loss_fn(batch)
            actor_loss.backward()
            self.state.optimizer_step("actor")
            info.update(actor_info)

        if "temperature" in networks_to_update:
            self.state.zero_grad(["temperature"])
            temperature_loss, temperature_info = self.temperature_loss_fn(batch)
            temperature_loss.backward()
            self.state.optimizer_step("temperature")
            info.update(temperature_info)

        self.state.step += 1
        info.update(self.state.lr_info())

        return self, info

    def update_critics_calql(
        self,
        batch: Batch,
        *,
        calql_alpha: float,
        calql_n_actions: int,
        calql_temperature: float,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACAgent", dict]:
        """仅更新 critic，并启用 Cal-QL/CQL-style 保守项。"""
        del pmap_axis
        batch = _to_torch(batch, self.device)
        self.state.zero_grad(["critic"])
        critic_loss, critic_info = self.critic_loss_fn(
            batch,
            calql_alpha=float(calql_alpha),
            calql_n_actions=int(calql_n_actions),
            calql_temperature=float(calql_temperature),
        )
        critic_loss.backward()
        self.state.optimizer_step("critic")
        self.state.target_update(self.config["soft_target_update_rate"])
        self.state.step += 1
        critic_info.update(self.state.lr_info())
        return self, critic_info

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[int] = None,
        argmax: bool = False,
        deterministic: Optional[bool] = None,
        **kwargs,
    ):
        del kwargs, seed
        if deterministic is not None:
            argmax = deterministic

        obs_t = _to_torch(observations, self.device)
        dist = self.forward_policy(obs_t, train=False)
        actions = dist.mode() if argmax else dist.sample()
        return actions.detach().cpu().numpy()

    @classmethod
    def create(
        cls,
        rng: Optional[int],
        observations: Data,
        actions,
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        critic_actions=None,
        actor_optimizer_kwargs={"learning_rate": 3e-4, "warmup_steps": 2000},
        critic_optimizer_kwargs={"learning_rate": 3e-4, "warmup_steps": 2000},
        temperature_optimizer_kwargs={"learning_rate": 3e-4},
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        otf_num_samples: int = 1,
        cql_n_actions: int = 10,
        cql_temperature: float = 1.0,
        action_transform: Optional[dict] = None,
    ):
        del rng
        if entropy_per_dim:
            raise NotImplementedError(
                "entropy_per_dim is not supported in torch migration"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        actor_def = actor_def.to(device)
        critic_def = critic_def.to(device)
        temperature_def = temperature_def.to(device)

        obs_t = _to_torch(observations, device)
        act_t = _to_torch(actions, device)
        critic_act_t = _to_torch(
            critic_actions if critic_actions is not None else actions, device
        )

        with torch.no_grad():
            actor_def(obs_t, train=False)
            critic_def(obs_t, critic_act_t, train=False)
            temperature_def()

        critic_param_ids = {id(p) for p in critic_def.parameters()}
        actor_only_params = [
            p for p in actor_def.parameters() if id(p) not in critic_param_ids
        ]
        actor_bundle = make_optimizer(actor_only_params, **actor_optimizer_kwargs)
        critic_bundle = make_optimizer(
            critic_def.parameters(), **critic_optimizer_kwargs
        )
        temp_bundle = make_optimizer(
            temperature_def.parameters(), **temperature_optimizer_kwargs
        )

        state = TorchRLTrainState(
            modules={
                "actor": actor_def,
                "critic": critic_def,
                "temperature": temperature_def,
            },
            target_modules={
                "critic": copy.deepcopy(critic_def).to(device),
            },
            optimizers={
                "actor": actor_bundle.optimizer,
                "critic": critic_bundle.optimizer,
                "temperature": temp_bundle.optimizer,
            },
            schedulers={
                "actor": actor_bundle.scheduler,
                "critic": critic_bundle.scheduler,
                "temperature": temp_bundle.scheduler,
            },
            grad_clip_norms={
                "actor": actor_bundle.clip_grad_norm,
                "critic": critic_bundle.clip_grad_norm,
                "temperature": temp_bundle.clip_grad_norm,
            },
            device=device,
        )

        if target_entropy is None:
            target_entropy = -float(act_t.shape[-1]) / 2.0

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                otf_num_samples=int(max(1, otf_num_samples)),
                cql_n_actions=int(max(1, cql_n_actions)),
                cql_temperature=float(max(1e-6, cql_temperature)),
                action_transform=copy.deepcopy(action_transform),
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng,
        observations,
        actions,
        encoder_def: nn.Module,
        critic_actions=None,
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        **kwargs,
    ):
        policy_network_kwargs = dict(policy_network_kwargs)
        critic_network_kwargs = dict(critic_network_kwargs)
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        encoder_wrapped = EncodingWrapper(
            encoder=encoder_def,
            use_proprio=use_proprio,
            enable_stacking=True,
        )

        if shared_encoder:
            actor_encoder = encoder_wrapped
            critic_encoder = encoder_wrapped
        else:
            actor_encoder = encoder_wrapped
            critic_encoder = copy.deepcopy(encoder_wrapped)

        policy_def = Policy(
            encoder=actor_encoder,
            network=MLP(**policy_network_kwargs),
            action_dim=np.asarray(actions).shape[-1],
            **policy_kwargs,
        )

        critic_ctor = lambda: Critic(
            encoder=critic_encoder, network=MLP(**critic_network_kwargs)
        )
        critic_def = CriticEnsemble(
            critic_ctor=critic_ctor, num_qs=critic_ensemble_size
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )

        return cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_actions=critic_actions,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            **kwargs,
        )

    @classmethod
    def create_states(
        cls,
        rng,
        observations,
        actions,
        critic_actions=None,
        critic_network_kwargs: dict = {"hidden_dims": [256, 256]},
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        policy_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        temperature_init: float = 1.0,
        **kwargs,
    ):
        policy_network_kwargs = dict(policy_network_kwargs)
        critic_network_kwargs = dict(critic_network_kwargs)
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        action_dim = np.asarray(actions).shape[-1]

        policy_def = Policy(
            encoder=None,
            network=MLP(**policy_network_kwargs),
            action_dim=action_dim,
            **policy_kwargs,
        )

        critic_ctor = lambda: Critic(encoder=None, network=MLP(**critic_network_kwargs))
        critic_def = CriticEnsemble(
            critic_ctor=critic_ctor, num_qs=critic_ensemble_size
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
        )

        return cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_actions=critic_actions,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            **kwargs,
        )

    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACAgent", dict]:
        del pmap_axis
        batch_t = _to_torch(batch, self.device)
        minibatches = _split_batch(batch_t, utd_ratio)

        critic_infos = []
        for i in range(utd_ratio):
            minibatch = _index_batch(minibatches, i)
            self, info = self.update(
                minibatch, networks_to_update=frozenset({"critic"})
            )
            critic_infos.append(info)

        _, actor_temp_info = self.update(
            batch_t,
            networks_to_update=frozenset({"actor", "temperature"}),
        )

        info = _tree_mean(critic_infos) if critic_infos else {}
        info.update(actor_temp_info)
        return self, info
