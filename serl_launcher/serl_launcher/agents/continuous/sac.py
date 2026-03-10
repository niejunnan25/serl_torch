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
            raise ValueError(f"Batch size {b} must be divisible by utd_ratio {utd_ratio}")
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

    def forward_critic(self, observations: Data, actions: torch.Tensor, train: bool = True):
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

    def temperature_lagrange_penalty(self, entropy: torch.Tensor):
        return self.state.modules["temperature"](
            lhs=entropy,
            rhs=torch.as_tensor(self.config["target_entropy"], device=self.device, dtype=torch.float32),
        )

    def _compute_next_actions(self, batch, *, num_samples: int = 1):
        """
        采样 next residual actions 以构建 TD target。

        - num_samples=1: 返回形状 (B, A), (B,)
        - num_samples>1: 返回形状 (K, B, A), (K, B)
        """
        next_action_distributions = self.forward_policy(batch["next_observations"], train=True)
        k = max(1, int(num_samples))
        if k == 1:
            next_actions, next_actions_log_probs = next_action_distributions.sample_and_log_prob()
            return next_actions, next_actions_log_probs

        next_actions_list = []
        next_log_probs_list = []
        for _ in range(k):
            sampled_actions, sampled_log_probs = next_action_distributions.sample_and_log_prob()
            next_actions_list.append(sampled_actions)
            next_log_probs_list.append(sampled_log_probs)
        return torch.stack(next_actions_list, dim=0), torch.stack(next_log_probs_list, dim=0)

    def critic_loss_fn(
        self,
        batch,
        *,
        calql_alpha: float = 0.0,
        calql_n_actions: Optional[int] = None,
        calql_temperature: Optional[float] = None,
    ):
        batch_size = batch["rewards"].shape[0]

        with torch.no_grad():
            otf_num_samples = max(1, int(self.config.get("otf_num_samples", 1)))
            if otf_num_samples == 1:
                next_actions, next_actions_log_probs = self._compute_next_actions(batch, num_samples=1)
                target_next_qs = self.forward_target_critic(batch["next_observations"], next_actions)
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
                next_actions_k, next_actions_log_probs_k = self._compute_next_actions(
                    batch,
                    num_samples=otf_num_samples,
                )  # (K,B,A), (K,B)
                target_next_qs = self.forward_target_critic(
                    batch["next_observations"],
                    next_actions_k.permute(1, 0, 2),
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
                target_next_min_q = q_min_bk.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
                next_actions_log_probs = next_actions_log_probs_k.permute(1, 0).gather(
                    -1,
                    best_idx.unsqueeze(-1),
                ).squeeze(-1)

            target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * target_next_min_q

            if self.config["backup_entropy"]:
                temperature = self.forward_temperature().detach()
                target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(batch["observations"], batch["actions"], train=True)
        if predicted_qs.ndim == 1:
            predicted_qs = predicted_qs.unsqueeze(0)

        target_qs = target_q.unsqueeze(0).expand_as(predicted_qs)
        td_critic_loss = torch.mean((predicted_qs - target_qs) ** 2)
        critic_loss = td_critic_loss

        calql_alpha = float(calql_alpha)
        cql_penalty = torch.as_tensor(0.0, device=predicted_qs.device)
        if calql_alpha > 0.0:
            # Cal-QL/CQL-style conservative regularization:
            # 约束 Q(s, a_data) 不应明显低于采样动作集合上的 log-sum-exp 估计。
            n_actions = int(calql_n_actions) if calql_n_actions is not None else int(self.config.get("cql_n_actions", 10))
            n_actions = max(1, n_actions)
            cql_temp = float(calql_temperature) if calql_temperature is not None else float(
                self.config.get("cql_temperature", 1.0)
            )
            cql_temp = max(cql_temp, 1e-6)

            action_dim = int(batch["actions"].shape[-1])
            random_actions = torch.empty(
                (batch_size, n_actions, action_dim),
                device=predicted_qs.device,
                dtype=predicted_qs.dtype,
            ).uniform_(-1.0, 1.0)

            action_distribution = self.forward_policy(batch["observations"], train=True)
            policy_actions = torch.stack([action_distribution.sample() for _ in range(n_actions)], dim=1)

            q_rand = self.forward_critic(batch["observations"], random_actions, train=True)  # (Q,B,N)
            q_pi = self.forward_critic(batch["observations"], policy_actions, train=True)  # (Q,B,N)
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
            "batch_size": int(batch_size),
            "otf_num_samples": int(self.config.get("otf_num_samples", 1)),
            "calql_alpha": float(calql_alpha),
        }

        return critic_loss, info

    def policy_loss_fn(self, batch):
        temperature = self.forward_temperature().detach()
        action_distributions = self.forward_policy(batch["observations"], train=True)
        actions, log_probs = action_distributions.sample_and_log_prob()

        predicted_qs = self.forward_critic(batch["observations"], actions, train=True)
        if predicted_qs.ndim == 1:
            predicted_q = predicted_qs
        else:
            predicted_q = predicted_qs.mean(dim=0)

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -torch.mean(actor_objective)

        info = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "temperature": float(temperature.detach().cpu()),
            "entropy": float((-log_probs.mean()).detach().cpu()),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch):
        with torch.no_grad():
            _, next_actions_log_probs = self._compute_next_actions(batch, num_samples=1)
        entropy = -next_actions_log_probs.mean()
        penalty = self.temperature_lagrange_penalty(entropy)
        temperature_loss = penalty.mean() if penalty.ndim > 0 else penalty
        return temperature_loss, {"temperature_loss": float(temperature_loss.detach().cpu())}

    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: FrozenSet[str] = frozenset({"actor", "critic", "temperature"}),
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
    ):
        del rng
        if entropy_per_dim:
            raise NotImplementedError("entropy_per_dim is not supported in torch migration")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        actor_def = actor_def.to(device)
        critic_def = critic_def.to(device)
        temperature_def = temperature_def.to(device)

        obs_t = _to_torch(observations, device)
        act_t = _to_torch(actions, device)

        with torch.no_grad():
            actor_def(obs_t, train=False)
            critic_def(obs_t, act_t, train=False)
            temperature_def()

        critic_param_ids = {id(p) for p in critic_def.parameters()}
        actor_only_params = [p for p in actor_def.parameters() if id(p) not in critic_param_ids]
        actor_bundle = make_optimizer(actor_only_params, **actor_optimizer_kwargs)
        critic_bundle = make_optimizer(critic_def.parameters(), **critic_optimizer_kwargs)
        temp_bundle = make_optimizer(temperature_def.parameters(), **temperature_optimizer_kwargs)

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
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng,
        observations,
        actions,
        encoder_def: nn.Module,
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

        critic_ctor = lambda: Critic(encoder=critic_encoder, network=MLP(**critic_network_kwargs))
        critic_def = CriticEnsemble(critic_ctor=critic_ctor, num_qs=critic_ensemble_size)

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
        critic_def = CriticEnsemble(critic_ctor=critic_ctor, num_qs=critic_ensemble_size)

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
            self, info = self.update(minibatch, networks_to_update=frozenset({"critic"}))
            critic_infos.append(info)

        _, actor_temp_info = self.update(
            batch_t,
            networks_to_update=frozenset({"actor", "temperature"}),
        )

        info = _tree_mean(critic_infos) if critic_infos else {}
        info.update(actor_temp_info)
        return self, info
