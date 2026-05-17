from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


class DiagGaussianDistribution:
    def __init__(
        self,
        loc: torch.Tensor,
        scale_diag: torch.Tensor,
        tanh_squash: bool = False,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        self.loc = loc
        self.scale_diag = scale_diag
        self.tanh_squash = tanh_squash
        self.low = low
        self.high = high
        self.eps = eps
        self._normal_dist = Normal(loc, scale_diag)
        self._base_dist = Independent(self._normal_dist, 1)

    def _squash(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(x)
        if self.low is not None and self.high is not None:
            y = (y + 1.0) * 0.5
            y = y * (self.high - self.low) + self.low
        return y

    def _unsquash(self, y: torch.Tensor) -> torch.Tensor:
        if self.low is not None and self.high is not None:
            y = (y - self.low) / (self.high - self.low + self.eps)
            y = 2.0 * y - 1.0
        y = torch.clamp(y, -1.0 + self.eps, 1.0 - self.eps)
        return 0.5 * torch.log((1.0 + y) / (1.0 - y))

    def sample(self, seed=None) -> torch.Tensor:
        z = self._base_dist.rsample()
        if self.tanh_squash:
            return self._squash(z)
        return z

    def _log_prob_per_dim_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        log_prob_per_dim = self._normal_dist.log_prob(z)
        if not self.tanh_squash:
            return log_prob_per_dim

        tanh_z = torch.tanh(z)
        log_prob_per_dim = log_prob_per_dim - torch.log(1.0 - tanh_z.pow(2) + self.eps)
        if self.low is not None and self.high is not None:
            scale = torch.clamp((self.high - self.low) * 0.5, min=self.eps)
            log_prob_per_dim = log_prob_per_dim - torch.log(scale)
        return log_prob_per_dim

    def sample_and_log_prob_per_dim(self, seed=None):
        z = self._base_dist.rsample()
        action = self._squash(z) if self.tanh_squash else z
        log_prob_per_dim = self._log_prob_per_dim_from_latent(z)
        return action, log_prob_per_dim

    def sample_and_log_prob(self, seed=None):
        action, log_prob_per_dim = self.sample_and_log_prob_per_dim(seed=seed)
        log_prob = log_prob_per_dim.sum(dim=-1)
        return action, log_prob

    def log_prob_per_dim(self, action: torch.Tensor) -> torch.Tensor:
        if not self.tanh_squash:
            return self._normal_dist.log_prob(action)

        z = self._unsquash(action)
        return self._log_prob_per_dim_from_latent(z)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.log_prob_per_dim(action).sum(dim=-1)

    def mode(self) -> torch.Tensor:
        if self.tanh_squash:
            return self._squash(self.loc)
        return self.loc

    def stddev(self) -> torch.Tensor:
        return self.scale_diag


class ValueCritic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.value_head = nn.LazyLinear(1)
        self.init_final = init_final
        self._head_initialized = False

    def forward(self, observations, train: bool = False):
        obs = (
            observations
            if self.encoder is None
            else self.encoder(observations, train=train)
        )
        outputs = self.network(obs, train=train)
        value = self.value_head(outputs)
        if self.init_final is not None and not self._head_initialized:
            nn.init.uniform_(self.value_head.weight, -self.init_final, self.init_final)
            nn.init.uniform_(self.value_head.bias, -self.init_final, self.init_final)
            self._head_initialized = True
        return value.squeeze(-1)


class Critic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.q_head = nn.LazyLinear(1)
        self.init_final = init_final
        self._head_initialized = False

    def _encode_obs(self, observations, train: bool = False):
        if self.encoder is None:
            return observations
        return self.encoder(observations, train=train)

    def _forward_single(
        self, obs_enc: torch.Tensor, actions: torch.Tensor, train: bool = False
    ):
        inputs = torch.cat([obs_enc, actions], dim=-1)
        outputs = self.network(inputs, train=train)
        q = self.q_head(outputs)
        if self.init_final is not None and not self._head_initialized:
            nn.init.uniform_(self.q_head.weight, -self.init_final, self.init_final)
            nn.init.uniform_(self.q_head.bias, -self.init_final, self.init_final)
            self._head_initialized = True
        return q.squeeze(-1)

    def forward_encoded(
        self, obs_enc: torch.Tensor, actions: torch.Tensor, train: bool = False
    ):
        if actions.ndim == 3:
            bsz, num_actions, act_dim = actions.shape
            flat_actions = actions.reshape(bsz * num_actions, act_dim)
            flat_obs = (
                obs_enc.unsqueeze(1)
                .expand(-1, num_actions, -1)
                .reshape(bsz * num_actions, -1)
            )
            q_values = self._forward_single(flat_obs, flat_actions, train=train)
            return q_values.reshape(bsz, num_actions)

        return self._forward_single(obs_enc, actions, train=train)

    def forward(self, observations, actions: torch.Tensor, train: bool = False):
        obs_enc = self._encode_obs(observations, train=train)
        return self.forward_encoded(obs_enc, actions, train=train)


class DistributionalCritic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        q_low: float,
        q_high: float,
        num_atoms: int = 51,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.q_low = q_low
        self.q_high = q_high
        self.num_atoms = num_atoms
        self.logit_head = nn.LazyLinear(num_atoms)
        self.init_final = init_final
        self._head_initialized = False

    def forward(self, observations, actions: torch.Tensor, train: bool = False):
        obs_enc = (
            observations
            if self.encoder is None
            else self.encoder(observations, train=train)
        )
        x = torch.cat([obs_enc, actions], dim=-1)
        logits = self.logit_head(self.network(x, train=train))
        if self.init_final is not None and not self._head_initialized:
            nn.init.uniform_(self.logit_head.weight, -self.init_final, self.init_final)
            nn.init.uniform_(self.logit_head.bias, -self.init_final, self.init_final)
            self._head_initialized = True
        atoms = torch.linspace(
            self.q_low,
            self.q_high,
            self.num_atoms,
            device=logits.device,
            dtype=logits.dtype,
        ).expand_as(logits)
        return logits, atoms


class ContrastiveCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        sa_net: nn.Module,
        g_net: nn.Module,
        repr_dim: int = 16,
        twin_q: bool = True,
        sa_net2: Optional[nn.Module] = None,
        g_net2: Optional[nn.Module] = None,
        init_final: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.sa_net = sa_net
        self.g_net = g_net
        self.repr_dim = repr_dim
        self.twin_q = twin_q
        self.sa_net2 = sa_net2
        self.g_net2 = g_net2
        self.head_sa = nn.LazyLinear(repr_dim)
        self.head_g = nn.LazyLinear(repr_dim)
        self.head_sa2 = nn.LazyLinear(repr_dim) if twin_q else None
        self.head_g2 = nn.LazyLinear(repr_dim) if twin_q else None
        self.init_final = init_final

    def forward(self, observations, actions: torch.Tensor, train: bool = False):
        obs_goal_encoding = self.encoder(observations)
        encoding_dim = obs_goal_encoding.shape[-1] // 2
        obs_encoding, goal_encoding = (
            obs_goal_encoding[..., :encoding_dim],
            obs_goal_encoding[..., encoding_dim:],
        )

        sa_inputs = torch.cat([obs_encoding, actions], dim=-1)
        sa_repr = self.head_sa(self.sa_net(sa_inputs, train=train))
        g_repr = self.head_g(self.g_net(goal_encoding, train=train))
        outer = torch.einsum("ik,jk->ij", sa_repr, g_repr)

        if self.twin_q and self.sa_net2 is not None and self.g_net2 is not None:
            sa_repr2 = self.head_sa2(self.sa_net2(sa_inputs, train=train))
            g_repr2 = self.head_g2(self.g_net2(goal_encoding, train=train))
            outer2 = torch.einsum("ik,jk->ij", sa_repr2, g_repr2)
            outer = torch.stack([outer, outer2], dim=-1)

        return outer


class CriticEnsemble(nn.Module):
    def __init__(self, critic_ctor, num_qs: int):
        super().__init__()
        self.models = nn.ModuleList([critic_ctor() for _ in range(num_qs)])
        self._can_share_encoder_forward = self._detect_shared_encoder()

    def _detect_shared_encoder(self) -> bool:
        if len(self.models) <= 1:
            return False
        if not all(hasattr(critic, "forward_encoded") for critic in self.models):
            return False
        first_encoder = getattr(self.models[0], "encoder", None)
        if first_encoder is None:
            return False
        return all(
            getattr(critic, "encoder", None) is first_encoder
            for critic in self.models[1:]
        )

    def forward(self, observations, actions: torch.Tensor, train: bool = False):
        if self._can_share_encoder_forward:
            first_critic = self.models[0]
            obs_enc = first_critic._encode_obs(observations, train=train)
            qs = [
                critic.forward_encoded(obs_enc, actions, train=train)
                for critic in self.models
            ]
        else:
            qs = [critic(observations, actions, train=train) for critic in self.models]
        return torch.stack(qs, dim=0)


class Policy(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        action_dim: int,
        init_final: Optional[float] = None,
        std_parameterization: str = "exp",
        std_min: Optional[float] = 1e-5,
        std_max: Optional[float] = 10.0,
        tanh_squash_distribution: bool = False,
        fixed_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.init_final = init_final
        self.std_parameterization = std_parameterization
        self.std_min = std_min
        self.std_max = std_max
        self.tanh_squash_distribution = tanh_squash_distribution
        self.fixed_std = fixed_std

        self.mean_head = nn.LazyLinear(action_dim)
        if fixed_std is None and std_parameterization != "uniform":
            self.std_head = nn.LazyLinear(action_dim)
        else:
            self.std_head = None

        if std_parameterization == "uniform":
            self.log_stds = nn.Parameter(torch.zeros(action_dim))
        else:
            self.log_stds = None

    def _encode(self, observations, train: bool):
        if self.encoder is None:
            return observations

        try:
            return self.encoder(observations, train=train, stop_gradient=True)
        except TypeError:
            return self.encoder(observations, train=train)

    def forward(self, observations, temperature: float = 1.0, train: bool = False):
        # 1) 先对原始观测做编码（像素任务里会走视觉编码器；状态任务则可直接透传）
        obs_enc = self._encode(observations, train=train)
        # 2) 编码后的特征进入策略 MLP 主干
        outputs = self.network(obs_enc, train=train)

        # 3) 预测动作高斯分布的均值 μ
        means = self.mean_head(outputs)

        # 4) 预测动作高斯分布的标准差 σ
        #    支持三种参数化：
        #    - exp:      σ = exp(raw)
        #    - softplus: σ = softplus(raw)
        #    - uniform:  使用全局可学习 log_stds（与观测无关）
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                stds = torch.exp(self.std_head(outputs))
            elif self.std_parameterization == "softplus":
                stds = F.softplus(self.std_head(outputs))
            elif self.std_parameterization == "uniform":
                stds = torch.exp(self.log_stds).expand_as(means)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            if self.std_parameterization != "fixed":
                raise ValueError("fixed_std requires std_parameterization='fixed'")
            stds = torch.as_tensor(
                self.fixed_std, device=means.device, dtype=means.dtype
            )
            if stds.ndim == 1:
                stds = stds.expand_as(means)

        # 5) 对标准差做上下界裁剪，避免过小导致数值不稳定或过大导致动作噪声过强
        if self.std_min is not None or self.std_max is not None:
            stds = torch.clamp(stds, min=self.std_min, max=self.std_max)

        # 6) 温度缩放：temperature 越大，策略采样噪声越大（探索更强）
        stds = stds * (temperature**0.5)

        # 7) 返回对角高斯分布对象；后续由上层选择 sample()/mode()
        #    tanh_squash=True 时会将动作压到 (-1, 1) 区间
        return DiagGaussianDistribution(
            loc=means,
            scale_diag=stds,
            tanh_squash=self.tanh_squash_distribution,
        )


class TanhMultivariateNormalDiag(DiagGaussianDistribution):
    def __init__(
        self,
        loc: torch.Tensor,
        scale_diag: torch.Tensor,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            loc=loc,
            scale_diag=scale_diag,
            tanh_squash=True,
            low=low,
            high=high,
        )


def ensemblize(cls_or_ctor, num_qs, out_axes=0):
    class _Ensemble(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            kwargs.pop("name", None)
            self.models = nn.ModuleList(
                [cls_or_ctor(*args, **kwargs) for _ in range(num_qs)]
            )
            self.out_axes = out_axes

        def forward(self, *args, **kwargs):
            outputs = [model(*args, **kwargs) for model in self.models]
            return torch.stack(outputs, dim=self.out_axes)

    return _Ensemble
