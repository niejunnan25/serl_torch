"""Hydra 配置解析与 agent 构建的共享工具函数（训练/评估通用）。"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from policy.action import resolve_control_indices


def set_global_seeds(seed: int) -> None:
    """统一设置 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_hidden_dims(values) -> list[int]:
    """将配置中的 hidden_dims 元素统一转成 int。"""
    return [int(v) for v in values]


def build_optimizer_kwargs(cfg: DictConfig, *, for_temperature: bool = False) -> Dict[str, Any]:
    """从配置构建优化器参数（adam / adamw + warmup / cosine / grad_clip）。"""
    opt_cfg = cfg.sac.get("optimizer", None)
    kwargs: Dict[str, Any] = {"learning_rate": float(cfg.sac.learning_rate)}
    if opt_cfg is None:
        return kwargs

    opt_type = str(opt_cfg.get("type", "adam")).lower()
    if opt_type not in {"adam", "adamw"}:
        raise ValueError(f"Unsupported sac.optimizer.type: {opt_type}")
    if opt_type == "adamw":
        kwargs["weight_decay"] = float(opt_cfg.get("weight_decay", 0.0))

    warmup_steps = opt_cfg.get("warmup_steps", None)
    if warmup_steps is not None:
        kwargs["warmup_steps"] = int(warmup_steps)
    cosine_decay_steps = opt_cfg.get("cosine_decay_steps", None)
    if cosine_decay_steps is not None:
        kwargs["cosine_decay_steps"] = int(cosine_decay_steps)

    grad_clip = opt_cfg.get("grad_clip_norm", None)
    if grad_clip is not None:
        kwargs["clip_grad_norm"] = float(grad_clip)

    if for_temperature and ("weight_decay" in kwargs):
        temp_wd = opt_cfg.get("temperature_weight_decay", None)
        if temp_wd is None:
            kwargs.pop("weight_decay", None)
        else:
            kwargs["weight_decay"] = float(temp_wd)
    return kwargs


def resolve_image_keys(cfg: DictConfig) -> Tuple[str, ...]:
    """解析残差观测图像键：residual.image_keys 优先，否则回退到 sac.image_keys。"""
    image_keys_cfg = cfg.residual.get("image_keys", None)
    source = image_keys_cfg if image_keys_cfg is not None else cfg.sac.image_keys
    image_keys = tuple(str(k) for k in source)
    if len(image_keys) == 0:
        raise ValueError("At least one residual image key is required")
    return image_keys


def resolve_control_indices_from_cfg(cfg: DictConfig) -> np.ndarray:
    """解析残差动作维度/下标配置。"""
    action_dim_cfg = cfg.residual.get("action_dim", None)
    action_dim = int(action_dim_cfg) if action_dim_cfg is not None else None

    action_indices_cfg = cfg.residual.get("action_indices", None)
    action_indices = (
        [int(v) for v in action_indices_cfg] if action_indices_cfg is not None else None
    )

    control_gripper_cfg = cfg.residual.get("control_gripper", None)
    control_gripper = bool(control_gripper_cfg) if control_gripper_cfg is not None else None

    return resolve_control_indices(
        action_dim=action_dim,
        action_indices=action_indices,
        control_gripper=control_gripper,
    )


def build_drq_agent(
    cfg: DictConfig,
    sample_obs: Dict[str, np.ndarray],
    action_dim: int,
    image_keys: Tuple[str, ...],
):
    """根据 Hydra 配置构建 DrQAgent（视觉编码 + SAC 主体）。"""
    from serl_launcher.agents.continuous.drq import DrQAgent

    sample_action = np.zeros((action_dim,), dtype=np.float32)
    actor_optim_kwargs = build_optimizer_kwargs(cfg, for_temperature=False)
    critic_optim_kwargs = build_optimizer_kwargs(cfg, for_temperature=False)
    temp_optim_kwargs = build_optimizer_kwargs(cfg, for_temperature=True)

    resnet_kwargs = None
    resnet_cfg = cfg.sac.get("resnet", None)
    if resnet_cfg is not None:
        model_name = str(resnet_cfg.get("model_name", "microsoft/resnet-18"))
        if not os.path.isabs(model_name) and not model_name.startswith(("http://", "https://")):
            candidate = os.path.join(get_original_cwd(), model_name)
            if os.path.isdir(candidate):
                model_name = candidate
        resnet_kwargs = {
            "model_name": model_name,
            "pretrained": bool(resnet_cfg.get("pretrained", True)),
            "freeze_backbone": bool(resnet_cfg.get("freeze_backbone", False)),
            "pooling_method": str(resnet_cfg.get("pooling_method", "spatial_learned_embeddings")),
            "num_spatial_blocks": int(resnet_cfg.get("num_spatial_blocks", 8)),
            "bottleneck_dim": int(resnet_cfg.get("bottleneck_dim", 256)),
        }

    return DrQAgent.create_drq(
        int(cfg.seed),
        sample_obs,
        sample_action,
        encoder_type=str(cfg.sac.encoder_type),
        shared_encoder=bool(cfg.sac.shared_encoder),
        use_proprio=bool(cfg.sac.use_proprio),
        image_keys=image_keys,
        resnet_kwargs=resnet_kwargs,
        critic_network_kwargs={
            "activations": str(cfg.sac.critic_activation),
            "use_layer_norm": bool(cfg.sac.critic_layer_norm),
            "hidden_dims": to_hidden_dims(cfg.sac.critic_hidden_dims),
        },
        policy_network_kwargs={
            "activations": str(cfg.sac.policy_activation),
            "use_layer_norm": bool(cfg.sac.policy_layer_norm),
            "hidden_dims": to_hidden_dims(cfg.sac.policy_hidden_dims),
        },
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": float(cfg.sac.std_min),
            "std_max": float(cfg.sac.std_max),
        },
        actor_optimizer_kwargs=actor_optim_kwargs,
        critic_optimizer_kwargs=critic_optim_kwargs,
        temperature_optimizer_kwargs=temp_optim_kwargs,
        discount=float(cfg.sac.discount),
        soft_target_update_rate=float(cfg.sac.soft_target_update_rate),
        backup_entropy=bool(cfg.sac.backup_entropy),
        otf_num_samples=int(cfg.sac.get("otf_num_samples", 1)),
        cql_n_actions=int(cfg.sac.get("cql_n_actions", 10)),
        cql_temperature=float(cfg.sac.get("cql_temperature", 1.0)),
        critic_ensemble_size=int(cfg.sac.critic_ensemble_size),
        critic_subsample_size=(
            int(cfg.sac.critic_subsample_size)
            if cfg.sac.critic_subsample_size is not None
            else None
        ),
        temperature_init=float(cfg.sac.temperature_init),
    )


def sample_probing_steps(probing_cfg: DictConfig, *, episode_horizon: int) -> int:
    """
    按配置采样每个 episode 的 base probing 步数。

    probing_cfg 可以是 cfg.training 或 cfg.eval（字段名相同）。
    优先按 U(0, alpha * T) 采样；否则回退到 min_steps / max_steps 区间。
    """
    if not bool(probing_cfg.get("enable_base_probing", False)):
        return 0

    alpha_cfg = probing_cfg.get("probing_alpha", None)
    if alpha_cfg is not None:
        alpha = float(np.clip(float(alpha_cfg), 0.0, 1.0))
        max_steps = int(max(0, round(alpha * float(max(0, int(episode_horizon))))))
        if max_steps <= 0:
            return 0
        return int(random.randint(0, max_steps))

    min_steps = int(probing_cfg.get("probing_min_steps", 0))
    max_steps = int(probing_cfg.get("probing_max_steps", 0))
    min_steps = max(0, min_steps)
    max_steps = max(0, max_steps)
    if max_steps < min_steps:
        min_steps, max_steps = max_steps, min_steps
    if max_steps == 0:
        return 0
    return int(random.randint(min_steps, max_steps))
