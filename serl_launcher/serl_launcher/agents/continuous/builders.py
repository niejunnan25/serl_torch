"""Builders for continuous-control agents."""
from __future__ import annotations

import os
from typing import Any
from typing import Dict

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def _to_hidden_dims(values) -> list[int]:
    return [int(v) for v in values]


def _build_optimizer_kwargs(
    cfg: DictConfig,
    *,
    for_temperature: bool = False,
) -> Dict[str, Any]:
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


def _build_mixed_precision_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    training_cfg = cfg.get("training", {})
    mixed_precision_cfg = training_cfg.get("mixed_precision", None)
    if mixed_precision_cfg is None:
        return {
            "enabled": False,
            "dtype": "bfloat16",
        }
    return {
        "enabled": bool(mixed_precision_cfg.get("enabled", False)),
        "dtype": str(mixed_precision_cfg.get("dtype", "bfloat16")),
    }


def build_drq_agent(
    cfg: DictConfig,
    sample_obs: Dict[str, np.ndarray],
    action_dim: int,
    image_keys: tuple[str, ...],
    *,
    critic_action_dim: int | None = None,
    action_transform: Dict[str, Any] | None = None,
    device: str | torch.device | None = None,
):
    from serl_launcher.agents.continuous.drq import DrQAgent

    sample_action = np.zeros((action_dim,), dtype=np.float32)
    sample_critic_action = np.zeros(
        (int(critic_action_dim) if critic_action_dim is not None else int(action_dim),),
        dtype=np.float32,
    )
    actor_optim_kwargs = _build_optimizer_kwargs(cfg, for_temperature=False)
    critic_optim_kwargs = _build_optimizer_kwargs(cfg, for_temperature=False)
    temp_optim_kwargs = _build_optimizer_kwargs(cfg, for_temperature=True)

    resnet_kwargs = None
    resnet_cfg = cfg.sac.get("resnet", None)
    if resnet_cfg is not None:
        model_name = str(resnet_cfg.get("model_name", "microsoft/resnet-18"))
        if not os.path.isabs(model_name) and not model_name.startswith(
            ("http://", "https://")
        ):
            candidate = os.path.join(get_original_cwd(), model_name)
            if os.path.isdir(candidate):
                model_name = candidate
        resnet_kwargs = {
            "model_name": model_name,
            "pretrained": bool(resnet_cfg.get("pretrained", True)),
            "freeze_backbone": bool(resnet_cfg.get("freeze_backbone", False)),
            "pooling_method": str(
                resnet_cfg.get("pooling_method", "spatial_learned_embeddings")
            ),
            "num_spatial_blocks": int(resnet_cfg.get("num_spatial_blocks", 8)),
            "bottleneck_dim": int(resnet_cfg.get("bottleneck_dim", 256)),
        }

    kwargs = dict(
        critic_actions=sample_critic_action,
        encoder_type=str(cfg.sac.encoder_type),
        shared_encoder=bool(cfg.sac.shared_encoder),
        use_proprio=bool(cfg.sac.use_proprio),
        image_keys=image_keys,
        resnet_kwargs=resnet_kwargs,
        critic_network_kwargs={
            "activations": str(cfg.sac.critic_activation),
            "use_layer_norm": bool(cfg.sac.critic_layer_norm),
            "hidden_dims": _to_hidden_dims(cfg.sac.critic_hidden_dims),
        },
        policy_network_kwargs={
            "activations": str(cfg.sac.policy_activation),
            "use_layer_norm": bool(cfg.sac.policy_layer_norm),
            "hidden_dims": _to_hidden_dims(cfg.sac.policy_hidden_dims),
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
        action_transform=action_transform,
        mixed_precision=_build_mixed_precision_kwargs(cfg),
    )
    te = cfg.sac.get("target_entropy", None)
    if te is not None:
        kwargs["target_entropy"] = float(te)

    return DrQAgent.create_drq(
        int(cfg.seed),
        sample_obs,
        sample_action,
        device=device,
        **kwargs,
    )
