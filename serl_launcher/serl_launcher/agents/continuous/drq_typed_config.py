"""Typed-config DRQ agent creation helpers for the current residual training flow."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import get_original_cwd

from serl_launcher.agents.continuous.drq import DrQAgent


def _to_hidden_dims(values: Any) -> list[int]:
    return [int(v) for v in values]


def _resolve_original_cwd() -> Path:
    try:
        return Path(get_original_cwd())
    except Exception:  # noqa: BLE001
        return Path.cwd()


def _optimizer_kwargs(
    cfg: Any,
    *,
    for_temperature: bool = False,
) -> dict[str, Any]:
    opt_cfg = cfg.sac.optimizer
    kwargs: dict[str, Any] = {
        "learning_rate": cfg.sac.learning_rate,
    }
    if opt_cfg.type == "adamw":
        kwargs["weight_decay"] = opt_cfg.weight_decay

    if opt_cfg.warmup_steps is not None:
        kwargs["warmup_steps"] = opt_cfg.warmup_steps
    if opt_cfg.cosine_decay_steps is not None:
        kwargs["cosine_decay_steps"] = opt_cfg.cosine_decay_steps
    if opt_cfg.grad_clip_norm is not None:
        kwargs["clip_grad_norm"] = opt_cfg.grad_clip_norm

    if for_temperature and ("weight_decay" in kwargs):
        if opt_cfg.temperature_weight_decay is None:
            kwargs.pop("weight_decay", None)
        else:
            kwargs["weight_decay"] = opt_cfg.temperature_weight_decay
    return kwargs


def _mixed_precision_kwargs(cfg: Any) -> dict[str, Any]:
    return {
        "enabled": cfg.training.mixed_precision.enabled,
        "dtype": cfg.training.mixed_precision.dtype,
    }


def _proprio_latent_dim(cfg: Any) -> int:
    latent_dim = int(cfg.encoder.proprio_latent_dim)
    if latent_dim <= 0:
        raise ValueError(
            f"encoder.proprio_latent_dim must be positive, got {latent_dim}"
        )
    return latent_dim


def _resnet_kwargs(cfg: Any) -> dict[str, Any] | None:
    resnet_cfg = cfg.encoder.resnet
    if resnet_cfg is None:
        return None

    model_name = resnet_cfg.model_name
    if not Path(model_name).is_absolute() and not model_name.startswith(
        ("http://", "https://")
    ):
        candidate = _resolve_original_cwd() / model_name
        if candidate.is_dir():
            model_name = str(candidate)

    return {
        "model_name": model_name,
        "pretrained": resnet_cfg.pretrained,
        "freeze_backbone": resnet_cfg.freeze_backbone,
        "pooling_method": resnet_cfg.pooling_method,
        "num_spatial_blocks": resnet_cfg.num_spatial_blocks,
        "bottleneck_dim": resnet_cfg.bottleneck_dim,
    }


def create_drq_agent_from_typed_cfg(
    cfg: Any,
    *,
    sample_obs: dict[str, np.ndarray],
    action_dim: int,
    image_keys: tuple[str, ...],
    critic_action_dim: int | None = None,
    action_transform: Any | None = None,
    device: str | torch.device | None = None,
) -> DrQAgent:
    if cfg.encoder.use_proprio and cfg.obs.vector_obs_keys is None:
        raise ValueError(
            "encoder.use_proprio=true requires obs.vector_obs_keys to be configured"
        )

    sample_action = np.zeros((int(action_dim),), dtype=np.float32)
    sample_critic_action = np.zeros(
        (int(critic_action_dim) if critic_action_dim is not None else int(action_dim)),
        dtype=np.float32,
    )
    kwargs: dict[str, Any] = {
        "critic_actions": sample_critic_action,
        "encoder_type": cfg.encoder.type,
        "shared_encoder": cfg.encoder.shared,
        "use_proprio": cfg.encoder.use_proprio,
        "image_keys": tuple(image_keys),
        "vector_obs_keys": cfg.obs.vector_obs_keys,
        "proprio_latent_dim": _proprio_latent_dim(cfg),
        "resnet_kwargs": _resnet_kwargs(cfg),
        "critic_network_kwargs": {
            "activations": cfg.network.critic_activation,
            "use_layer_norm": cfg.network.critic_layer_norm,
            "hidden_dims": _to_hidden_dims(cfg.network.critic_hidden_dims),
        },
        "policy_network_kwargs": {
            "activations": cfg.network.policy_activation,
            "use_layer_norm": cfg.network.policy_layer_norm,
            "hidden_dims": _to_hidden_dims(cfg.network.policy_hidden_dims),
        },
        "policy_kwargs": {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": cfg.sac.std_min,
            "std_max": cfg.sac.std_max,
        },
        "actor_optimizer_kwargs": _optimizer_kwargs(cfg, for_temperature=False),
        "critic_optimizer_kwargs": _optimizer_kwargs(cfg, for_temperature=False),
        "temperature_optimizer_kwargs": _optimizer_kwargs(cfg, for_temperature=True),
        "discount": cfg.sac.discount,
        "soft_target_update_rate": cfg.sac.soft_target_update_rate,
        "backup_entropy": cfg.sac.backup_entropy,
        "otf_num_samples": cfg.sac.otf_num_samples,
        "cql_n_actions": cfg.sac.cql_n_actions,
        "cql_temperature": cfg.sac.cql_temperature,
        "critic_ensemble_size": cfg.sac.critic_ensemble_size,
        "critic_subsample_size": cfg.sac.critic_subsample_size,
        "temperature_init": cfg.sac.temperature_init,
        "action_transform": action_transform,
        "mixed_precision": _mixed_precision_kwargs(cfg),
    }
    if cfg.sac.target_entropy is not None:
        kwargs["target_entropy"] = cfg.sac.target_entropy

    return DrQAgent.create_drq(
        cfg.global_seed,
        sample_obs,
        sample_action,
        device=device,
        **kwargs,
    )
