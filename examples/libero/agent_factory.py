from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from serl_launcher.agents.continuous.drq import DrQAgent


def _to_hidden_dims(values: Any) -> list[int]:
    return [int(v) for v in values]


def _resolve_cfg_seed(cfg: DictConfig) -> int:
    seed_value = cfg.get("seed", 0)
    if seed_value is None:
        return 0
    return int(seed_value)


def _resolve_original_cwd() -> Path:
    try:
        return Path(get_original_cwd())
    except Exception:  # noqa: BLE001
        return Path.cwd()


def _optimizer_kwargs(
    cfg: DictConfig,
    *,
    for_temperature: bool = False,
) -> dict[str, Any]:
    sac_cfg = cfg.get("sac", {})
    opt_cfg = sac_cfg.get("optimizer", None)
    kwargs: dict[str, Any] = {
        "learning_rate": float(sac_cfg.get("learning_rate", 3e-4))
    }
    if opt_cfg is None:
        return kwargs

    opt_type = str(opt_cfg.get("type", "adam")).strip().lower()
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
        temp_weight_decay = opt_cfg.get("temperature_weight_decay", None)
        if temp_weight_decay is None:
            kwargs.pop("weight_decay", None)
        else:
            kwargs["weight_decay"] = float(temp_weight_decay)
    return kwargs


def _mixed_precision_kwargs(cfg: DictConfig) -> dict[str, Any]:
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


def _vector_obs_keys(cfg: DictConfig) -> tuple[str, ...] | None:
    sac_cfg = cfg.get("sac", {})
    keys = sac_cfg.get("vector_obs_keys", None)
    if keys is None:
        return None
    resolved = tuple(str(key) for key in keys)
    if len(resolved) == 0:
        raise ValueError("sac.vector_obs_keys must not be empty when configured")
    return resolved


def _proprio_latent_dim(cfg: DictConfig) -> int:
    sac_cfg = cfg.get("sac", {})
    latent_dim = int(sac_cfg.get("proprio_latent_dim", 64))
    if latent_dim <= 0:
        raise ValueError(
            f"sac.proprio_latent_dim must be positive, got {latent_dim}"
        )
    return latent_dim


def _resnet_kwargs(cfg: DictConfig) -> dict[str, Any] | None:
    sac_cfg = cfg.get("sac", {})
    resnet_cfg = sac_cfg.get("resnet", None)
    if resnet_cfg is None:
        return None

    model_name = str(resnet_cfg.get("model_name", "microsoft/resnet-18"))
    if not Path(model_name).is_absolute() and not model_name.startswith(
        ("http://", "https://")
    ):
        candidate = _resolve_original_cwd() / model_name
        if candidate.is_dir():
            model_name = str(candidate)

    return {
        "model_name": model_name,
        "pretrained": bool(resnet_cfg.get("pretrained", True)),
        "freeze_backbone": bool(resnet_cfg.get("freeze_backbone", False)),
        "pooling_method": str(
            resnet_cfg.get("pooling_method", "spatial_learned_embeddings")
        ),
        "num_spatial_blocks": int(resnet_cfg.get("num_spatial_blocks", 8)),
        "bottleneck_dim": int(resnet_cfg.get("bottleneck_dim", 256)),
    }


def make_drq_agent(
    cfg: DictConfig,
    *,
    sample_obs: dict[str, np.ndarray],
    action_dim: int,
    image_keys: tuple[str, ...],
    critic_action_dim: int | None = None,
    action_transform: Any | None = None,
    device: Any = None,
) -> DrQAgent:
    sac_cfg = cfg.get("sac", {})
    vector_obs_keys = _vector_obs_keys(cfg)
    if bool(sac_cfg.get("use_proprio", False)) and vector_obs_keys is None and (
        "state" not in sample_obs
    ):
        raise ValueError(
            "sac.use_proprio=true requires either observations['state'] or "
            "sac.vector_obs_keys to be configured"
        )
    sample_action = np.zeros((int(action_dim),), dtype=np.float32)
    sample_critic_action = np.zeros(
        (int(critic_action_dim) if critic_action_dim is not None else int(action_dim)),
        dtype=np.float32,
    )
    kwargs: dict[str, Any] = {
        "critic_actions": sample_critic_action,
        "encoder_type": str(sac_cfg.get("encoder_type", "small")),
        "shared_encoder": bool(sac_cfg.get("shared_encoder", True)),
        "use_proprio": bool(sac_cfg.get("use_proprio", False)),
        "image_keys": tuple(image_keys),
        "vector_obs_keys": vector_obs_keys,
        "proprio_latent_dim": _proprio_latent_dim(cfg),
        "resnet_kwargs": _resnet_kwargs(cfg),
        "critic_network_kwargs": {
            "activations": str(sac_cfg.get("critic_activation", "relu")),
            "use_layer_norm": bool(sac_cfg.get("critic_layer_norm", False)),
            "hidden_dims": _to_hidden_dims(
                sac_cfg.get("critic_hidden_dims", [256, 256])
            ),
        },
        "policy_network_kwargs": {
            "activations": str(sac_cfg.get("policy_activation", "relu")),
            "use_layer_norm": bool(sac_cfg.get("policy_layer_norm", False)),
            "hidden_dims": _to_hidden_dims(
                sac_cfg.get("policy_hidden_dims", [256, 256])
            ),
        },
        "policy_kwargs": {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": float(sac_cfg.get("std_min", 1e-5)),
            "std_max": float(sac_cfg.get("std_max", 5.0)),
        },
        "actor_optimizer_kwargs": _optimizer_kwargs(cfg, for_temperature=False),
        "critic_optimizer_kwargs": _optimizer_kwargs(cfg, for_temperature=False),
        "temperature_optimizer_kwargs": _optimizer_kwargs(cfg, for_temperature=True),
        "discount": float(sac_cfg.get("discount", 0.99)),
        "soft_target_update_rate": float(
            sac_cfg.get("soft_target_update_rate", 0.005)
        ),
        "backup_entropy": bool(sac_cfg.get("backup_entropy", False)),
        "otf_num_samples": int(sac_cfg.get("otf_num_samples", 1)),
        "cql_n_actions": int(sac_cfg.get("cql_n_actions", 10)),
        "cql_temperature": float(sac_cfg.get("cql_temperature", 1.0)),
        "critic_ensemble_size": int(sac_cfg.get("critic_ensemble_size", 2)),
        "critic_subsample_size": (
            int(sac_cfg.get("critic_subsample_size"))
            if sac_cfg.get("critic_subsample_size", None) is not None
            else None
        ),
        "temperature_init": float(sac_cfg.get("temperature_init", 1.0)),
        "action_transform": action_transform,
        "mixed_precision": _mixed_precision_kwargs(cfg),
    }
    target_entropy = sac_cfg.get("target_entropy", None)
    if target_entropy is not None:
        kwargs["target_entropy"] = float(target_entropy)

    return DrQAgent.create_drq(
        _resolve_cfg_seed(cfg),
        sample_obs,
        sample_action,
        device=device,
        **kwargs,
    )
