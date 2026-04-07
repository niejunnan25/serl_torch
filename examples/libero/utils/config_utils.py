"""Shared Hydra config helpers."""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from serl_launcher.residual.action_spec import resolve_action_mask
from serl_launcher.residual.action_spec import resolve_control_indices
from serl_launcher.residual.observation import normalize_residual_observation_state_mode

from ..schema import resolve_libero_image_keys


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_hidden_dims(values) -> list[int]:
    return [int(v) for v in values]


def build_optimizer_kwargs(
    cfg: DictConfig, *, for_temperature: bool = False
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


def resolve_image_keys(cfg: DictConfig) -> Tuple[str, ...]:
    image_keys_cfg = cfg.residual.get("image_keys", None)
    source = image_keys_cfg if image_keys_cfg is not None else cfg.sac.image_keys
    return resolve_libero_image_keys(str(k) for k in source)


def build_mixed_precision_kwargs(cfg: DictConfig) -> Dict[str, Any]:
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


def resolve_residual_observation_state_mode(cfg: DictConfig) -> str:
    residual_cfg = cfg.get("residual", None)
    observation_cfg = (
        residual_cfg.get("observation", None)
        if residual_cfg is not None
        else None
    )
    state_mode = (
        observation_cfg.get("state_mode", "fused")
        if observation_cfg is not None
        else "fused"
    )
    return normalize_residual_observation_state_mode(state_mode)


def resolve_action_mask_from_cfg(
    cfg: DictConfig, *, full_action_dim: int
) -> np.ndarray:
    action_mask_cfg = cfg.residual.get("action_mask", None)
    action_mask = (
        [bool(v) for v in action_mask_cfg] if action_mask_cfg is not None else None
    )
    return resolve_action_mask(
        full_action_dim=int(full_action_dim),
        action_mask=action_mask,
    )


def resolve_control_indices_from_cfg(
    cfg: DictConfig, *, full_action_dim: int
) -> np.ndarray:
    action_mask = resolve_action_mask_from_cfg(
        cfg,
        full_action_dim=int(full_action_dim),
    )
    return resolve_control_indices(
        full_action_dim=int(full_action_dim),
        action_mask=action_mask,
    )


def build_residual_action_transform(
    *,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    full_action_dim: int,
    chunk_horizon: int,
    chunk_step_enabled: bool,
    clip_gripper: bool,
) -> Dict[str, Any]:
    return {
        "type": "residual_combined",
        "control_indices": [
            int(v) for v in np.asarray(control_indices, dtype=np.int64).reshape(-1)
        ],
        "limits": [
            float(v) for v in np.asarray(residual_limits, dtype=np.float32).reshape(-1)
        ],
        "full_action_dim": int(full_action_dim),
        "chunk_horizon": int(chunk_horizon),
        "chunk_step_enabled": bool(chunk_step_enabled),
        "clip_gripper": bool(clip_gripper),
        "base_action_key": "base_action",
        "base_action_chunk_key": "base_action_chunk",
        "scale_key": "alpha",
    }


def build_drq_agent(
    cfg: DictConfig,
    sample_obs: Dict[str, np.ndarray],
    action_dim: int,
    image_keys: Tuple[str, ...],
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
    actor_optim_kwargs = build_optimizer_kwargs(cfg, for_temperature=False)
    critic_optim_kwargs = build_optimizer_kwargs(cfg, for_temperature=False)
    temp_optim_kwargs = build_optimizer_kwargs(cfg, for_temperature=True)

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
        action_transform=action_transform,
        mixed_precision=build_mixed_precision_kwargs(cfg),
    )
    # Optional SAC knobs (YAML-only ablations). When unset, SAC uses defaults
    # (e.g. target_entropy = -policy_dim/2).
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


def sample_probing_steps(probing_cfg: DictConfig, *, episode_horizon: int) -> int:
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
