# !/usr/bin/env python3

from pathlib import Path
from typing import Any, Optional

import tensorflow_datasets as tfds

from agentlace.data.tfds import populate_datastore
from agentlace.trainer import TrainerConfig

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.vice import VICEAgent
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.data.data_store import ReplayBufferDataStore


##############################################################################


def make_bc_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="small",
    resnet_kwargs=None,
):
    return BCAgent.create(
        seed,
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": "tanh",
            "use_layer_norm": False,
            "hidden_dims": [256, 256],
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        resnet_kwargs=resnet_kwargs,
    )


def make_sac_agent(seed, sample_obs, sample_action, discount=0.99):
    return SACAgent.create_states(
        seed,
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": "tanh",
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": "tanh",
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )


def _to_hidden_dims(values) -> list[int]:
    return [int(v) for v in values]


def _optimizer_kwargs_from_cfg(cfg) -> dict[str, Any]:
    sac_cfg = cfg.sac
    optimizer_cfg = sac_cfg.optimizer
    kwargs: dict[str, Any] = {"learning_rate": float(sac_cfg.learning_rate)}

    optimizer_type = str(optimizer_cfg.type).strip().lower()
    if optimizer_type not in {"adam", "adamw"}:
        raise ValueError(f"Unsupported sac.optimizer.type: {optimizer_type}")
    if optimizer_type == "adamw":
        kwargs["weight_decay"] = float(optimizer_cfg.weight_decay)

    warmup_steps = optimizer_cfg.warmup_steps
    if warmup_steps is not None:
        kwargs["warmup_steps"] = int(warmup_steps)
    cosine_decay_steps = optimizer_cfg.cosine_decay_steps
    if cosine_decay_steps is not None:
        kwargs["cosine_decay_steps"] = int(cosine_decay_steps)
    grad_clip_norm = optimizer_cfg.grad_clip_norm
    if grad_clip_norm is not None:
        kwargs["clip_grad_norm"] = float(grad_clip_norm)
    return kwargs


def _temperature_optimizer_kwargs_from_cfg(cfg) -> dict[str, Any]:
    kwargs = _optimizer_kwargs_from_cfg(cfg)
    optimizer_cfg = cfg.sac.optimizer
    if "weight_decay" in kwargs:
        temperature_weight_decay = optimizer_cfg.temperature_weight_decay
        if temperature_weight_decay is None:
            kwargs.pop("weight_decay", None)
        else:
            kwargs["weight_decay"] = float(temperature_weight_decay)
    return kwargs


def _mixed_precision_kwargs_from_cfg(cfg) -> dict[str, Any]:
    mixed_precision_cfg = cfg.training.mixed_precision
    return {
        "enabled": bool(mixed_precision_cfg.enabled),
        "dtype": str(mixed_precision_cfg.dtype),
    }


def _resnet_kwargs_from_cfg(
    cfg, *, original_cwd: Path | None = None
) -> dict[str, Any] | None:
    sac_cfg = cfg.sac
    if str(sac_cfg.encoder_type).strip().lower() not in {"resnet", "resnet-pretrained"}:
        return None
    resnet_cfg = sac_cfg.get("resnet", None)
    if resnet_cfg is None:
        raise ValueError("sac.resnet must be set when sac.encoder_type is resnet")

    model_name = str(resnet_cfg.model_name)
    if not Path(model_name).is_absolute() and not model_name.startswith(
        ("http://", "https://")
    ):
        base_dir = original_cwd or Path.cwd()
        candidate = base_dir / model_name
        if candidate.is_dir():
            model_name = str(candidate)

    return {
        "model_name": model_name,
        "pretrained": bool(resnet_cfg.pretrained),
        "freeze_backbone": bool(resnet_cfg.freeze_backbone),
        "pooling_method": str(resnet_cfg.pooling_method),
        "num_spatial_blocks": int(resnet_cfg.num_spatial_blocks),
        "bottleneck_dim": int(resnet_cfg.bottleneck_dim),
    }


def make_drq_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="small",
    discount=0.96,
    resnet_kwargs=None,
    *,
    cfg=None,
    original_cwd: str | Path | None = None,
    critic_actions=None,
    shared_encoder=True,
    use_proprio=True,
    critic_network_kwargs=None,
    policy_network_kwargs=None,
    policy_kwargs=None,
    actor_optimizer_kwargs=None,
    critic_optimizer_kwargs=None,
    temperature_optimizer_kwargs=None,
    soft_target_update_rate=0.005,
    temperature_init=1e-2,
    backup_entropy=False,
    critic_ensemble_size=10,
    critic_subsample_size=2,
    action_transform=None,
    mixed_precision=None,
    otf_num_samples=None,
    cql_n_actions=None,
    cql_temperature=None,
    target_entropy=None,
):
    if cfg is not None:
        sac_cfg = cfg.sac
        encoder_type = str(sac_cfg.encoder_type)
        discount = float(sac_cfg.discount)
        shared_encoder = bool(sac_cfg.shared_encoder)
        use_proprio = bool(sac_cfg.use_proprio)
        critic_network_kwargs = {
            "activations": str(sac_cfg.critic_activation),
            "use_layer_norm": bool(sac_cfg.critic_layer_norm),
            "hidden_dims": _to_hidden_dims(sac_cfg.critic_hidden_dims),
        }
        policy_network_kwargs = {
            "activations": str(sac_cfg.policy_activation),
            "use_layer_norm": bool(sac_cfg.policy_layer_norm),
            "hidden_dims": _to_hidden_dims(sac_cfg.policy_hidden_dims),
        }
        policy_kwargs = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": float(sac_cfg.std_min),
            "std_max": float(sac_cfg.std_max),
        }
        actor_optimizer_kwargs = _optimizer_kwargs_from_cfg(cfg)
        critic_optimizer_kwargs = _optimizer_kwargs_from_cfg(cfg)
        temperature_optimizer_kwargs = _temperature_optimizer_kwargs_from_cfg(cfg)
        soft_target_update_rate = float(sac_cfg.soft_target_update_rate)
        temperature_init = float(sac_cfg.temperature_init)
        backup_entropy = bool(sac_cfg.backup_entropy)
        critic_ensemble_size = int(sac_cfg.critic_ensemble_size)
        critic_subsample_size = (
            int(sac_cfg.critic_subsample_size)
            if sac_cfg.get("critic_subsample_size", None) is not None
            else None
        )
        resnet_kwargs = _resnet_kwargs_from_cfg(
            cfg,
            original_cwd=Path(original_cwd) if original_cwd is not None else None,
        )
        mixed_precision = _mixed_precision_kwargs_from_cfg(cfg)
        otf_num_samples = int(sac_cfg.otf_num_samples)
        cql_n_actions = int(sac_cfg.cql_n_actions)
        cql_temperature = float(sac_cfg.cql_temperature)
        target_entropy = (
            float(sac_cfg.target_entropy)
            if sac_cfg.get("target_entropy", None) is not None
            else None
        )

    create_kwargs: dict[str, Any] = {
        "encoder_type": encoder_type,
        "shared_encoder": shared_encoder,
        "use_proprio": use_proprio,
        "image_keys": image_keys,
        "resnet_kwargs": resnet_kwargs,
        "policy_kwargs": policy_kwargs
        or {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        "critic_network_kwargs": critic_network_kwargs
        or {
            "activations": "tanh",
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        "policy_network_kwargs": policy_network_kwargs
        or {
            "activations": "tanh",
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        "actor_optimizer_kwargs": actor_optimizer_kwargs or {"learning_rate": 3e-4},
        "critic_optimizer_kwargs": critic_optimizer_kwargs or {"learning_rate": 3e-4},
        "temperature_optimizer_kwargs": temperature_optimizer_kwargs
        or {"learning_rate": 3e-4},
        "discount": discount,
        "soft_target_update_rate": soft_target_update_rate,
        "temperature_init": temperature_init,
        "backup_entropy": backup_entropy,
        "critic_ensemble_size": critic_ensemble_size,
        "critic_subsample_size": critic_subsample_size,
    }
    if critic_actions is not None:
        create_kwargs["critic_actions"] = critic_actions
    if action_transform is not None:
        create_kwargs["action_transform"] = action_transform
    if mixed_precision is not None:
        create_kwargs["mixed_precision"] = mixed_precision
    if otf_num_samples is not None:
        create_kwargs["otf_num_samples"] = otf_num_samples
    if cql_n_actions is not None:
        create_kwargs["cql_n_actions"] = cql_n_actions
    if cql_temperature is not None:
        create_kwargs["cql_temperature"] = cql_temperature
    if target_entropy is not None:
        create_kwargs["target_entropy"] = target_entropy

    return DrQAgent.create_drq(
        seed,
        sample_obs,
        sample_action,
        **create_kwargs,
    )


def make_vice_agent(
    seed,
    sample_obs,
    sample_action,
    sample_vice_obs,
    image_keys=("image",),
    vice_image_keys=("image",),
    encoder_type="small",
    discount=0.96,
    resnet_kwargs=None,
):
    return VICEAgent.create_vice(
        seed,
        sample_obs,
        sample_action,
        sample_vice_obs,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        vice_image_keys=vice_image_keys,
        resnet_kwargs=resnet_kwargs,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": "tanh",
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        vice_network_kwargs={
            "activations": "leaky_relu",
            "use_layer_norm": True,
            "hidden_dims": [256],
            "dropout_rate": 0.1,
        },
        policy_network_kwargs={
            "activations": "tanh",
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )


def make_trainer_config(port_number: int = 5488, broadcast_port: int = 5489):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "agentlace",
    description: str = "serl_launcher",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger


def make_replay_buffer(
    env,
    capacity: int = 1000000,
    rlds_logger_path: Optional[str] = None,
    type: str = "replay_buffer",
    image_keys: list = None,
    preload_rlds_path: Optional[str] = None,
    preload_data_transform: Optional[callable] = None,
):
    image_keys = image_keys or []

    print("shape of observation space and action space")
    print(env.observation_space)
    print(env.action_space)

    if rlds_logger_path:
        from oxe_envlogger.rlds_logger import RLDSLogger

        rlds_logger = RLDSLogger(
            observation_space=env.observation_space,
            action_space=env.action_space,
            dataset_name="serl_rlds_dataset",
            directory=rlds_logger_path,
            max_episodes_per_file=5,
        )
    else:
        rlds_logger = None

    if type == "replay_buffer":
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            rlds_logger=rlds_logger,
        )
    elif type == "memory_efficient_replay_buffer":
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            rlds_logger=rlds_logger,
            image_keys=image_keys,
        )
    else:
        raise ValueError(f"Unsupported replay_buffer_type: {type}")

    if preload_rlds_path:
        print(f" - Preloaded {preload_rlds_path} to replay buffer")
        dataset = tfds.builder_from_directory(preload_rlds_path).as_dataset(split="all")
        populate_datastore(
            replay_buffer,
            dataset,
            data_transform=preload_data_transform,
            type="with_dones",
        )
        print(f" - done populated {len(replay_buffer)} samples to replay buffer")

    return replay_buffer
