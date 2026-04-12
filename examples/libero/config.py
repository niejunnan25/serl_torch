from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Literal
from typing import cast

from omegaconf import DictConfig

from serl_launcher.utils.serialization import to_jsonable

from .env.observation import resolve_libero_image_keys

RuntimeRole = Literal["actor", "learner"]
EnvBackend = Literal["local", "remote"]
PolicyBackend = Literal["openpi", "joyra"]
OptimizerType = Literal["adam", "adamw"]


@dataclass(frozen=True, slots=True)
class TaskConfig:
    suite_name: str
    task_id: int


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    role: RuntimeRole
    trainer_host: str
    trainer_port: int
    broadcast_port: int
    data_store_queue_size: int


@dataclass(frozen=True, slots=True)
class WandbConfig:
    project: str
    exp_name: str
    group: str | None
    debug: bool


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    type: PolicyBackend
    host: str
    port: int
    id: str | None = None


@dataclass(frozen=True, slots=True)
class RemoteEnvConfig:
    host: str
    port: int
    timeout_sec: float


@dataclass(frozen=True, slots=True)
class EnvConfig:
    action_dim: int
    backend: EnvBackend
    resolution: int
    num_steps_wait: int
    max_episode_steps: int | None
    seed: int
    remote: RemoteEnvConfig


@dataclass(frozen=True, slots=True)
class ObsConfig:
    image_keys: tuple[str, ...]
    vector_obs_keys: tuple[str, ...] | None
    stack_horizon: int


@dataclass(frozen=True, slots=True)
class ResidualConfig:
    alpha: float
    action_mask: tuple[bool, ...] | None
    action_limits: tuple[float, ...]
    clip_gripper: bool
    chunk_horizon: int


@dataclass(frozen=True, slots=True)
class ResnetConfig:
    model_name: str
    pretrained: bool
    freeze_backbone: bool
    pooling_method: str
    num_spatial_blocks: int
    bottleneck_dim: int


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    type: str
    shared: bool
    use_proprio: bool
    proprio_latent_dim: int
    resnet: ResnetConfig | None


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    policy_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    policy_activation: str
    critic_activation: str
    policy_layer_norm: bool
    critic_layer_norm: bool


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    type: OptimizerType
    weight_decay: float
    temperature_weight_decay: float | None
    warmup_steps: int | None
    cosine_decay_steps: int | None
    grad_clip_norm: float | None


@dataclass(frozen=True, slots=True)
class SacConfig:
    learning_rate: float
    std_min: float
    std_max: float
    discount: float
    soft_target_update_rate: float
    temperature_init: float
    backup_entropy: bool
    critic_ensemble_size: int
    critic_subsample_size: int | None
    utd_ratio: int
    otf_num_samples: int
    cql_n_actions: int
    cql_temperature: float
    target_entropy: float | None
    optimizer: OptimizerConfig


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class MixedPrecisionConfig:
    enabled: bool
    dtype: str


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    every_steps: int
    keep: int
    dir: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    training_starts: int
    steps_per_update: int
    critic_actor_ratio: int
    max_env_steps: int
    max_update_steps: int
    log_period: int
    mixed_precision: MixedPrecisionConfig
    checkpoint: CheckpointConfig


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    summary_file: str


@dataclass(frozen=True, slots=True)
class LiberoTrainConfig:
    global_seed: int
    libero_root: str | None
    libero_config_dir: str | None
    libero_datasets_root: str | None
    task: TaskConfig
    runtime: RuntimeConfig
    wandb: WandbConfig
    policy: PolicyConfig
    env: EnvConfig
    obs: ObsConfig
    residual: ResidualConfig
    encoder: EncoderConfig
    network: NetworkConfig
    sac: SacConfig
    replay: ReplayConfig
    training: TrainingConfig
    logging: LoggingConfig


def cfg_to_log_payload(cfg: LiberoTrainConfig) -> dict[str, Any]:
    payload = to_jsonable(asdict(cfg))
    if not isinstance(payload, dict):
        raise TypeError("typed config payload must serialize to a dict")
    return payload


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip()
    return resolved if resolved else None


def _required_str(value: Any, field_name: str) -> str:
    resolved = _optional_str(value)
    if resolved is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return resolved


def _int_value(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"{field_name} must be an int-like value, got {value!r}"
        ) from exc


def _nonnegative_int(value: Any, field_name: str) -> int:
    resolved = _int_value(value, field_name)
    if resolved < 0:
        raise ValueError(f"{field_name} must be >= 0, got {resolved}")
    return resolved


def _positive_int(value: Any, field_name: str) -> int:
    resolved = _int_value(value, field_name)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {resolved}")
    return resolved


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field_name)


def _float_value(value: Any, field_name: str) -> float:
    try:
        resolved = float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"{field_name} must be a float-like value, got {value!r}"
        ) from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return resolved


def _nonnegative_float(value: Any, field_name: str) -> float:
    resolved = _float_value(value, field_name)
    if resolved < 0.0:
        raise ValueError(f"{field_name} must be >= 0.0, got {resolved}")
    return resolved


def _positive_float(value: Any, field_name: str) -> float:
    resolved = _float_value(value, field_name)
    if resolved <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {resolved}")
    return resolved


def _optional_nonnegative_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_float(value, field_name)


def _parse_choice(
    value: Any,
    field_name: str,
    *,
    allowed: Iterable[str],
) -> str:
    resolved = _required_str(value, field_name).lower()
    allowed_values = {str(item) for item in allowed}
    if resolved not in allowed_values:
        raise ValueError(
            f"{field_name} must be one of {sorted(allowed_values)}, got {value!r}"
        )
    return resolved


def _parse_hidden_dims(values: Any, field_name: str) -> tuple[int, ...]:
    dims = tuple(_positive_int(value, field_name) for value in values)
    if not dims:
        raise ValueError(f"{field_name} must not be empty")
    return dims


def _parse_vector_obs_keys(values: Any) -> tuple[str, ...] | None:
    if values is None:
        return None
    resolved = tuple(_required_str(value, "obs.vector_obs_keys[]") for value in values)
    if not resolved:
        raise ValueError("obs.vector_obs_keys must not be empty when configured")
    return resolved


def _parse_runtime_cfg(cfg: DictConfig) -> RuntimeConfig:
    runtime_cfg = cfg.get("runtime", {})
    role = _parse_choice(
        runtime_cfg.get("role", "actor"),
        "runtime.role",
        allowed=("actor", "learner"),
    )
    return RuntimeConfig(
        role=cast(RuntimeRole, role),
        trainer_host=_required_str(
            runtime_cfg.get("trainer_host", "127.0.0.1"),
            "runtime.trainer_host",
        ),
        trainer_port=_positive_int(
            runtime_cfg.get("trainer_port", 5688),
            "runtime.trainer_port",
        ),
        broadcast_port=_positive_int(
            runtime_cfg.get("broadcast_port", 5689),
            "runtime.broadcast_port",
        ),
        data_store_queue_size=_positive_int(
            runtime_cfg.get("data_store_queue_size", 2000),
            "runtime.data_store_queue_size",
        ),
    )


def _parse_wandb_cfg(cfg: DictConfig, *, task: TaskConfig) -> WandbConfig:
    wandb_cfg = cfg.get("wandb", {})
    default_exp_name = f"{task.suite_name}_task_{task.task_id}_residual"
    return WandbConfig(
        project=_required_str(wandb_cfg.get("project", "libero"), "wandb.project"),
        exp_name=_required_str(
            wandb_cfg.get("exp_name", default_exp_name),
            "wandb.exp_name",
        ),
        group=_optional_str(wandb_cfg.get("group", None)),
        debug=bool(wandb_cfg.get("debug", False)),
    )


def _parse_policy_cfg(cfg: DictConfig) -> PolicyConfig:
    policy_cfg = cfg.get("policy", {})
    policy_type = _parse_choice(
        policy_cfg.get("type", "openpi"),
        "policy.type",
        allowed=("openpi", "joyra"),
    )
    return PolicyConfig(
        type=cast(PolicyBackend, policy_type),
        host=_required_str(policy_cfg.get("host", "localhost"), "policy.host"),
        port=_positive_int(policy_cfg.get("port", 30001), "policy.port"),
        id=_optional_str(policy_cfg.get("id", None)),
    )


def _parse_env_cfg(cfg: DictConfig) -> EnvConfig:
    env_cfg = cfg.get("env", {})
    backend = _parse_choice(
        env_cfg.get("backend", "remote"),
        "env.backend",
        allowed=("local", "remote"),
    )
    remote_cfg = env_cfg.get("remote", {})
    return EnvConfig(
        action_dim=_positive_int(env_cfg.get("action_dim", None), "env.action_dim"),
        backend=cast(EnvBackend, backend),
        resolution=_positive_int(env_cfg.get("resolution", 256), "env.resolution"),
        num_steps_wait=_positive_int(
            env_cfg.get("num_steps_wait", 10),
            "env.num_steps_wait",
        ),
        max_episode_steps=_optional_positive_int(
            env_cfg.get("max_episode_steps", None),
            "env.max_episode_steps",
        ),
        seed=_int_value(env_cfg.get("seed", None), "env.seed"),
        remote=RemoteEnvConfig(
            host=_required_str(remote_cfg.get("host", "127.0.0.1"), "env.remote.host"),
            port=_positive_int(remote_cfg.get("port", 30000), "env.remote.port"),
            timeout_sec=_positive_float(
                remote_cfg.get("timeout_sec", 120.0),
                "env.remote.timeout_sec",
            ),
        ),
    )


def _parse_obs_cfg(cfg: DictConfig) -> ObsConfig:
    obs_cfg = cfg.get("obs", {})
    resolved_image_keys = resolve_libero_image_keys(
        _required_str(value, "obs.image_keys[]")
        for value in obs_cfg.get("image_keys", ())
    )
    stack_horizon = _positive_int(obs_cfg.get("stack_horizon", 1), "obs.stack_horizon")
    if stack_horizon != 1:
        raise ValueError(
            "obs.stack_horizon must currently be 1 for LIBERO residual DRQ, "
            f"got {stack_horizon}"
        )
    return ObsConfig(
        image_keys=tuple(resolved_image_keys),
        vector_obs_keys=_parse_vector_obs_keys(obs_cfg.get("vector_obs_keys", None)),
        stack_horizon=stack_horizon,
    )


def _parse_residual_cfg(cfg: DictConfig, *, action_dim: int) -> ResidualConfig:
    residual_cfg = cfg.get("residual", {})
    action_mask_cfg = residual_cfg.get("action_mask", None)
    action_mask = (
        None
        if action_mask_cfg is None
        else tuple(bool(value) for value in action_mask_cfg)
    )
    if action_mask is not None:
        if len(action_mask) != action_dim:
            raise ValueError(
                "residual.action_mask length mismatch: "
                f"got {len(action_mask)}, expected env.action_dim={action_dim}"
            )
        if not any(action_mask):
            raise ValueError("residual.action_mask must enable at least one action dim")

    action_limits = tuple(
        _nonnegative_float(value, "residual.action_limits[]")
        for value in residual_cfg.get("action_limits", ())
    )
    if len(action_limits) != action_dim:
        raise ValueError(
            "residual.action_limits length mismatch: "
            f"got {len(action_limits)}, expected env.action_dim={action_dim}"
        )

    alpha = _nonnegative_float(residual_cfg.get("alpha", None), "residual.alpha")
    return ResidualConfig(
        alpha=alpha,
        action_mask=action_mask,
        action_limits=action_limits,
        clip_gripper=bool(residual_cfg.get("clip_gripper", True)),
        chunk_horizon=_positive_int(
            residual_cfg.get("chunk_horizon", 1),
            "residual.chunk_horizon",
        ),
    )


def _parse_encoder_cfg(cfg: DictConfig) -> EncoderConfig:
    encoder_cfg = cfg.get("encoder", {})
    resnet_cfg = encoder_cfg.get("resnet", None)
    parsed_resnet = None
    if resnet_cfg is not None:
        parsed_resnet = ResnetConfig(
            model_name=_required_str(
                resnet_cfg.get("model_name", "microsoft/resnet-18"),
                "encoder.resnet.model_name",
            ),
            pretrained=bool(resnet_cfg.get("pretrained", True)),
            freeze_backbone=bool(resnet_cfg.get("freeze_backbone", False)),
            pooling_method=_required_str(
                resnet_cfg.get("pooling_method", "spatial_learned_embeddings"),
                "encoder.resnet.pooling_method",
            ),
            num_spatial_blocks=_positive_int(
                resnet_cfg.get("num_spatial_blocks", 8),
                "encoder.resnet.num_spatial_blocks",
            ),
            bottleneck_dim=_positive_int(
                resnet_cfg.get("bottleneck_dim", 256),
                "encoder.resnet.bottleneck_dim",
            ),
        )
    return EncoderConfig(
        type=_required_str(encoder_cfg.get("type", "small"), "encoder.type"),
        shared=bool(encoder_cfg.get("shared", True)),
        use_proprio=bool(encoder_cfg.get("use_proprio", False)),
        proprio_latent_dim=_positive_int(
            encoder_cfg.get("proprio_latent_dim", 64),
            "encoder.proprio_latent_dim",
        ),
        resnet=parsed_resnet,
    )


def _parse_network_cfg(cfg: DictConfig) -> NetworkConfig:
    network_cfg = cfg.get("network", {})
    return NetworkConfig(
        policy_hidden_dims=_parse_hidden_dims(
            network_cfg.get("policy_hidden_dims", (256, 256)),
            "network.policy_hidden_dims[]",
        ),
        critic_hidden_dims=_parse_hidden_dims(
            network_cfg.get("critic_hidden_dims", (256, 256)),
            "network.critic_hidden_dims[]",
        ),
        policy_activation=_required_str(
            network_cfg.get("policy_activation", "relu"),
            "network.policy_activation",
        ),
        critic_activation=_required_str(
            network_cfg.get("critic_activation", "relu"),
            "network.critic_activation",
        ),
        policy_layer_norm=bool(network_cfg.get("policy_layer_norm", False)),
        critic_layer_norm=bool(network_cfg.get("critic_layer_norm", False)),
    )


def _parse_sac_cfg(cfg: DictConfig) -> SacConfig:
    sac_cfg = cfg.get("sac", {})
    optimizer_cfg = sac_cfg.get("optimizer", {})
    optimizer_type = _parse_choice(
        optimizer_cfg.get("type", "adam"),
        "sac.optimizer.type",
        allowed=("adam", "adamw"),
    )
    critic_ensemble_size = _positive_int(
        sac_cfg.get("critic_ensemble_size", 2),
        "sac.critic_ensemble_size",
    )
    critic_subsample_size = _optional_positive_int(
        sac_cfg.get("critic_subsample_size", None),
        "sac.critic_subsample_size",
    )
    if (
        critic_subsample_size is not None
        and critic_subsample_size > critic_ensemble_size
    ):
        raise ValueError(
            "sac.critic_subsample_size must be <= sac.critic_ensemble_size, "
            f"got {critic_subsample_size} > {critic_ensemble_size}"
        )

    return SacConfig(
        learning_rate=_positive_float(
            sac_cfg.get("learning_rate", 3e-4),
            "sac.learning_rate",
        ),
        std_min=_nonnegative_float(sac_cfg.get("std_min", 1e-5), "sac.std_min"),
        std_max=_positive_float(sac_cfg.get("std_max", 5.0), "sac.std_max"),
        discount=_positive_float(sac_cfg.get("discount", 0.99), "sac.discount"),
        soft_target_update_rate=_positive_float(
            sac_cfg.get("soft_target_update_rate", 0.005),
            "sac.soft_target_update_rate",
        ),
        temperature_init=_positive_float(
            sac_cfg.get("temperature_init", 1.0),
            "sac.temperature_init",
        ),
        backup_entropy=bool(sac_cfg.get("backup_entropy", False)),
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        utd_ratio=_positive_int(sac_cfg.get("utd_ratio", 1), "sac.utd_ratio"),
        otf_num_samples=_positive_int(
            sac_cfg.get("otf_num_samples", 1),
            "sac.otf_num_samples",
        ),
        cql_n_actions=_positive_int(
            sac_cfg.get("cql_n_actions", 10),
            "sac.cql_n_actions",
        ),
        cql_temperature=_positive_float(
            sac_cfg.get("cql_temperature", 1.0),
            "sac.cql_temperature",
        ),
        target_entropy=(
            None
            if sac_cfg.get("target_entropy", None) is None
            else _float_value(sac_cfg.get("target_entropy"), "sac.target_entropy")
        ),
        optimizer=OptimizerConfig(
            type=cast(OptimizerType, optimizer_type),
            weight_decay=_nonnegative_float(
                optimizer_cfg.get("weight_decay", 0.0),
                "sac.optimizer.weight_decay",
            ),
            temperature_weight_decay=_optional_nonnegative_float(
                optimizer_cfg.get("temperature_weight_decay", None),
                "sac.optimizer.temperature_weight_decay",
            ),
            warmup_steps=_optional_nonnegative_int(
                optimizer_cfg.get("warmup_steps", None),
                "sac.optimizer.warmup_steps",
            ),
            cosine_decay_steps=_optional_nonnegative_int(
                optimizer_cfg.get("cosine_decay_steps", None),
                "sac.optimizer.cosine_decay_steps",
            ),
            grad_clip_norm=_optional_positive_float(
                optimizer_cfg.get("grad_clip_norm", None),
                "sac.optimizer.grad_clip_norm",
            ),
        ),
    )


def _parse_replay_cfg(cfg: DictConfig) -> ReplayConfig:
    replay_cfg = cfg.get("replay", {})
    return ReplayConfig(
        capacity=_positive_int(replay_cfg.get("capacity", 250000), "replay.capacity"),
        batch_size=_positive_int(
            replay_cfg.get("batch_size", 128),
            "replay.batch_size",
        ),
    )


def _parse_training_cfg(cfg: DictConfig) -> TrainingConfig:
    training_cfg = cfg.get("training", {})
    mixed_precision_cfg = training_cfg.get("mixed_precision", {})
    checkpoint_cfg = training_cfg.get("checkpoint", {})
    return TrainingConfig(
        training_starts=_nonnegative_int(
            training_cfg.get("training_starts", 0),
            "training.training_starts",
        ),
        steps_per_update=_positive_int(
            training_cfg.get("steps_per_update", 1),
            "training.steps_per_update",
        ),
        critic_actor_ratio=_positive_int(
            training_cfg.get("critic_actor_ratio", 1),
            "training.critic_actor_ratio",
        ),
        max_env_steps=_positive_int(
            training_cfg.get("max_env_steps", 1),
            "training.max_env_steps",
        ),
        max_update_steps=_positive_int(
            training_cfg.get("max_update_steps", 1),
            "training.max_update_steps",
        ),
        log_period=_positive_int(
            training_cfg.get("log_period", 1),
            "training.log_period",
        ),
        mixed_precision=MixedPrecisionConfig(
            enabled=bool(mixed_precision_cfg.get("enabled", False)),
            dtype=_required_str(
                mixed_precision_cfg.get("dtype", "bfloat16"),
                "training.mixed_precision.dtype",
            ),
        ),
        checkpoint=CheckpointConfig(
            every_steps=_nonnegative_int(
                checkpoint_cfg.get("every_steps", 0),
                "training.checkpoint.every_steps",
            ),
            keep=_nonnegative_int(
                checkpoint_cfg.get("keep", 0),
                "training.checkpoint.keep",
            ),
            dir=_required_str(
                checkpoint_cfg.get("dir", "checkpoints"),
                "training.checkpoint.dir",
            ),
        ),
    )


def _parse_logging_cfg(cfg: DictConfig) -> LoggingConfig:
    logging_cfg = cfg.get("logging", {})
    return LoggingConfig(
        summary_file=_required_str(
            logging_cfg.get("summary_file", "summary.json"),
            "logging.summary_file",
        )
    )


def parse_train_cfg(cfg: DictConfig) -> LiberoTrainConfig:
    task = TaskConfig(
        suite_name=_required_str(cfg.task.suite_name, "task.suite_name"),
        task_id=_nonnegative_int(cfg.task.task_id, "task.task_id"),
    )
    env = _parse_env_cfg(cfg)
    runtime = _parse_runtime_cfg(cfg)
    obs = _parse_obs_cfg(cfg)
    residual = _parse_residual_cfg(cfg, action_dim=env.action_dim)
    encoder = _parse_encoder_cfg(cfg)

    if encoder.use_proprio and obs.vector_obs_keys is None:
        raise ValueError(
            "encoder.use_proprio=true requires obs.vector_obs_keys to be configured"
        )

    return LiberoTrainConfig(
        global_seed=_int_value(cfg.get("global_seed", 0), "global_seed"),
        libero_root=_optional_str(cfg.get("libero_root", None)),
        libero_config_dir=_optional_str(cfg.get("libero_config_dir", None)),
        libero_datasets_root=_optional_str(cfg.get("libero_datasets_root", None)),
        task=task,
        runtime=runtime,
        wandb=_parse_wandb_cfg(cfg, task=task),
        policy=_parse_policy_cfg(cfg),
        env=env,
        obs=obs,
        residual=residual,
        encoder=encoder,
        network=_parse_network_cfg(cfg),
        sac=_parse_sac_cfg(cfg),
        replay=_parse_replay_cfg(cfg),
        training=_parse_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg),
    )


__all__ = [
    "CheckpointConfig",
    "EncoderConfig",
    "EnvBackend",
    "EnvConfig",
    "LiberoTrainConfig",
    "LoggingConfig",
    "MixedPrecisionConfig",
    "NetworkConfig",
    "ObsConfig",
    "OptimizerConfig",
    "OptimizerType",
    "PolicyBackend",
    "PolicyConfig",
    "RemoteEnvConfig",
    "ReplayConfig",
    "ResidualConfig",
    "ResnetConfig",
    "RuntimeConfig",
    "RuntimeRole",
    "SacConfig",
    "TaskConfig",
    "TrainingConfig",
    "WandbConfig",
    "cfg_to_log_payload",
    "parse_train_cfg",
]
