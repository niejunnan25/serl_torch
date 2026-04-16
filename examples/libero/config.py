from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
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
class OfflinePrepareConfig:
    raw_dataset_path: str | None
    output_root: str
    expert_reference_scale: float
    clip_residual_to_unit: bool


@dataclass(frozen=True, slots=True)
class OfflineConfig:
    enabled: bool
    prepared_path: str | None
    ratio: float
    capacity: int
    load_max_episodes: int | None
    load_max_transitions: int | None
    pretrain_steps: int
    prepare: OfflinePrepareConfig


@dataclass(frozen=True, slots=True)
class MixedPrecisionConfig:
    enabled: bool
    dtype: str


@dataclass(frozen=True, slots=True)
class TorchCompileConfig:
    enabled: bool
    target: str
    backend: str
    mode: str
    fullgraph: bool
    dynamic: bool


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    every_steps: int
    keep: int
    dir: str


@dataclass(frozen=True, slots=True)
class AsyncEvalCheckpointConfig:
    dir: str
    keep: int


@dataclass(frozen=True, slots=True)
class AsyncEvalEnvConfig:
    backend: EnvBackend
    remote: RemoteEnvConfig


@dataclass(frozen=True, slots=True)
class AsyncEvalConfig:
    enabled: bool
    every_episodes: int
    episodes: int
    start_episode_idx: int
    max_env_steps_per_episode: int | None
    deterministic: bool
    poll_interval_sec: float
    queue_file: str
    summary_jsonl: str
    worker_log_file: str
    checkpoint: AsyncEvalCheckpointConfig
    env: AsyncEvalEnvConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    training_starts: int
    steps_per_update: int
    critic_actor_ratio: int
    max_env_steps: int
    max_update_steps: int
    log_period: int
    mixed_precision: MixedPrecisionConfig
    torch_compile: TorchCompileConfig
    checkpoint: CheckpointConfig
    async_eval: AsyncEvalConfig


@dataclass(frozen=True, slots=True)
class EvalTrainingConfig:
    mixed_precision: MixedPrecisionConfig
    torch_compile: TorchCompileConfig


@dataclass(frozen=True, slots=True)
class EvalConfig:
    episodes: int
    start_episode_idx: int
    max_env_steps_per_episode: int | None
    deterministic: bool
    checkpoint_path: str | None
    checkpoint_step: int | None


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    summary_file: str
    episode_log_file: str | None = None


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
    offline: OfflineConfig
    training: TrainingConfig
    logging: LoggingConfig


@dataclass(frozen=True, slots=True)
class LiberoEvalConfig:
    global_seed: int
    libero_root: str | None
    libero_config_dir: str | None
    libero_datasets_root: str | None
    task: TaskConfig
    policy: PolicyConfig
    env: EnvConfig
    obs: ObsConfig
    residual: ResidualConfig
    encoder: EncoderConfig
    network: NetworkConfig
    sac: SacConfig
    training: EvalTrainingConfig
    logging: LoggingConfig
    eval: EvalConfig


LiberoRunConfig = LiberoTrainConfig | LiberoEvalConfig


def cfg_to_log_payload(cfg: Any) -> dict[str, Any]:
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


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, field_name)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _optional_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_or_default(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except Exception:
        return float(default)
    return resolved if math.isfinite(resolved) else float(default)


def _str_or_default(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    resolved = str(value)
    return resolved if resolved else str(default)


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


def _parse_task_cfg(cfg: DictConfig) -> TaskConfig:
    task_cfg = cfg.get("task", {})
    return TaskConfig(
        suite_name=_required_str(task_cfg.get("suite_name", None), "task.suite_name"),
        task_id=_nonnegative_int(task_cfg.get("task_id", None), "task.task_id"),
    )


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


def _parse_remote_env_cfg(remote_cfg: Any, *, field_prefix: str) -> RemoteEnvConfig:
    return RemoteEnvConfig(
        host=_required_str(
            remote_cfg.get("host", "127.0.0.1"),
            f"{field_prefix}.host",
        ),
        port=_positive_int(
            remote_cfg.get("port", 30000),
            f"{field_prefix}.port",
        ),
        timeout_sec=_positive_float(
            remote_cfg.get("timeout_sec", 120.0),
            f"{field_prefix}.timeout_sec",
        ),
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
        remote=_parse_remote_env_cfg(remote_cfg, field_prefix="env.remote"),
    )


def _parse_async_eval_env_cfg(async_eval_cfg: Any) -> AsyncEvalEnvConfig:
    env_cfg = async_eval_cfg.get("env", {})
    backend = _parse_choice(
        env_cfg.get("backend", "remote"),
        "training.async_eval.env.backend",
        allowed=("local", "remote"),
    )
    return AsyncEvalEnvConfig(
        backend=cast(EnvBackend, backend),
        remote=_parse_remote_env_cfg(
            env_cfg.get("remote", {}),
            field_prefix="training.async_eval.env.remote",
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


def _parse_prepared_path(values: Any) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return _optional_str(values)
    resolved_values = [
        _required_str(value, "offline.prepared_path")
        for value in values
    ]
    if not resolved_values:
        return None
    if len(resolved_values) > 1:
        raise ValueError(
            "offline.prepared_path accepts a single path; "
            f"got {len(resolved_values)} entries"
        )
    return resolved_values[0]


def _parse_offline_prepare_cfg(cfg: DictConfig) -> OfflinePrepareConfig:
    offline_cfg = cfg.get("offline", {})
    prepare_cfg = offline_cfg.get("prepare", {})
    return OfflinePrepareConfig(
        raw_dataset_path=_optional_str(
            prepare_cfg.get("raw_dataset_path", None),
        ),
        output_root=_required_str(
            prepare_cfg.get(
                "output_root",
                "data/residual/offline_data",
            ),
            "offline.prepare.output_root",
        ),
        expert_reference_scale=_positive_float(
            prepare_cfg.get("expert_reference_scale", 1.0),
            "offline.prepare.expert_reference_scale",
        ),
        clip_residual_to_unit=bool(
            prepare_cfg.get("clip_residual_to_unit", True)
        ),
    )


def _parse_offline_cfg(cfg: DictConfig) -> OfflineConfig:
    offline_cfg = cfg.get("offline", {})
    enabled = bool(offline_cfg.get("enabled", False))
    ratio = _nonnegative_float(offline_cfg.get("ratio", 0.5), "offline.ratio")
    if ratio > 1.0:
        raise ValueError(f"offline.ratio must be <= 1.0, got {ratio}")
    prepared_path_value = offline_cfg.get("prepared_path", None)
    if prepared_path_value is None:
        prepared_path_value = offline_cfg.get("prepared_paths", None)
    load_max_episodes_value = offline_cfg.get("load_max_episodes", None)
    if load_max_episodes_value is None:
        load_max_episodes_value = offline_cfg.get("max_episodes", None)
    load_max_transitions_value = offline_cfg.get("load_max_transitions", None)
    if load_max_transitions_value is None:
        load_max_transitions_value = offline_cfg.get("max_transitions", None)
    return OfflineConfig(
        enabled=enabled,
        prepared_path=_parse_prepared_path(prepared_path_value),
        ratio=ratio,
        capacity=_positive_int(
            offline_cfg.get("capacity", 50000),
            "offline.capacity",
        ),
        load_max_episodes=_optional_positive_int(
            load_max_episodes_value,
            "offline.load_max_episodes",
        ),
        load_max_transitions=_optional_positive_int(
            load_max_transitions_value,
            "offline.load_max_transitions",
        ),
        pretrain_steps=_nonnegative_int(
            offline_cfg.get("pretrain_steps", 0),
            "offline.pretrain_steps",
        ),
        prepare=_parse_offline_prepare_cfg(cfg),
    )


def _parse_mixed_precision_cfg(
    mixed_precision_cfg: Any,
    *,
    field_prefix: str,
) -> MixedPrecisionConfig:
    return MixedPrecisionConfig(
        enabled=bool(mixed_precision_cfg.get("enabled", False)),
        dtype=_required_str(
            mixed_precision_cfg.get("dtype", "bfloat16"),
            f"{field_prefix}.dtype",
        ),
    )


def _parse_torch_compile_cfg(
    torch_compile_cfg: Any,
    *,
    field_prefix: str,
) -> TorchCompileConfig:
    target = _parse_choice(
        torch_compile_cfg.get("target", "actor_critic"),
        f"{field_prefix}.target",
        allowed=("critic", "actor_critic"),
    )
    return TorchCompileConfig(
        enabled=bool(torch_compile_cfg.get("enabled", False)),
        target=target,
        backend=_required_str(
            torch_compile_cfg.get("backend", "inductor"),
            f"{field_prefix}.backend",
        ),
        mode=_required_str(
            torch_compile_cfg.get("mode", "default"),
            f"{field_prefix}.mode",
        ),
        fullgraph=bool(torch_compile_cfg.get("fullgraph", True)),
        dynamic=bool(torch_compile_cfg.get("dynamic", False)),
    )


def _parse_async_eval_cfg(cfg: DictConfig) -> AsyncEvalConfig:
    training_cfg = cfg.get("training", {})
    async_eval_cfg = training_cfg.get("async_eval", {})
    if "every_steps" in async_eval_cfg:
        raise ValueError(
            "training.async_eval.every_steps has been removed; "
            "use training.async_eval.every_episodes instead"
        )
    enabled = bool(async_eval_cfg.get("enabled", False))
    every_episodes = _nonnegative_int(
        async_eval_cfg.get("every_episodes", 20),
        "training.async_eval.every_episodes",
    )
    if enabled and every_episodes <= 0:
        raise ValueError(
            "training.async_eval.enabled=true requires training.async_eval.every_episodes > 0"
        )
    if not enabled:
        return AsyncEvalConfig(
            enabled=False,
            every_episodes=every_episodes,
            episodes=_int_or_default(async_eval_cfg.get("episodes", 10), 10),
            start_episode_idx=_int_or_default(
                async_eval_cfg.get("start_episode_idx", 0),
                0,
            ),
            max_env_steps_per_episode=_optional_int_or_none(
                async_eval_cfg.get("max_env_steps_per_episode", None)
            ),
            deterministic=bool(async_eval_cfg.get("deterministic", True)),
            poll_interval_sec=_float_or_default(
                async_eval_cfg.get("poll_interval_sec", 5.0),
                5.0,
            ),
            queue_file=_str_or_default(
                async_eval_cfg.get("queue_file", "async_eval_queue.jsonl"),
                "async_eval_queue.jsonl",
            ),
            summary_jsonl=_str_or_default(
                async_eval_cfg.get("summary_jsonl", "async_eval_results.jsonl"),
                "async_eval_results.jsonl",
            ),
            worker_log_file=_str_or_default(
                async_eval_cfg.get("worker_log_file", "async_eval_worker.log"),
                "async_eval_worker.log",
            ),
            checkpoint=AsyncEvalCheckpointConfig(
                dir=_str_or_default(
                    async_eval_cfg.get("checkpoint", {}).get(
                        "dir", "async_eval_checkpoints"
                    ),
                    "async_eval_checkpoints",
                ),
                keep=_int_or_default(
                    async_eval_cfg.get("checkpoint", {}).get("keep", 0),
                    0,
                ),
            ),
            env=AsyncEvalEnvConfig(
                backend="remote",
                remote=RemoteEnvConfig(
                    host=_str_or_default(
                        async_eval_cfg.get("env", {}).get("remote", {}).get(
                            "host", "127.0.0.1"
                        ),
                        "127.0.0.1",
                    ),
                    port=_int_or_default(
                        async_eval_cfg.get("env", {}).get("remote", {}).get(
                            "port", 30010
                        ),
                        30010,
                    ),
                    timeout_sec=_float_or_default(
                        async_eval_cfg.get("env", {}).get("remote", {}).get(
                            "timeout_sec", 180.0
                        ),
                        180.0,
                    ),
                ),
            ),
        )
    checkpoint_cfg = async_eval_cfg.get("checkpoint", {})
    return AsyncEvalConfig(
        enabled=enabled,
        every_episodes=every_episodes,
        episodes=_positive_int(
            async_eval_cfg.get("episodes", 10),
            "training.async_eval.episodes",
        ),
        start_episode_idx=_nonnegative_int(
            async_eval_cfg.get("start_episode_idx", 0),
            "training.async_eval.start_episode_idx",
        ),
        max_env_steps_per_episode=_optional_positive_int(
            async_eval_cfg.get("max_env_steps_per_episode", None),
            "training.async_eval.max_env_steps_per_episode",
        ),
        deterministic=bool(async_eval_cfg.get("deterministic", True)),
        poll_interval_sec=_positive_float(
            async_eval_cfg.get("poll_interval_sec", 5.0),
            "training.async_eval.poll_interval_sec",
        ),
        queue_file=_required_str(
            async_eval_cfg.get("queue_file", "async_eval_queue.jsonl"),
            "training.async_eval.queue_file",
        ),
        summary_jsonl=_required_str(
            async_eval_cfg.get("summary_jsonl", "async_eval_results.jsonl"),
            "training.async_eval.summary_jsonl",
        ),
        worker_log_file=_required_str(
            async_eval_cfg.get("worker_log_file", "async_eval_worker.log"),
            "training.async_eval.worker_log_file",
        ),
        checkpoint=AsyncEvalCheckpointConfig(
            dir=_required_str(
                checkpoint_cfg.get("dir", "async_eval_checkpoints"),
                "training.async_eval.checkpoint.dir",
            ),
            keep=_nonnegative_int(
                checkpoint_cfg.get("keep", 0),
                "training.async_eval.checkpoint.keep",
            ),
        ),
        env=_parse_async_eval_env_cfg(async_eval_cfg),
    )


def _parse_training_cfg(cfg: DictConfig) -> TrainingConfig:
    training_cfg = cfg.get("training", {})
    mixed_precision_cfg = training_cfg.get("mixed_precision", {})
    torch_compile_cfg = training_cfg.get("torch_compile", {})
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
        mixed_precision=_parse_mixed_precision_cfg(
            mixed_precision_cfg,
            field_prefix="training.mixed_precision",
        ),
        torch_compile=_parse_torch_compile_cfg(
            torch_compile_cfg,
            field_prefix="training.torch_compile",
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
        async_eval=_parse_async_eval_cfg(cfg),
    )


def _parse_eval_training_cfg(cfg: DictConfig) -> EvalTrainingConfig:
    training_cfg = cfg.get("training", {})
    return EvalTrainingConfig(
        mixed_precision=_parse_mixed_precision_cfg(
            training_cfg.get("mixed_precision", {}),
            field_prefix="training.mixed_precision",
        ),
        torch_compile=_parse_torch_compile_cfg(
            training_cfg.get("torch_compile", {}),
            field_prefix="training.torch_compile",
        ),
    )


def _parse_eval_cfg_block(cfg: DictConfig) -> EvalConfig:
    eval_cfg = cfg.get("eval", {})
    return EvalConfig(
        episodes=_positive_int(eval_cfg.get("episodes", 10), "eval.episodes"),
        start_episode_idx=_nonnegative_int(
            eval_cfg.get("start_episode_idx", 0),
            "eval.start_episode_idx",
        ),
        max_env_steps_per_episode=_optional_positive_int(
            eval_cfg.get("max_env_steps_per_episode", None),
            "eval.max_env_steps_per_episode",
        ),
        deterministic=bool(eval_cfg.get("deterministic", True)),
        checkpoint_path=_optional_str(eval_cfg.get("checkpoint_path", None)),
        checkpoint_step=_optional_positive_int(
            eval_cfg.get("checkpoint_step", None),
            "eval.checkpoint_step",
        ),
    )


def _parse_logging_cfg(
    cfg: DictConfig,
    *,
    default_episode_log_file: str | None = None,
) -> LoggingConfig:
    logging_cfg = cfg.get("logging", {})
    return LoggingConfig(
        summary_file=_required_str(
            logging_cfg.get("summary_file", "summary.json"),
            "logging.summary_file",
        ),
        episode_log_file=_optional_str(
            logging_cfg.get("episode_log_file", default_episode_log_file)
        ),
    )


def parse_train_cfg(cfg: DictConfig) -> LiberoTrainConfig:
    task = _parse_task_cfg(cfg)
    env = _parse_env_cfg(cfg)
    runtime = _parse_runtime_cfg(cfg)
    obs = _parse_obs_cfg(cfg)
    residual = _parse_residual_cfg(cfg, action_dim=env.action_dim)
    encoder = _parse_encoder_cfg(cfg)
    offline = _parse_offline_cfg(cfg)

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
        offline=offline,
        training=_parse_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg),
    )


def parse_eval_cfg(cfg: DictConfig) -> LiberoEvalConfig:
    task = _parse_task_cfg(cfg)
    env = _parse_env_cfg(cfg)
    obs = _parse_obs_cfg(cfg)
    residual = _parse_residual_cfg(cfg, action_dim=env.action_dim)
    encoder = _parse_encoder_cfg(cfg)

    if encoder.use_proprio and obs.vector_obs_keys is None:
        raise ValueError(
            "encoder.use_proprio=true requires obs.vector_obs_keys to be configured"
        )

    return LiberoEvalConfig(
        global_seed=_int_value(cfg.get("global_seed", 0), "global_seed"),
        libero_root=_optional_str(cfg.get("libero_root", None)),
        libero_config_dir=_optional_str(cfg.get("libero_config_dir", None)),
        libero_datasets_root=_optional_str(cfg.get("libero_datasets_root", None)),
        task=task,
        policy=_parse_policy_cfg(cfg),
        env=env,
        obs=obs,
        residual=residual,
        encoder=encoder,
        network=_parse_network_cfg(cfg),
        sac=_parse_sac_cfg(cfg),
        training=_parse_eval_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg, default_episode_log_file="episode_logs.jsonl"),
        eval=_parse_eval_cfg_block(cfg),
    )


def train_cfg_to_eval_cfg(
    train_cfg: LiberoTrainConfig,
    *,
    eval_cfg: EvalConfig,
    env_override: AsyncEvalEnvConfig | None = None,
    logging: LoggingConfig | None = None,
) -> LiberoEvalConfig:
    eval_env = (
        train_cfg.env
        if env_override is None
        else replace(
            train_cfg.env,
            backend=env_override.backend,
            remote=env_override.remote,
        )
    )
    eval_logging = logging
    if eval_logging is None:
        eval_logging = LoggingConfig(
            summary_file="summary.json",
            episode_log_file="episode_logs.jsonl",
        )
    elif eval_logging.episode_log_file is None:
        eval_logging = replace(eval_logging, episode_log_file="episode_logs.jsonl")

    return LiberoEvalConfig(
        global_seed=train_cfg.global_seed,
        libero_root=train_cfg.libero_root,
        libero_config_dir=train_cfg.libero_config_dir,
        libero_datasets_root=train_cfg.libero_datasets_root,
        task=train_cfg.task,
        policy=train_cfg.policy,
        env=eval_env,
        obs=train_cfg.obs,
        residual=train_cfg.residual,
        encoder=train_cfg.encoder,
        network=train_cfg.network,
        sac=train_cfg.sac,
        training=EvalTrainingConfig(
            mixed_precision=train_cfg.training.mixed_precision,
            torch_compile=train_cfg.training.torch_compile,
        ),
        logging=eval_logging,
        eval=eval_cfg,
    )


__all__ = [
    "AsyncEvalCheckpointConfig",
    "AsyncEvalConfig",
    "AsyncEvalEnvConfig",
    "CheckpointConfig",
    "EncoderConfig",
    "EnvBackend",
    "EnvConfig",
    "EvalConfig",
    "EvalTrainingConfig",
    "LiberoEvalConfig",
    "LiberoRunConfig",
    "LiberoTrainConfig",
    "LoggingConfig",
    "MixedPrecisionConfig",
    "NetworkConfig",
    "OfflineConfig",
    "OfflinePrepareConfig",
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
    "parse_eval_cfg",
    "parse_train_cfg",
    "train_cfg_to_eval_cfg",
]
