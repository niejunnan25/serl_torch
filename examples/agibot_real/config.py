from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Literal
from typing import cast

from omegaconf import DictConfig

from serl_launcher.common.trainer_transport import SUPPORTED_TRANSPORT_MODES
from serl_launcher.common.trainer_transport import TrainerTransportConfig
from serl_launcher.common.trainer_transport import validate_transport_mode
from serl_launcher.utils.serialization import to_jsonable

from .schema import build_agibot_task_key
from .schema import resolve_agibot_image_keys

RuntimeRole = Literal["actor", "learner"]
EnvBackend = Literal["local"]
PolicyBackend = Literal["openpi", "joyra"]
OptimizerType = Literal["adam", "adamw"]


@dataclass(frozen=True, slots=True)
class TaskConfig:
    name: str
    task_key: str
    prompt: str
    control_mode: str
    hz: float
    trajectory_time: float | None
    use_smooth_trajectory: bool
    max_episode_steps: int | None
    reset_hook: str | None
    success_hook: str | None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    role: RuntimeRole
    trainer_host: str
    trainer_port: int
    broadcast_port: int
    data_store_queue_size: int
    trainer_transport: TrainerTransportConfig


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
class BackfillPolicyConfig:
    enabled: bool
    host: str
    port: int
    max_pending_chunks: int
    mode: str


@dataclass(frozen=True, slots=True)
class RobotConfig:
    assets_root: str | None
    retargeter_urdf_path: str | None
    retargeter_camera_extrinsic_path: str | None


@dataclass(frozen=True, slots=True)
class ControllerKeysConfig:
    ready: str
    pause: str
    reset: str
    success: str
    fail: str
    help: str


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    enabled: bool
    interface: str
    poll_interval_sec: float
    terminal_grace_sec: float
    keys: ControllerKeysConfig


@dataclass(frozen=True, slots=True)
class EnvConfig:
    action_dim: int
    backend: EnvBackend


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
    filter_unrepresentable_steps: bool


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
class TrainingConfig:
    training_starts: int
    steps_per_update: int
    critic_actor_ratio: int
    max_env_steps: int
    max_update_steps: int
    max_episodes: int
    log_period: int
    mixed_precision: MixedPrecisionConfig
    torch_compile: TorchCompileConfig
    checkpoint: CheckpointConfig


@dataclass(frozen=True, slots=True)
class EvalTrainingConfig:
    mixed_precision: MixedPrecisionConfig
    torch_compile: TorchCompileConfig


@dataclass(frozen=True, slots=True)
class EvalConfig:
    episodes: int
    max_env_steps_per_episode: int | None
    deterministic: bool
    checkpoint_path: str | None
    checkpoint_step: int | None


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    summary_file: str
    episode_log_file: str | None = None


@dataclass(frozen=True, slots=True)
class VideoConfig:
    enabled: bool
    camera_key: str
    fps: float
    output_dir: str
    max_pending_frames: int
    drop_frames_when_busy: bool


@dataclass(frozen=True, slots=True)
class AgiBotTrainConfig:
    global_seed: int
    task: TaskConfig
    runtime: RuntimeConfig
    wandb: WandbConfig
    policy: PolicyConfig
    backfill_policy: BackfillPolicyConfig
    robot: RobotConfig
    controller: ControllerConfig
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
    video: VideoConfig


@dataclass(frozen=True, slots=True)
class AgiBotEvalConfig:
    global_seed: int
    task: TaskConfig
    policy: PolicyConfig
    robot: RobotConfig
    controller: ControllerConfig
    env: EnvConfig
    obs: ObsConfig
    residual: ResidualConfig
    encoder: EncoderConfig
    network: NetworkConfig
    sac: SacConfig
    training: EvalTrainingConfig
    logging: LoggingConfig
    eval: EvalConfig
    video: VideoConfig


AgiBotRunConfig = AgiBotTrainConfig | AgiBotEvalConfig


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


def _required_mapping(value: Any, field_name: str) -> DictConfig | dict[str, Any]:
    if value is None:
        raise ValueError(
            f"{field_name} must be declared explicitly in the train yaml"
        )
    if not isinstance(value, (DictConfig, dict)):
        raise ValueError(
            f"{field_name} must be a mapping, got {type(value).__name__}"
        )
    return value


def _required_mapping_key(
    mapping: DictConfig | dict[str, Any],
    key: str,
    field_name: str,
) -> Any:
    if key not in mapping:
        raise ValueError(
            f"{field_name} must be declared explicitly in the train yaml"
        )
    return mapping[key]


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


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _float_value(value, field_name)


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, field_name)


def _optional_nonnegative_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_float(value, field_name)


def _parse_prepared_path(values: Any) -> str | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        resolved_values = [
            _required_str(value, "offline.prepared_path")
            for value in values
            if _optional_str(value) is not None
        ]
        if not resolved_values:
            return None
        if len(resolved_values) > 1:
            raise ValueError(
                "offline.prepared_path accepts a single path; "
                f"got {resolved_values!r}"
            )
        return resolved_values[0]
    return _required_str(values, "offline.prepared_path")


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


def _resolve_task_key(task_cfg: Any) -> str:
    explicit_task_key = None
    if hasattr(task_cfg, "get"):
        explicit_task_key = task_cfg.get("task_key", None)
    resolved = _optional_str(explicit_task_key)
    if resolved is not None:
        return resolved
    name = (
        task_cfg.get("name", "default")
        if hasattr(task_cfg, "get")
        else getattr(task_cfg, "name", "default")
    )
    return build_agibot_task_key(str(name))


def _parse_task_cfg(cfg: DictConfig) -> TaskConfig:
    task_cfg = cfg.get("task", {})
    task_name = _required_str(task_cfg.get("name", "agibot_real_default"), "task.name")
    prompt = _required_str(
        task_cfg.get("prompt", task_name),
        "task.prompt",
    )
    control_mode = _required_str(
        task_cfg.get("control_mode", "camera_position"),
        "task.control_mode",
    )
    if control_mode != "camera_position":
        raise ValueError(
            "task.control_mode must currently be 'camera_position' for AgiBot v1, "
            f"got {control_mode!r}"
        )
    return TaskConfig(
        name=task_name,
        task_key=_resolve_task_key(task_cfg),
        prompt=prompt,
        control_mode=control_mode,
        hz=_positive_float(task_cfg.get("hz", 20.0), "task.hz"),
        trajectory_time=_optional_float(
            task_cfg.get("trajectory_time", None),
            "task.trajectory_time",
        ),
        use_smooth_trajectory=bool(task_cfg.get("use_smooth_trajectory", False)),
        max_episode_steps=_optional_positive_int(
            task_cfg.get("max_episode_steps", None),
            "task.max_episode_steps",
        ),
        reset_hook=_optional_str(task_cfg.get("reset_hook", None)),
        success_hook=_optional_str(task_cfg.get("success_hook", None)),
    )


def _parse_runtime_cfg(cfg: DictConfig) -> RuntimeConfig:
    runtime_cfg = cfg.get("runtime", {})
    role = _parse_choice(
        runtime_cfg.get("role", "actor"),
        "runtime.role",
        allowed=("actor", "learner"),
    )
    trainer_port = _positive_int(
        runtime_cfg.get("trainer_port", 5488),
        "runtime.trainer_port",
    )
    transport_cfg = _parse_trainer_transport_cfg(
        runtime_cfg=runtime_cfg,
        default_data_port=int(trainer_port + 2),
    )
    return RuntimeConfig(
        role=cast(RuntimeRole, role),
        trainer_host=_required_str(
            runtime_cfg.get("trainer_host", "127.0.0.1"),
            "runtime.trainer_host",
        ),
        trainer_port=int(trainer_port),
        broadcast_port=_positive_int(
            runtime_cfg.get("broadcast_port", 5489),
            "runtime.broadcast_port",
        ),
        data_store_queue_size=_positive_int(
            runtime_cfg.get("data_store_queue_size", 2000),
            "runtime.data_store_queue_size",
        ),
        trainer_transport=transport_cfg,
    )


def _parse_trainer_transport_cfg(
    *,
    runtime_cfg: DictConfig | dict[str, Any],
    default_data_port: int,
) -> TrainerTransportConfig:
    raw_cfg = runtime_cfg.get("trainer_transport", {})
    raw_mode = _parse_choice(
        raw_cfg.get("mode", "sync_commit"),
        "runtime.trainer_transport.mode",
        allowed=SUPPORTED_TRANSPORT_MODES,
    )
    mode = validate_transport_mode(raw_mode)
    return TrainerTransportConfig(
        mode=mode,
        data_port=_positive_int(
            raw_cfg.get("data_port", default_data_port),
            "runtime.trainer_transport.data_port",
        ),
        control_timeout_ms=_positive_int(
            raw_cfg.get("control_timeout_ms", 800),
            "runtime.trainer_transport.control_timeout_ms",
        ),
        data_queue_capacity=_positive_int(
            raw_cfg.get("data_queue_capacity", 8),
            "runtime.trainer_transport.data_queue_capacity",
        ),
        data_socket_hwm=_positive_int(
            raw_cfg.get("data_socket_hwm", 8),
            "runtime.trainer_transport.data_socket_hwm",
        ),
        commit_poll_ms=_positive_int(
            raw_cfg.get("commit_poll_ms", 20),
            "runtime.trainer_transport.commit_poll_ms",
        ),
        wait_committed_on_episode_end=bool(
            raw_cfg.get("wait_committed_on_episode_end", False)
        ),
        wait_committed_on_shutdown=bool(
            raw_cfg.get("wait_committed_on_shutdown", True)
        ),
    )


def _parse_wandb_cfg(cfg: DictConfig, *, task: TaskConfig) -> WandbConfig:
    wandb_cfg = cfg.get("wandb", {})
    default_exp_name = f"{task.task_key}_residual"
    return WandbConfig(
        project=_required_str(
            wandb_cfg.get("project", "agibot_real"),
            "wandb.project",
        ),
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
    legacy_policy_cfg = cfg.get(str(policy_type), {})
    return PolicyConfig(
        type=cast(PolicyBackend, policy_type),
        host=_required_str(
            policy_cfg.get("host", legacy_policy_cfg.get("host", "127.0.0.1")),
            "policy.host",
        ),
        port=_positive_int(
            policy_cfg.get("port", legacy_policy_cfg.get("port", 9001)),
            "policy.port",
        ),
        id=_optional_str(policy_cfg.get("id", None)),
    )


def _parse_robot_cfg(cfg: DictConfig) -> RobotConfig:
    robot_cfg = cfg.get("robot", {})
    return RobotConfig(
        assets_root=_optional_str(robot_cfg.get("assets_root", None)),
        retargeter_urdf_path=_optional_str(
            robot_cfg.get("retargeter_urdf_path", None)
        ),
        retargeter_camera_extrinsic_path=_optional_str(
            robot_cfg.get("retargeter_camera_extrinsic_path", None)
        ),
    )


def _parse_backfill_policy_cfg(
    cfg: DictConfig,
    *,
    policy: PolicyConfig,
) -> BackfillPolicyConfig:
    backfill_cfg = _required_mapping(
        cfg.get("backfill_policy", None),
        "backfill_policy",
    )
    mode = _parse_choice(
        _required_mapping_key(
            backfill_cfg,
            "mode",
            "backfill_policy.mode",
        ),
        "backfill_policy.mode",
        allowed=("thread",),
    )
    return BackfillPolicyConfig(
        enabled=bool(
            _required_mapping_key(
                backfill_cfg,
                "enabled",
                "backfill_policy.enabled",
            )
        ),
        host=_required_str(
            _required_mapping_key(
                backfill_cfg,
                "host",
                "backfill_policy.host",
            ),
            "backfill_policy.host",
        ),
        port=_positive_int(
            _required_mapping_key(
                backfill_cfg,
                "port",
                "backfill_policy.port",
            ),
            "backfill_policy.port",
        ),
        max_pending_chunks=_positive_int(
            _required_mapping_key(
                backfill_cfg,
                "max_pending_chunks",
                "backfill_policy.max_pending_chunks",
            ),
            "backfill_policy.max_pending_chunks",
        ),
        mode=mode,
    )


def _parse_controller_cfg(cfg: DictConfig) -> ControllerConfig:
    controller_cfg = cfg.get("controller", {})
    keys_cfg = controller_cfg.get("keys", {})
    return ControllerConfig(
        enabled=bool(controller_cfg.get("enabled", False)),
        interface=_required_str(
            controller_cfg.get("interface", "terminal"),
            "controller.interface",
        ),
        poll_interval_sec=_positive_float(
            controller_cfg.get("poll_interval_sec", 0.05),
            "controller.poll_interval_sec",
        ),
        terminal_grace_sec=_nonnegative_float(
            controller_cfg.get("terminal_grace_sec", 0.15),
            "controller.terminal_grace_sec",
        ),
        keys=ControllerKeysConfig(
            ready=_required_str(keys_cfg.get("ready", "g"), "controller.keys.ready"),
            pause=_required_str(keys_cfg.get("pause", "p"), "controller.keys.pause"),
            reset=_required_str(keys_cfg.get("reset", "r"), "controller.keys.reset"),
            success=_required_str(
                keys_cfg.get("success", "s"),
                "controller.keys.success",
            ),
            fail=_required_str(keys_cfg.get("fail", "f"), "controller.keys.fail"),
            help=_required_str(keys_cfg.get("help", "h"), "controller.keys.help"),
        ),
    )


def _validate_canonical_controller_cfg(
    controller: ControllerConfig,
    *,
    context: str,
) -> None:
    if not bool(controller.enabled):
        raise ValueError(
            f"AgiBot canonical {context} flow requires controller.enabled=true"
        )


def _parse_env_cfg(cfg: DictConfig) -> EnvConfig:
    env_cfg = cfg.get("env", {})
    backend = _parse_choice(
        env_cfg.get("backend", "local"),
        "env.backend",
        allowed=("local",),
    )
    action_dim = _positive_int(env_cfg.get("action_dim", None), "env.action_dim")
    if action_dim != 14:
        raise ValueError(
            f"AgiBot v1 currently requires env.action_dim=14, got {action_dim}"
        )
    return EnvConfig(
        action_dim=action_dim,
        backend=cast(EnvBackend, backend),
    )


def _parse_obs_cfg(cfg: DictConfig) -> ObsConfig:
    obs_cfg = cfg.get("obs", {})
    image_keys_source = obs_cfg.get("image_keys", None)
    if image_keys_source is None:
        image_keys_source = resolve_agibot_cfg_image_keys(cfg)
    resolved_image_keys = resolve_agibot_image_keys(
        _required_str(value, "obs.image_keys[]") for value in image_keys_source
    )
    stack_horizon = _positive_int(obs_cfg.get("stack_horizon", 1), "obs.stack_horizon")
    if stack_horizon != 1:
        raise ValueError(
            "obs.stack_horizon must currently be 1 for AgiBot residual DRQ, "
            f"got {stack_horizon}"
        )
    vector_obs_keys = _parse_vector_obs_keys(obs_cfg.get("vector_obs_keys", None))
    if vector_obs_keys is None:
        vector_obs_keys = ("robot_proprio", "base_action_chunk", "alpha")
    return ObsConfig(
        image_keys=tuple(resolved_image_keys),
        vector_obs_keys=vector_obs_keys,
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

    return ResidualConfig(
        alpha=_nonnegative_float(
            residual_cfg.get("alpha", None),
            "residual.alpha",
        ),
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
        type=_required_str(encoder_cfg.get("type", "resnet"), "encoder.type"),
        shared=bool(encoder_cfg.get("shared", True)),
        use_proprio=bool(encoder_cfg.get("use_proprio", True)),
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
            network_cfg.get("policy_hidden_dims", (256, 256, 256)),
            "network.policy_hidden_dims[]",
        ),
        critic_hidden_dims=_parse_hidden_dims(
            network_cfg.get("critic_hidden_dims", (256, 256, 256)),
            "network.critic_hidden_dims[]",
        ),
        policy_activation=_required_str(
            network_cfg.get("policy_activation", "tanh"),
            "network.policy_activation",
        ),
        critic_activation=_required_str(
            network_cfg.get("critic_activation", "tanh"),
            "network.critic_activation",
        ),
        policy_layer_norm=bool(network_cfg.get("policy_layer_norm", True)),
        critic_layer_norm=bool(network_cfg.get("critic_layer_norm", True)),
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


def _parse_offline_prepare_cfg(cfg: DictConfig) -> OfflinePrepareConfig:
    offline_cfg = cfg.get("offline", {})
    prepare_cfg = offline_cfg.get("prepare", {})
    return OfflinePrepareConfig(
        raw_dataset_path=_optional_str(prepare_cfg.get("raw_dataset_path", None)),
        output_root=_required_str(
            prepare_cfg.get("output_root", "data/residual/offline_data"),
            "offline.prepare.output_root",
        ),
        expert_reference_scale=_positive_float(
            prepare_cfg.get("expert_reference_scale", 1.0),
            "offline.prepare.expert_reference_scale",
        ),
        clip_residual_to_unit=bool(
            prepare_cfg.get("clip_residual_to_unit", True)
        ),
        filter_unrepresentable_steps=bool(
            prepare_cfg.get("filter_unrepresentable_steps", False)
        ),
    )


def _parse_offline_cfg(cfg: DictConfig) -> OfflineConfig:
    offline_cfg = cfg.get("offline", {})
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
        enabled=bool(offline_cfg.get("enabled", False)),
        prepared_path=_parse_prepared_path(prepared_path_value),
        ratio=ratio,
        capacity=_positive_int(
            offline_cfg.get("capacity", 50000),
            "offline.capacity",
        ),
        load_max_episodes=_optional_nonnegative_int(
            load_max_episodes_value,
            "offline.load_max_episodes",
        ),
        load_max_transitions=_optional_nonnegative_int(
            load_max_transitions_value,
            "offline.load_max_transitions",
        ),
        pretrain_steps=_nonnegative_int(
            offline_cfg.get("pretrain_steps", 0),
            "offline.pretrain_steps",
        ),
        prepare=_parse_offline_prepare_cfg(cfg),
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
        max_episodes=_positive_int(
            training_cfg.get("max_episodes", 1),
            "training.max_episodes",
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
    )


def _parse_eval_training_cfg(cfg: DictConfig) -> EvalTrainingConfig:
    training_cfg = cfg.get("training", {})
    mixed_precision_cfg = training_cfg.get("mixed_precision", {})
    torch_compile_cfg = training_cfg.get("torch_compile", {})
    return EvalTrainingConfig(
        mixed_precision=MixedPrecisionConfig(
            enabled=bool(mixed_precision_cfg.get("enabled", False)),
            dtype=_required_str(
                mixed_precision_cfg.get("dtype", "bfloat16"),
                "training.mixed_precision.dtype",
            ),
        ),
        torch_compile=_parse_torch_compile_cfg(
            torch_compile_cfg,
            field_prefix="training.torch_compile",
        ),
    )


def _parse_eval_cfg_block(cfg: DictConfig) -> EvalConfig:
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("start_episode_idx", None) is not None:
        raise ValueError(
            "eval.start_episode_idx has been removed; "
            "AgiBot eval now resets every episode to the same initial pose"
        )
    return EvalConfig(
        episodes=_positive_int(eval_cfg.get("episodes", 10), "eval.episodes"),
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


def _parse_video_cfg(cfg: DictConfig, *, default_fps: float) -> VideoConfig:
    video_cfg = cfg.get("video", {})
    return VideoConfig(
        enabled=bool(video_cfg.get("enabled", False)),
        camera_key=_required_str(
            video_cfg.get("camera_key", "image/head"),
            "video.camera_key",
        ),
        fps=_positive_float(video_cfg.get("fps", default_fps), "video.fps"),
        output_dir=_required_str(
            video_cfg.get("output_dir", "videos"),
            "video.output_dir",
        ),
        max_pending_frames=_positive_int(
            video_cfg.get("max_pending_frames", 64),
            "video.max_pending_frames",
        ),
        drop_frames_when_busy=bool(video_cfg.get("drop_frames_when_busy", True)),
    )


def parse_train_cfg(cfg: DictConfig) -> AgiBotTrainConfig:
    task = _parse_task_cfg(cfg)
    env = _parse_env_cfg(cfg)
    runtime = _parse_runtime_cfg(cfg)
    policy = _parse_policy_cfg(cfg)
    backfill_policy = _parse_backfill_policy_cfg(cfg, policy=policy)
    controller = _parse_controller_cfg(cfg)
    _validate_canonical_controller_cfg(controller, context="train")
    obs = _parse_obs_cfg(cfg)
    residual = _parse_residual_cfg(cfg, action_dim=env.action_dim)
    encoder = _parse_encoder_cfg(cfg)

    if encoder.use_proprio and obs.vector_obs_keys is None:
        raise ValueError(
            "encoder.use_proprio=true requires obs.vector_obs_keys to be configured"
        )

    return AgiBotTrainConfig(
        global_seed=_int_value(cfg.get("global_seed", cfg.get("seed", 0)), "global_seed"),
        task=task,
        runtime=runtime,
        wandb=_parse_wandb_cfg(cfg, task=task),
        policy=policy,
        backfill_policy=backfill_policy,
        robot=_parse_robot_cfg(cfg),
        controller=controller,
        env=env,
        obs=obs,
        residual=residual,
        encoder=encoder,
        network=_parse_network_cfg(cfg),
        sac=_parse_sac_cfg(cfg),
        replay=_parse_replay_cfg(cfg),
        offline=_parse_offline_cfg(cfg),
        training=_parse_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg, default_episode_log_file="episode_logs.jsonl"),
        video=_parse_video_cfg(cfg, default_fps=task.hz),
    )


def parse_eval_cfg(cfg: DictConfig) -> AgiBotEvalConfig:
    task = _parse_task_cfg(cfg)
    env = _parse_env_cfg(cfg)
    controller = _parse_controller_cfg(cfg)
    _validate_canonical_controller_cfg(controller, context="eval")
    obs = _parse_obs_cfg(cfg)
    residual = _parse_residual_cfg(cfg, action_dim=env.action_dim)
    encoder = _parse_encoder_cfg(cfg)

    if encoder.use_proprio and obs.vector_obs_keys is None:
        raise ValueError(
            "encoder.use_proprio=true requires obs.vector_obs_keys to be configured"
        )

    return AgiBotEvalConfig(
        global_seed=_int_value(cfg.get("global_seed", cfg.get("seed", 0)), "global_seed"),
        task=task,
        policy=_parse_policy_cfg(cfg),
        robot=_parse_robot_cfg(cfg),
        controller=controller,
        env=env,
        obs=obs,
        residual=residual,
        encoder=encoder,
        network=_parse_network_cfg(cfg),
        sac=_parse_sac_cfg(cfg),
        training=_parse_eval_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg, default_episode_log_file="episode_logs.jsonl"),
        eval=_parse_eval_cfg_block(cfg),
        video=_parse_video_cfg(cfg, default_fps=task.hz),
    )


def resolve_agibot_cfg_image_keys(cfg: DictConfig | AgiBotRunConfig) -> tuple[str, ...]:
    if isinstance(cfg, (AgiBotTrainConfig, AgiBotEvalConfig)):
        return tuple(cfg.obs.image_keys)

    obs_cfg = cfg.get("obs", None)
    if obs_cfg is not None and obs_cfg.get("image_keys", None) is not None:
        source = obs_cfg.get("image_keys")
    else:
        residual_cfg = cfg.get("residual", {})
        image_keys_cfg = residual_cfg.get("image_keys", None)
        source = image_keys_cfg if image_keys_cfg is not None else cfg.get("sac", {}).get(
            "image_keys", ()
        )
    return resolve_agibot_image_keys(str(k) for k in source)


def resolve_agibot_cfg_task_key(cfg: DictConfig | AgiBotRunConfig) -> str:
    if isinstance(cfg, (AgiBotTrainConfig, AgiBotEvalConfig)):
        return str(cfg.task.task_key)
    task_cfg = cfg.get("task", {})
    explicit_task_key = task_cfg.get("task_key", None)
    if explicit_task_key is not None:
        task_key = str(explicit_task_key).strip()
        if task_key:
            return task_key
    return build_agibot_task_key(task_cfg.get("name", "default"))


__all__ = [
    "AgiBotEvalConfig",
    "AgiBotRunConfig",
    "AgiBotTrainConfig",
    "BackfillPolicyConfig",
    "CheckpointConfig",
    "ControllerConfig",
    "ControllerKeysConfig",
    "EncoderConfig",
    "EnvBackend",
    "EnvConfig",
    "EvalConfig",
    "EvalTrainingConfig",
    "LoggingConfig",
    "MixedPrecisionConfig",
    "NetworkConfig",
    "ObsConfig",
    "OfflineConfig",
    "OfflinePrepareConfig",
    "OptimizerConfig",
    "OptimizerType",
    "PolicyBackend",
    "PolicyConfig",
    "ReplayConfig",
    "ResidualConfig",
    "ResnetConfig",
    "RobotConfig",
    "RuntimeConfig",
    "RuntimeRole",
    "SacConfig",
    "TaskConfig",
    "TorchCompileConfig",
    "TrainingConfig",
    "VideoConfig",
    "WandbConfig",
    "cfg_to_log_payload",
    "parse_eval_cfg",
    "parse_train_cfg",
    "resolve_agibot_cfg_image_keys",
    "resolve_agibot_cfg_task_key",
]
