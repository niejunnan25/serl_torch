from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

from omegaconf import DictConfig

from serl_launcher.utils.serialization import to_jsonable

RuntimeRole = Literal["actor", "learner"]
EnvBackend = Literal["local", "remote"]


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
class RLTConfig:
    """RLT Stage 2 algorithm configuration."""

    pi0_config_name: str
    pi0_checkpoint_path: str
    rlt_encoder_path: str

    # VLA feature server connection
    vla_server_host: str = "127.0.0.1"
    vla_server_port: int = 8765

    eval_vla_server_host: str = "127.0.0.1"
    eval_vla_server_port: int = 8865

    action_dim: int = 7
    proprio_dim: int = 8
    chunk_size: int = 10
    execute_horizon: int = 5
    z_rl_dim: int = 2048

    utd_ratio: int = 5
    policy_update_freq: int = 2
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    bc_reg_coeff: float = 4.0
    ref_dropout: float = 0.5
    clip_grad_norm: float = 10.0
    num_critics: int = 2
    warmup_steps: int = 500
    batch_size: int = 128

    actor_hidden_dims: tuple[int, ...] = (512, 512, 512)
    critic_hidden_dims: tuple[int, ...] = (512, 512, 512)
    actor_std: float = 0.01
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int
    batch_size: int


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
    max_env_steps: int
    max_update_steps: int
    log_period: int
    checkpoint: CheckpointConfig
    async_eval: AsyncEvalConfig


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    summary_file: str
    episode_log_file: str | None = None
    save_videos: bool = False


@dataclass(frozen=True, slots=True)
class LiberoRLTTrainConfig:
    global_seed: int
    libero_root: str | None
    libero_config_dir: str | None
    libero_datasets_root: str | None
    task: TaskConfig
    runtime: RuntimeConfig
    wandb: WandbConfig
    env: EnvConfig
    rlt: RLTConfig
    replay: ReplayConfig
    training: TrainingConfig
    logging: LoggingConfig


def cfg_to_log_payload(cfg: Any) -> dict[str, Any]:
    payload = to_jsonable(asdict(cfg))
    if not isinstance(payload, dict):
        raise TypeError("typed config payload must serialize to a dict")
    return payload


# ── Parsing helpers ──────────────────────────────────────────────────────


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


def _positive_int(value: Any, field_name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive, got {resolved}")
    return resolved


def _nonnegative_int(value: Any, field_name: str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be >= 0, got {resolved}")
    return resolved


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _float_value(value: Any, field_name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return resolved


def _positive_float(value: Any, field_name: str) -> float:
    resolved = _float_value(value, field_name)
    if resolved <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {resolved}")
    return resolved


def _nonnegative_float(value: Any, field_name: str) -> float:
    resolved = _float_value(value, field_name)
    if resolved < 0.0:
        raise ValueError(f"{field_name} must be >= 0.0, got {resolved}")
    return resolved


def _parse_hidden_dims(values: Any, field_name: str) -> tuple[int, ...]:
    dims = tuple(_positive_int(v, field_name) for v in values)
    if not dims:
        raise ValueError(f"{field_name} must not be empty")
    return dims


# ── Section parsers ──────────────────────────────────────────────────────


def _parse_task_cfg(cfg: DictConfig) -> TaskConfig:
    task_cfg = cfg.get("task", {})
    return TaskConfig(
        suite_name=_required_str(task_cfg.get("suite_name"), "task.suite_name"),
        task_id=_nonnegative_int(task_cfg.get("task_id"), "task.task_id"),
    )


def _parse_runtime_cfg(cfg: DictConfig) -> RuntimeConfig:
    runtime_cfg = cfg.get("runtime", {})
    role = _required_str(runtime_cfg.get("role", "actor"), "runtime.role")
    if role not in ("actor", "learner"):
        raise ValueError(f"runtime.role must be 'actor' or 'learner', got {role!r}")
    return RuntimeConfig(
        role=cast(RuntimeRole, role),
        trainer_host=_required_str(runtime_cfg.get("trainer_host", "127.0.0.1"), "runtime.trainer_host"),
        trainer_port=_positive_int(runtime_cfg.get("trainer_port", 5688), "runtime.trainer_port"),
        broadcast_port=_positive_int(runtime_cfg.get("broadcast_port", 5689), "runtime.broadcast_port"),
        data_store_queue_size=_positive_int(runtime_cfg.get("data_store_queue_size", 2000), "runtime.data_store_queue_size"),
    )


def _parse_wandb_cfg(cfg: DictConfig, *, task: TaskConfig) -> WandbConfig:
    wandb_cfg = cfg.get("wandb", {})
    default_exp_name = f"{task.suite_name}_task_{task.task_id}_rlt"
    return WandbConfig(
        project=_required_str(wandb_cfg.get("project", "libero_rlt"), "wandb.project"),
        exp_name=_required_str(wandb_cfg.get("exp_name", default_exp_name), "wandb.exp_name"),
        group=_optional_str(wandb_cfg.get("group", None)),
        debug=bool(wandb_cfg.get("debug", False)),
    )


def _parse_remote_env_cfg(remote_cfg: Any, *, field_prefix: str) -> RemoteEnvConfig:
    return RemoteEnvConfig(
        host=_required_str(remote_cfg.get("host", "127.0.0.1"), f"{field_prefix}.host"),
        port=_positive_int(remote_cfg.get("port", 30000), f"{field_prefix}.port"),
        timeout_sec=_positive_float(remote_cfg.get("timeout_sec", 120.0), f"{field_prefix}.timeout_sec"),
    )


def _parse_env_cfg(cfg: DictConfig) -> EnvConfig:
    env_cfg = cfg.get("env", {})
    backend = _required_str(env_cfg.get("backend", "remote"), "env.backend")
    if backend not in ("local", "remote"):
        raise ValueError(f"env.backend must be 'local' or 'remote', got {backend!r}")
    remote_cfg = env_cfg.get("remote", {})
    return EnvConfig(
        action_dim=_positive_int(env_cfg.get("action_dim", 7), "env.action_dim"),
        backend=cast(EnvBackend, backend),
        resolution=_positive_int(env_cfg.get("resolution", 256), "env.resolution"),
        num_steps_wait=_positive_int(env_cfg.get("num_steps_wait", 10), "env.num_steps_wait"),
        max_episode_steps=_optional_positive_int(env_cfg.get("max_episode_steps", None), "env.max_episode_steps"),
        seed=_nonnegative_int(env_cfg.get("seed", 0), "env.seed"),
        remote=_parse_remote_env_cfg(remote_cfg, field_prefix="env.remote"),
    )


def _parse_rlt_cfg(cfg: DictConfig) -> RLTConfig:
    rlt_cfg = cfg.get("rlt", {})
    chunk_size = _positive_int(rlt_cfg.get("chunk_size", 10), "rlt.chunk_size")
    execute_horizon = _positive_int(
        rlt_cfg.get("execute_horizon", 5), "rlt.execute_horizon"
    )
    if execute_horizon > chunk_size:
        raise ValueError(
            "rlt.execute_horizon must be <= rlt.chunk_size, "
            f"got execute_horizon={execute_horizon}, chunk_size={chunk_size}"
        )
    return RLTConfig(
        pi0_config_name=_required_str(rlt_cfg.get("pi0_config_name", "pi0_libero"), "rlt.pi0_config_name"),
        pi0_checkpoint_path=_required_str(rlt_cfg.get("pi0_checkpoint_path"), "rlt.pi0_checkpoint_path"),
        rlt_encoder_path=_required_str(rlt_cfg.get("rlt_encoder_path"), "rlt.rlt_encoder_path"),
        vla_server_host=_required_str(rlt_cfg.get("vla_server_host", "127.0.0.1"), "rlt.vla_server_host"),
        vla_server_port=_positive_int(rlt_cfg.get("vla_server_port", 8765), "rlt.vla_server_port"),
        eval_vla_server_host=_required_str(rlt_cfg.get("eval_vla_server_host", "127.0.0.1"), "rlt.eval_vla_server_host"),
        eval_vla_server_port=_positive_int(rlt_cfg.get("eval_vla_server_port", 8865), "rlt.eval_vla_server_port"),
        action_dim=_positive_int(rlt_cfg.get("action_dim", 7), "rlt.action_dim"),
        proprio_dim=_nonnegative_int(rlt_cfg.get("proprio_dim", 8), "rlt.proprio_dim"),
        chunk_size=chunk_size,
        execute_horizon=execute_horizon,
        z_rl_dim=_positive_int(rlt_cfg.get("z_rl_dim", 2048), "rlt.z_rl_dim"),
        utd_ratio=_positive_int(rlt_cfg.get("utd_ratio", 5), "rlt.utd_ratio"),
        policy_update_freq=_positive_int(rlt_cfg.get("policy_update_freq", 2), "rlt.policy_update_freq"),
        discount=_positive_float(rlt_cfg.get("discount", 0.99), "rlt.discount"),
        tau=_positive_float(rlt_cfg.get("tau", 0.005), "rlt.tau"),
        actor_lr=_positive_float(rlt_cfg.get("actor_lr", 3e-4), "rlt.actor_lr"),
        critic_lr=_positive_float(rlt_cfg.get("critic_lr", 3e-4), "rlt.critic_lr"),
        bc_reg_coeff=_nonnegative_float(rlt_cfg.get("bc_reg_coeff", 4.0), "rlt.bc_reg_coeff"),
        ref_dropout=_nonnegative_float(rlt_cfg.get("ref_dropout", 0.5), "rlt.ref_dropout"),
        clip_grad_norm=_positive_float(rlt_cfg.get("clip_grad_norm", 10.0), "rlt.clip_grad_norm"),
        num_critics=_positive_int(rlt_cfg.get("num_critics", 2), "rlt.num_critics"),
        warmup_steps=_nonnegative_int(rlt_cfg.get("warmup_steps", 500), "rlt.warmup_steps"),
        batch_size=_positive_int(rlt_cfg.get("batch_size", 128), "rlt.batch_size"),
        actor_hidden_dims=_parse_hidden_dims(rlt_cfg.get("actor_hidden_dims", [512, 512, 512]), "rlt.actor_hidden_dims"),
        critic_hidden_dims=_parse_hidden_dims(rlt_cfg.get("critic_hidden_dims", [512, 512, 512]), "rlt.critic_hidden_dims"),
        actor_std=_positive_float(rlt_cfg.get("actor_std", 0.01), "rlt.actor_std"),
        device=_required_str(rlt_cfg.get("device", "cuda"), "rlt.device"),
    )


def _parse_replay_cfg(cfg: DictConfig) -> ReplayConfig:
    replay_cfg = cfg.get("replay", {})
    return ReplayConfig(
        capacity=_positive_int(replay_cfg.get("capacity", 100000), "replay.capacity"),
        batch_size=_positive_int(replay_cfg.get("batch_size", 128), "replay.batch_size"),
    )


def _parse_checkpoint_cfg(ckpt_cfg: Any) -> CheckpointConfig:
    return CheckpointConfig(
        every_steps=_positive_int(ckpt_cfg.get("every_steps", 2000), "training.checkpoint.every_steps"),
        keep=_nonnegative_int(ckpt_cfg.get("keep", 5), "training.checkpoint.keep"),
        dir=_required_str(ckpt_cfg.get("dir", "checkpoints"), "training.checkpoint.dir"),
    )


def _parse_async_eval_cfg(cfg: DictConfig) -> AsyncEvalConfig:
    eval_cfg = cfg.get("async_eval", cfg.get("training", {}).get("async_eval", {}))
    ckpt_cfg = eval_cfg.get("checkpoint", {})
    env_cfg = eval_cfg.get("env", {})
    remote_cfg = env_cfg.get("remote", {})
    backend = _required_str(env_cfg.get("backend", "remote"), "async_eval.env.backend")
    return AsyncEvalConfig(
        enabled=bool(eval_cfg.get("enabled", True)),
        every_episodes=_positive_int(eval_cfg.get("every_episodes", 10), "async_eval.every_episodes"),
        episodes=_positive_int(eval_cfg.get("episodes", 5), "async_eval.episodes"),
        start_episode_idx=_nonnegative_int(eval_cfg.get("start_episode_idx", 0), "async_eval.start_episode_idx"),
        max_env_steps_per_episode=_optional_positive_int(eval_cfg.get("max_env_steps_per_episode", None), "async_eval.max_env_steps_per_episode"),
        deterministic=bool(eval_cfg.get("deterministic", True)),
        poll_interval_sec=_positive_float(eval_cfg.get("poll_interval_sec", 5.0), "async_eval.poll_interval_sec"),
        queue_file=_required_str(eval_cfg.get("queue_file", "eval_queue.jsonl"), "async_eval.queue_file"),
        summary_jsonl=_required_str(eval_cfg.get("summary_jsonl", "eval_summary.jsonl"), "async_eval.summary_jsonl"),
        worker_log_file=_required_str(eval_cfg.get("worker_log_file", "eval_worker.log"), "async_eval.worker_log_file"),
        checkpoint=AsyncEvalCheckpointConfig(
            dir=_required_str(ckpt_cfg.get("dir", "eval_checkpoints"), "async_eval.checkpoint.dir"),
            keep=_nonnegative_int(ckpt_cfg.get("keep", 3), "async_eval.checkpoint.keep"),
        ),
        env=AsyncEvalEnvConfig(
            backend=cast(EnvBackend, backend),
            remote=_parse_remote_env_cfg(remote_cfg, field_prefix="async_eval.env.remote"),
        ),
    )


def _parse_training_cfg(cfg: DictConfig) -> TrainingConfig:
    training_cfg = cfg.get("training", {})
    ckpt_cfg = training_cfg.get("checkpoint", {})
    async_eval_cfg = training_cfg.get("async_eval", {})
    return TrainingConfig(
        training_starts=_positive_int(training_cfg.get("training_starts", 500), "training.training_starts"),
        steps_per_update=_positive_int(training_cfg.get("steps_per_update", 1), "training.steps_per_update"),
        max_env_steps=_positive_int(training_cfg.get("max_env_steps", 100000), "training.max_env_steps"),
        max_update_steps=_positive_int(training_cfg.get("max_update_steps", 100000), "training.max_update_steps"),
        log_period=_positive_int(training_cfg.get("log_period", 100), "training.log_period"),
        checkpoint=_parse_checkpoint_cfg(ckpt_cfg),
        async_eval=_parse_async_eval_cfg(cfg),
    )


def _parse_logging_cfg(cfg: DictConfig) -> LoggingConfig:
    logging_cfg = cfg.get("logging", {})
    return LoggingConfig(
        summary_file=_required_str(logging_cfg.get("summary_file", "summary.json"), "logging.summary_file"),
        episode_log_file=_optional_str(logging_cfg.get("episode_log_file", "episode_logs.jsonl")),
        save_videos=bool(logging_cfg.get("save_videos", False)),
    )


def parse_train_cfg(cfg: DictConfig) -> LiberoRLTTrainConfig:
    task = _parse_task_cfg(cfg)
    return LiberoRLTTrainConfig(
        global_seed=_nonnegative_int(cfg.get("global_seed", 42), "global_seed"),
        libero_root=_optional_str(cfg.get("libero_root", None)),
        libero_config_dir=_optional_str(cfg.get("libero_config_dir", None)),
        libero_datasets_root=_optional_str(cfg.get("libero_datasets_root", None)),
        task=task,
        runtime=_parse_runtime_cfg(cfg),
        wandb=_parse_wandb_cfg(cfg, task=task),
        env=_parse_env_cfg(cfg),
        rlt=_parse_rlt_cfg(cfg),
        replay=_parse_replay_cfg(cfg),
        training=_parse_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg),
    )
