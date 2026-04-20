from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from omegaconf import DictConfig

from serl_launcher.utils.serialization import to_jsonable

from ..config import AsyncEvalConfig
from ..config import AsyncEvalEnvConfig
from ..config import EncoderConfig
from ..config import EnvConfig
from ..config import EvalConfig
from ..config import EvalTrainingConfig
from ..config import LoggingConfig
from ..config import NetworkConfig
from ..config import ObsConfig
from ..config import ReplayConfig
from ..config import RuntimeConfig
from ..config import SacConfig
from ..config import TaskConfig
from ..config import TrainingConfig
from ..config import WandbConfig
from ..config import _int_value
from ..config import _nonnegative_float
from ..config import _nonnegative_int
from ..config import _optional_positive_int
from ..config import _optional_str
from ..config import _parse_encoder_cfg
from ..config import _parse_env_cfg
from ..config import _parse_eval_cfg_block
from ..config import _parse_eval_training_cfg
from ..config import _parse_logging_cfg
from ..config import _parse_network_cfg
from ..config import _parse_obs_cfg
from ..config import _parse_prepared_path
from ..config import _parse_replay_cfg
from ..config import _parse_runtime_cfg
from ..config import _parse_sac_cfg
from ..config import _parse_task_cfg
from ..config import _parse_training_cfg
from ..config import _positive_int
from ..config import _required_str


@dataclass(frozen=True, slots=True)
class RLPDOfflinePrepareConfig:
    raw_dataset_path: str | None
    output_root: str


@dataclass(frozen=True, slots=True)
class RLPDOfflineConfig:
    enabled: bool
    prepared_path: str | None
    ratio: float
    capacity: int
    load_max_episodes: int | None
    load_max_transitions: int | None
    pretrain_steps: int
    prepare: RLPDOfflinePrepareConfig


@dataclass(frozen=True, slots=True)
class LiberoRLPDTrainConfig:
    global_seed: int
    libero_root: str | None
    libero_config_dir: str | None
    libero_datasets_root: str | None
    task: TaskConfig
    runtime: RuntimeConfig
    wandb: WandbConfig
    env: EnvConfig
    obs: ObsConfig
    encoder: EncoderConfig
    network: NetworkConfig
    sac: SacConfig
    replay: ReplayConfig
    offline: RLPDOfflineConfig
    training: TrainingConfig
    logging: LoggingConfig


@dataclass(frozen=True, slots=True)
class LiberoRLPDEvalConfig:
    global_seed: int
    libero_root: str | None
    libero_config_dir: str | None
    libero_datasets_root: str | None
    task: TaskConfig
    env: EnvConfig
    obs: ObsConfig
    encoder: EncoderConfig
    network: NetworkConfig
    sac: SacConfig
    training: EvalTrainingConfig
    logging: LoggingConfig
    eval: EvalConfig


LiberoRLPDRunConfig = LiberoRLPDTrainConfig | LiberoRLPDEvalConfig


def cfg_to_log_payload(cfg: Any) -> dict[str, Any]:
    payload = to_jsonable(asdict(cfg))
    if not isinstance(payload, dict):
        raise TypeError("typed config payload must serialize to a dict")
    return payload


def _parse_wandb_cfg(cfg: DictConfig, *, task: TaskConfig) -> WandbConfig:
    wandb_cfg = cfg.get("wandb", {})
    default_exp_name = f"{task.suite_name}_task_{task.task_id}_rlpd"
    return WandbConfig(
        project=_required_str(wandb_cfg.get("project", "libero"), "wandb.project"),
        exp_name=_required_str(
            wandb_cfg.get("exp_name", default_exp_name),
            "wandb.exp_name",
        ),
        group=_optional_str(wandb_cfg.get("group", None)),
        debug=bool(wandb_cfg.get("debug", False)),
    )


def _parse_offline_prepare_cfg(cfg: DictConfig) -> RLPDOfflinePrepareConfig:
    offline_cfg = cfg.get("offline", {})
    prepare_cfg = offline_cfg.get("prepare", {})
    return RLPDOfflinePrepareConfig(
        raw_dataset_path=_optional_str(
            prepare_cfg.get("raw_dataset_path", None),
        ),
        output_root=_required_str(
            prepare_cfg.get(
                "output_root",
                "data/rlpd/offline_data",
            ),
            "offline.prepare.output_root",
        ),
    )


def _parse_offline_cfg(cfg: DictConfig) -> RLPDOfflineConfig:
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
    return RLPDOfflineConfig(
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
        pretrain_steps=_positive_int(
            offline_cfg.get("pretrain_steps", 0),
            "offline.pretrain_steps",
        )
        if int(offline_cfg.get("pretrain_steps", 0)) > 0
        else _nonnegative_int(
            offline_cfg.get("pretrain_steps", 0),
            "offline.pretrain_steps",
        ),
        prepare=_parse_offline_prepare_cfg(cfg),
    )


def _normalize_obs_cfg(obs: ObsConfig) -> ObsConfig:
    if obs.vector_obs_keys is None:
        return replace(obs, vector_obs_keys=("robot_proprio",))
    if tuple(obs.vector_obs_keys) != ("robot_proprio",):
        raise ValueError(
            "obs.vector_obs_keys must be exactly ['robot_proprio'] for LIBERO RLPD"
        )
    return obs


def parse_train_cfg(cfg: DictConfig) -> LiberoRLPDTrainConfig:
    task = _parse_task_cfg(cfg)
    env = _parse_env_cfg(cfg)
    runtime = _parse_runtime_cfg(cfg)
    obs = _normalize_obs_cfg(_parse_obs_cfg(cfg))
    encoder = _parse_encoder_cfg(cfg)
    offline = _parse_offline_cfg(cfg)

    if encoder.use_proprio and obs.vector_obs_keys is None:
        raise ValueError(
            "encoder.use_proprio=true requires obs.vector_obs_keys to be configured"
        )

    return LiberoRLPDTrainConfig(
        global_seed=_int_value(cfg.get("global_seed", 0), "global_seed"),
        libero_root=_optional_str(cfg.get("libero_root", None)),
        libero_config_dir=_optional_str(cfg.get("libero_config_dir", None)),
        libero_datasets_root=_optional_str(cfg.get("libero_datasets_root", None)),
        task=task,
        runtime=runtime,
        wandb=_parse_wandb_cfg(cfg, task=task),
        env=env,
        obs=obs,
        encoder=encoder,
        network=_parse_network_cfg(cfg),
        sac=_parse_sac_cfg(cfg),
        replay=_parse_replay_cfg(cfg),
        offline=offline,
        training=_parse_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg),
    )


def parse_eval_cfg(cfg: DictConfig) -> LiberoRLPDEvalConfig:
    task = _parse_task_cfg(cfg)
    env = _parse_env_cfg(cfg)
    obs = _normalize_obs_cfg(_parse_obs_cfg(cfg))
    encoder = _parse_encoder_cfg(cfg)

    if encoder.use_proprio and obs.vector_obs_keys is None:
        raise ValueError(
            "encoder.use_proprio=true requires obs.vector_obs_keys to be configured"
        )

    return LiberoRLPDEvalConfig(
        global_seed=_int_value(cfg.get("global_seed", 0), "global_seed"),
        libero_root=_optional_str(cfg.get("libero_root", None)),
        libero_config_dir=_optional_str(cfg.get("libero_config_dir", None)),
        libero_datasets_root=_optional_str(cfg.get("libero_datasets_root", None)),
        task=task,
        env=env,
        obs=obs,
        encoder=encoder,
        network=_parse_network_cfg(cfg),
        sac=_parse_sac_cfg(cfg),
        training=_parse_eval_training_cfg(cfg),
        logging=_parse_logging_cfg(cfg, default_episode_log_file="episode_logs.jsonl"),
        eval=_parse_eval_cfg_block(cfg),
    )


def train_cfg_to_eval_cfg(
    train_cfg: LiberoRLPDTrainConfig,
    *,
    eval_cfg: EvalConfig,
    env_override: AsyncEvalEnvConfig | None = None,
    logging: LoggingConfig | None = None,
) -> LiberoRLPDEvalConfig:
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

    return LiberoRLPDEvalConfig(
        global_seed=train_cfg.global_seed,
        libero_root=train_cfg.libero_root,
        libero_config_dir=train_cfg.libero_config_dir,
        libero_datasets_root=train_cfg.libero_datasets_root,
        task=train_cfg.task,
        env=eval_env,
        obs=train_cfg.obs,
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
    "LiberoRLPDEvalConfig",
    "LiberoRLPDRunConfig",
    "LiberoRLPDTrainConfig",
    "RLPDOfflineConfig",
    "RLPDOfflinePrepareConfig",
    "cfg_to_log_payload",
    "parse_eval_cfg",
    "parse_train_cfg",
    "train_cfg_to_eval_cfg",
]
