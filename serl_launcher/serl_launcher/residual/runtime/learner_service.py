from __future__ import annotations

"""Standalone agentlace learner for LIBERO residual SAC."""

import logging
import sys
from pathlib import Path
from typing import Callable
from typing import Any, Dict, Optional, Tuple

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

    sys.modules["gym"] = gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from serl_launcher.data.normalizer import StateActionNormalizer, load_normalizer
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.data.training_loader import load_residual_training_buffer
from serl_launcher.residual.runtime.async_learning import _apply_agent_snapshot_payload
from serl_launcher.residual.runtime.async_learning import run_agentlace_learner_service
from serl_launcher.residual.runtime.checkpoint import _snapshot_agent_checkpoint_payload
from serl_launcher.residual.runtime.config_utils import build_drq_agent
from serl_launcher.residual.runtime.config_utils import build_residual_action_transform
from serl_launcher.residual.runtime.config_utils import resolve_control_indices_from_cfg
from serl_launcher.residual.runtime.config_utils import resolve_residual_observation_state_mode
from serl_launcher.residual.runtime.obs_utils import _obs_space_from_sample
from serl_launcher.residual.runtime.pretrain import _pretrain_critic_with_calql
from serl_launcher.residual.runtime.profiling import _RuntimeProfiler
from serl_launcher.residual.runtime.schedules import _scheduled_alpha
from serl_launcher.residual.runtime.step_chunk_replay import ChunkReplayBuffer
from serl_launcher.utils.agentlace_io import resolve_agentlace_bootstrap_path
from serl_launcher.utils.agentlace_io import wait_for_agentlace_bootstrap
from serl_launcher.utils.logger import JsonlLogger
from serl_launcher.utils.serialization import _to_jsonable
from torch.utils.tensorboard import SummaryWriter

from serl_launcher.data.replay_buffer import ReplayBuffer


class _LearnerStatsLogger:
    """Persist actor-originated stats into the learner's normal logging stack."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        step_logger: JsonlLogger,
        episode_logger: JsonlLogger,
        tb_writer: SummaryWriter,
        replay_buffer: Any,
        offline_buffer: Optional[Any],
    ) -> None:
        self.logger = logger
        self.step_logger = step_logger
        self.episode_logger = episode_logger
        self.tb_writer = tb_writer
        self.replay_buffer = replay_buffer
        self.offline_buffer = offline_buffer

    def _log_train_episode(self, payload: Dict[str, Any], *, update_steps: int) -> None:
        train_episode_id = int(payload.get("train_episode_id", 0))
        train_env_step = int(payload.get("train_env_step", 0))
        decision_step = payload.get("decision_step", None)
        record = {
            "phase": str(payload.get("phase", "train")),
            "warmup_episode_id": None,
            "train_episode_id": train_episode_id,
            "phase_episode_idx": train_episode_id,
            "seed": None,
            "init_state_idx": None,
            "success": bool(payload.get("success", False)),
            "episode_steps": int(payload.get("episode_steps", 0)),
            "episode_return": float(payload.get("episode_return", 0.0)),
            "train_env_step": train_env_step,
            "decision_step": (
                None if decision_step is None else int(decision_step)
            ),
            "running_success_rate": payload.get("running_success_rate", None),
            "recent_success_rate": payload.get("recent_success_rate", None),
        }
        self.episode_logger.write(record)
        self.tb_writer.add_scalar(
            "train_episode/success",
            int(record["success"]),
            train_episode_id,
        )
        self.tb_writer.add_scalar(
            "train_episode/return",
            float(record["episode_return"]),
            train_episode_id,
        )
        self.tb_writer.add_scalar(
            "train_episode/length",
            int(record["episode_steps"]),
            train_episode_id,
        )
        if record["running_success_rate"] is not None:
            self.tb_writer.add_scalar(
                "train_episode/running_success_rate",
                float(record["running_success_rate"]),
                train_episode_id,
            )
        if record["recent_success_rate"] is not None:
            self.tb_writer.add_scalar(
                "train_episode/recent_success_rate_20",
                float(record["recent_success_rate"]),
                train_episode_id,
            )
        self.tb_writer.add_scalar(
            "system/online_buffer_size",
            int(len(self.replay_buffer)),
            train_env_step,
        )
        if self.offline_buffer is not None:
            self.tb_writer.add_scalar(
                "system/offline_buffer_size",
                int(len(self.offline_buffer)),
                train_env_step,
            )
        if decision_step is not None:
            self.tb_writer.add_scalar(
                "system/decision_step",
                int(decision_step),
                train_env_step,
            )
        self.tb_writer.add_scalar(
            "system/learner_update_steps",
            int(update_steps),
            train_env_step,
        )
        self.logger.info(
            "agentlace train_episode=%s success=%s steps=%s return=%.2f "
            "train_env_step=%s learner_update_steps=%s",
            train_episode_id,
            bool(record["success"]),
            int(record["episode_steps"]),
            float(record["episode_return"]),
            train_env_step,
            int(update_steps),
        )

    def _log_timer_payload(self, payload: Dict[str, Any], *, update_steps: int) -> None:
        self.step_logger.write(
            {
                "record_type": "agentlace_timer",
                "learner_update_steps": int(update_steps),
                "payload": _to_jsonable(payload),
            }
        )
        for key, value in payload.items():
            if isinstance(value, (int, float, np.number)):
                self.tb_writer.add_scalar(
                    f"agentlace_timer/{key}",
                    float(value),
                    int(update_steps),
                )

    def handle_payload(
        self,
        payload: Dict[str, Any],
        update_steps: int,
        last_update_info: Dict[str, Any],
        replay_buffer: Any,
        offline_buffer: Optional[Any],
    ) -> None:
        del last_update_info
        self.replay_buffer = replay_buffer
        self.offline_buffer = offline_buffer

        handled = set()
        train_episode = payload.get("train_episode", None)
        if isinstance(train_episode, dict):
            self._log_train_episode(train_episode, update_steps=int(update_steps))
            handled.add("train_episode")

        timer_payload = payload.get("timer", None)
        if isinstance(timer_payload, dict):
            self._log_timer_payload(timer_payload, update_steps=int(update_steps))
            handled.add("timer")

        remaining = {key: value for key, value in payload.items() if key not in handled}
        if remaining:
            self.step_logger.write(
                {
                    "record_type": "agentlace_stats",
                    "learner_update_steps": int(update_steps),
                    "payload": _to_jsonable(remaining),
                }
            )


def _build_online_replay(
    cfg: DictConfig,
    *,
    sample_obs: Dict[str, np.ndarray],
    state_core_dim: int,
    critic_action_dim: int,
    env_action_dim: int,
    chunk_horizon: int,
    chunk_step_enabled: bool,
    state_mode: str,
) -> Any:
    chunk_step_cfg = cfg.get("chunk_step", None)
    chunk_step_sample_stride = (
        int(chunk_step_cfg.get("sample_stride", 1)) if chunk_step_cfg is not None else 1
    )
    chunk_step_require_full_horizon = (
        bool(chunk_step_cfg.get("require_full_horizon", False))
        if chunk_step_cfg is not None
        else False
    )
    chunk_step_pad_action = (
        bool(chunk_step_cfg.get("pad_action_to_horizon", True))
        if chunk_step_cfg is not None
        else True
    )
    if chunk_step_enabled:
        return ChunkReplayBuffer(
            sample_observation_template=sample_obs,
            state_core_dim=int(state_core_dim),
            step_action_dim=int(env_action_dim),
            chunk_horizon=int(chunk_horizon),
            discount=float(cfg.sac.discount),
            capacity=int(cfg.replay.capacity),
            sample_stride=chunk_step_sample_stride,
            require_full_horizon=chunk_step_require_full_horizon,
            pad_action_to_horizon=chunk_step_pad_action,
            state_mode=str(state_mode),
        )

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(int(critic_action_dim),),
        dtype=np.float32,
    )
    return ReplayBuffer(
        observation_space=_obs_space_from_sample(sample_obs),
        action_space=action_space,
        capacity=int(cfg.replay.capacity),
    )


def _build_offline_replay(
    cfg: DictConfig,
    *,
    sample_obs: Dict[str, np.ndarray],
    state_core_dim: int,
    critic_action_dim: int,
    env_action_dim: int,
    chunk_horizon: int,
    chunk_step_enabled: bool,
    state_mode: str,
) -> Any:
    chunk_step_cfg = cfg.get("chunk_step", None)
    chunk_step_sample_stride = (
        int(chunk_step_cfg.get("sample_stride", 1)) if chunk_step_cfg is not None else 1
    )
    chunk_step_require_full_horizon = (
        bool(chunk_step_cfg.get("require_full_horizon", False))
        if chunk_step_cfg is not None
        else False
    )
    chunk_step_pad_action = (
        bool(chunk_step_cfg.get("pad_action_to_horizon", True))
        if chunk_step_cfg is not None
        else True
    )
    if chunk_step_enabled:
        return ChunkReplayBuffer(
            sample_observation_template=sample_obs,
            state_core_dim=int(state_core_dim),
            step_action_dim=int(env_action_dim),
            chunk_horizon=int(chunk_horizon),
            discount=float(cfg.sac.discount),
            capacity=int(cfg.offline.capacity),
            sample_stride=chunk_step_sample_stride,
            require_full_horizon=chunk_step_require_full_horizon,
            pad_action_to_horizon=chunk_step_pad_action,
            state_mode=str(state_mode),
        )

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(int(critic_action_dim),),
        dtype=np.float32,
    )
    return ReplayBuffer(
        observation_space=_obs_space_from_sample(sample_obs),
        action_space=action_space,
        capacity=int(cfg.offline.capacity),
    )


def run_residual_learner_service(
    cfg: DictConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
    data_config: Any,
    resolve_cfg_image_keys: Callable[[DictConfig], tuple[str, ...]],
) -> None:
    async_cfg = cfg.training.get("async", None)
    async_enabled = (
        bool(async_cfg.get("enabled", False)) if async_cfg is not None else False
    )
    async_backend = (
        str(async_cfg.get("backend", "thread")).strip().lower()
        if async_cfg is not None
        else "thread"
    )
    if not async_enabled or async_backend != "agentlace":
        raise ValueError(
            "run_learner.py requires training.async.enabled=true and "
            "training.async.backend=agentlace"
        )

    async_update_frequency = (
        int(async_cfg.get("update_frequency", 1)) if async_cfg is not None else 1
    )
    async_idle_sleep_sec = (
        float(async_cfg.get("idle_sleep_sec", 0.002))
        if async_cfg is not None
        else 0.002
    )
    async_learner_device = (
        str(async_cfg.get("learner_device")).strip()
        if async_cfg is not None and async_cfg.get("learner_device", None) is not None
        else None
    )
    async_trainer_host = (
        str(async_cfg.get("trainer_host", "127.0.0.1"))
        if async_cfg is not None
        else "127.0.0.1"
    )
    async_trainer_port = (
        int(async_cfg.get("trainer_port", 5488)) if async_cfg is not None else 5488
    )
    async_broadcast_port = (
        int(async_cfg.get("broadcast_port", 5489)) if async_cfg is not None else 5489
    )
    async_agentlace_cfg = (
        async_cfg.get("agentlace", None) if async_cfg is not None else None
    )
    async_agentlace_bootstrap_file = (
        str(async_agentlace_cfg.get("bootstrap_file", "agentlace_bootstrap.pkl"))
        if async_agentlace_cfg is not None
        else "agentlace_bootstrap.pkl"
    )
    async_agentlace_connect_timeout_sec = (
        float(async_agentlace_cfg.get("connect_timeout_sec", 120.0))
        if async_agentlace_cfg is not None
        else 120.0
    )
    profiling_cfg = cfg.training.get("profiling", None)
    profiling_enabled = (
        bool(profiling_cfg.get("enabled", False))
        if profiling_cfg is not None
        else False
    )
    profiling_window_size = (
        int(profiling_cfg.get("window_size", 2048))
        if profiling_cfg is not None
        else 2048
    )
    profiling_log_period_steps = (
        int(profiling_cfg.get("log_period_steps", 500))
        if profiling_cfg is not None
        else 500
    )
    profiling_log_file = (
        str(profiling_cfg.get("log_file", "profiling_logs.jsonl"))
        if profiling_cfg is not None
        else "profiling_logs.jsonl"
    )

    if not Path(async_agentlace_bootstrap_file).expanduser().is_absolute():
        logger.warning(
            "training.async.agentlace.bootstrap_file is relative (%s). "
            "For a standalone learner, an absolute path is recommended.",
            async_agentlace_bootstrap_file,
        )
    bootstrap_path = resolve_agentlace_bootstrap_path(
        run_dir=run_dir,
        bootstrap_file=async_agentlace_bootstrap_file,
    )
    logger.info("Waiting for actor bootstrap: %s", bootstrap_path)
    bootstrap = wait_for_agentlace_bootstrap(
        bootstrap_path,
        timeout_sec=async_agentlace_connect_timeout_sec,
    )
    logger.info("Loaded bootstrap payload from %s", bootstrap_path)

    sample_obs = dict(bootstrap["sample_obs"])
    state_core_dim = int(bootstrap["state_core_dim"])
    env_action_dim = int(bootstrap["env_action_dim"])
    step_action_dim = int(bootstrap["step_action_dim"])
    agent_action_dim = int(bootstrap["agent_action_dim"])
    critic_action_dim = int(bootstrap["critic_action_dim"])
    image_keys = tuple(bootstrap.get("image_keys", resolve_cfg_image_keys(cfg)))
    chunk_step_enabled = bool(bootstrap.get("chunk_step_enabled", False))
    chunk_horizon = int(bootstrap.get("chunk_horizon", int(cfg.residual.chunk_horizon)))
    state_mode = str(
        bootstrap.get(
            "state_mode",
            resolve_residual_observation_state_mode(cfg),
        )
    )
    control_indices = resolve_control_indices_from_cfg(
        cfg, full_action_dim=int(env_action_dim)
    )
    residual_limits = build_residual_limits(
        control_indices,
        full_action_dim=int(env_action_dim),
        action_limits=cfg.residual.get("action_limits", None),
    )
    action_transform = bootstrap.get("action_transform", None)
    if action_transform is None:
        action_transform = build_residual_action_transform(
            control_indices=control_indices,
            residual_limits=residual_limits,
            full_action_dim=int(env_action_dim),
            chunk_horizon=int(chunk_horizon),
            chunk_step_enabled=bool(chunk_step_enabled),
            clip_gripper=bool(cfg.residual.get("clip_gripper", True)),
        )

    task_key = f"{cfg.task.suite_name}_task_{int(cfg.task.task_id)}"
    normalizer: StateActionNormalizer | None = None
    norm_cfg = cfg.get("normalization", None)
    if norm_cfg is not None and bool(norm_cfg.get("enabled", False)):
        stats_dir = norm_cfg.get(
            "stats_dir",
            str(Path(__file__).resolve().parents[1] / "data" / "stats"),
        )
        normalizer = load_normalizer(
            task_key, stats_dir=stats_dir
        )
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)

    replay_buffer = _build_online_replay(
        cfg,
        sample_obs=sample_obs,
        state_core_dim=state_core_dim,
        critic_action_dim=critic_action_dim,
        env_action_dim=env_action_dim,
        chunk_horizon=chunk_horizon,
        chunk_step_enabled=chunk_step_enabled,
        state_mode=state_mode,
    )

    learner_agent = build_drq_agent(
        cfg,
        sample_obs=sample_obs,
        action_dim=int(agent_action_dim),
        image_keys=tuple(image_keys),
        critic_action_dim=int(critic_action_dim),
        action_transform=action_transform,
        device=async_learner_device,
    )
    bootstrap_initial_payload = bootstrap.get("initial_agent_payload", None)
    if isinstance(bootstrap_initial_payload, dict):
        _apply_agent_snapshot_payload(
            learner_agent,
            bootstrap_initial_payload,
            load_optimizers=True,
        )

    offline_buffer = None
    learner_env = None
    online_prefill_stats: Dict[str, Any] = {
        "enabled": 0,
        "mode": "stepchunk" if chunk_step_enabled else "step",
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "episodes_loaded": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
        "success_episodes": 0,
        "episode_return_sum": 0.0,
        "episode_step_sum": 0,
        "recent_episode_successes": [],
    }
    offline_enabled = bool(cfg.offline.enabled)
    if offline_enabled:
        offline_dataset_paths_cfg = cfg.offline.get("dataset_paths", None)
        has_offline_dataset_paths = bool(offline_dataset_paths_cfg) and len(
            offline_dataset_paths_cfg
        ) > 0
        if not has_offline_dataset_paths:
            raise ValueError(
                "offline.enabled=true requires offline.dataset_paths to be set "
                "because offline bootstrap has been removed"
            )
        offline_buffer = _build_offline_replay(
            cfg,
            sample_obs=sample_obs,
            state_core_dim=state_core_dim,
            critic_action_dim=critic_action_dim,
            env_action_dim=env_action_dim,
            chunk_horizon=chunk_horizon,
            chunk_step_enabled=chunk_step_enabled,
            state_mode=state_mode,
        )
        offline_residual_alpha = _scheduled_alpha(
            cfg,
            base_alpha=float(cfg.residual.alpha),
            schedule_step=0,
        )
        offline_stats = load_residual_training_buffer(
            cfg.offline.dataset_paths,
            sample_obs_template=sample_obs,
            replay_buffer=offline_buffer,
            action_dim=int(env_action_dim),
            chunk_horizon=int(chunk_horizon),
            image_keys=tuple(image_keys),
            stack_horizon=int(cfg.sac.obs_stack_horizon),
            chunk_step_enabled=bool(chunk_step_enabled),
            logger=logger,
            data_config=data_config,
            normalizer=normalizer,
            profiler=None,
            state_mode=state_mode,
            max_transitions=cfg.offline.max_transitions,
            expected_task_key=task_key,
            expected_alpha=float(offline_residual_alpha),
            expected_projection={
                "control_indices": control_indices,
                "residual_limits": residual_limits,
                "expert_reference_scale": float(
                    cfg.offline.get("expert_reference_scale", 1.0)
                ),
                "clip_residual_to_unit": bool(
                    cfg.offline.get("clip_residual_to_unit", True)
                ),
            },
            dataset_label="offline residual training",
        )
        logger.info(
            "offline preload: buffer=%s files_loaded=%s/%s candidates=%s inserted=%s "
            "skipped=%s clipped=%s errors=%s",
            len(offline_buffer),
            offline_stats.get("files_loaded", 0),
            offline_stats.get("files_total", 0),
            offline_stats.get("candidates", 0),
            offline_stats.get("inserted", 0),
            offline_stats.get("skipped", 0),
            offline_stats.get("clipped_values", 0),
            offline_stats.get("errors", 0),
        )

    warmup_cfg = cfg.training.get("warmup", None)
    configured_warmup_episodes = (
        int(warmup_cfg.get("episodes", 0)) if warmup_cfg is not None else 0
    )
    online_prefill_cfg = cfg.training.get("online_prefill", None)
    online_prefill_enabled = (
        bool(online_prefill_cfg.get("enabled", False))
        if online_prefill_cfg is not None
        else False
    )
    if online_prefill_enabled and configured_warmup_episodes > 0:
        online_prefill_stats = load_residual_training_buffer(
            online_prefill_cfg.get("dataset_paths", None),
            replay_buffer=replay_buffer,
            sample_obs_template=sample_obs,
            action_dim=int(env_action_dim),
            chunk_horizon=int(chunk_horizon),
            image_keys=tuple(image_keys),
            stack_horizon=int(cfg.sac.obs_stack_horizon),
            chunk_step_enabled=bool(chunk_step_enabled),
            logger=logger,
            data_config=data_config,
            normalizer=normalizer,
            profiler=None,
            max_episodes=int(configured_warmup_episodes),
            state_mode=state_mode,
            expected_task_key=task_key,
            expected_alpha=0.0,
            dataset_label="online residual training",
        )
        online_prefill_loaded_episodes = int(
            online_prefill_stats.get("episodes_loaded", 0)
        )
        if (
            online_prefill_loaded_episodes <= 0
            or int(online_prefill_stats.get("inserted", 0)) <= 0
        ):
            raise RuntimeError(
                "training.online_prefill.enabled=true but no online prefill "
                "episodes were loaded into the learner replay buffer"
            )
        logger.info(
            "online prefill (learner-owned): episodes_loaded=%s/%s files_loaded=%s/%s "
            "inserted=%s success_episodes=%s",
            online_prefill_loaded_episodes,
            configured_warmup_episodes,
            online_prefill_stats.get("files_loaded", 0),
            online_prefill_stats.get("files_total", 0),
            online_prefill_stats.get("inserted", 0),
            online_prefill_stats.get("success_episodes", 0),
        )
    elif online_prefill_enabled:
        logger.info(
            "training.online_prefill.enabled=true but training.warmup.episodes=%s; "
            "skipping learner-owned online prefill load",
            configured_warmup_episodes,
        )

    step_logger = JsonlLogger(run_dir / str(cfg.logging.step_log_file))
    episode_logger = JsonlLogger(run_dir / str(cfg.logging.episode_log_file))
    profiling_logger: Optional[JsonlLogger] = None
    if profiling_enabled:
        profiling_logger = JsonlLogger(run_dir / profiling_log_file)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    profiler = _RuntimeProfiler(
        enabled=profiling_enabled,
        window_size=profiling_window_size,
    )
    stats_logger = _LearnerStatsLogger(
        logger=logger,
        step_logger=step_logger,
        episode_logger=episode_logger,
        tb_writer=tb_writer,
        replay_buffer=replay_buffer,
        offline_buffer=offline_buffer,
    )

    initial_payload = _snapshot_agent_checkpoint_payload(
        learner_agent,
        step=int(learner_agent.state.step),
    )
    critic_pretrain = _pretrain_critic_with_calql(
        cfg,
        agent=learner_agent,
        offline_buffer=offline_buffer,
        logger=logger,
        tb_writer=tb_writer,
    )
    if int(critic_pretrain.get("enabled", 0)) > 0:
        initial_payload = _snapshot_agent_checkpoint_payload(
            learner_agent,
            step=int(learner_agent.state.step),
        )

    logger.info(
        "Starting standalone agentlace learner at %s:%s (broadcast=%s) "
        "training_starts=%s online_capacity=%s offline_size=%s calql_enabled=%s",
        async_trainer_host,
        async_trainer_port,
        async_broadcast_port,
        int(cfg.training.training_starts),
        int(getattr(replay_buffer, "capacity", getattr(replay_buffer, "_capacity", 0))),
        int(len(offline_buffer)) if offline_buffer is not None else 0,
        int(critic_pretrain.get("enabled", 0)),
    )

    del learner_agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        run_agentlace_learner_service(
            cfg_dict=OmegaConf.to_container(cfg, resolve=True),
            sample_obs=sample_obs,
            action_dim=int(agent_action_dim),
            critic_action_dim=int(critic_action_dim),
            image_keys=tuple(image_keys),
            action_transform=action_transform,
            learner_device=async_learner_device,
            update_frequency=async_update_frequency,
            idle_sleep_sec=async_idle_sleep_sec,
            training_starts=int(cfg.training.training_starts),
            initial_payload=initial_payload,
            replay_buffer=replay_buffer,
            offline_buffer=offline_buffer,
            batch_size=int(cfg.replay.batch_size),
            offline_ratio=float(cfg.offline.ratio),
            symmetric_replay=bool(cfg.offline.get("symmetric_replay", False)),
            host=async_trainer_host,
            port_number=async_trainer_port,
            broadcast_port=async_broadcast_port,
            profiler=profiler,
            profiling_log_period_steps=profiling_log_period_steps,
            profiling_logger=profiling_logger,
            profiling_tb_writer=tb_writer,
            profiling_python_logger=logger,
            online_prefill_stats=online_prefill_stats,
            status_queue=None,
            command_queue=None,
            stats_request_callback=stats_logger.handle_payload,
        )
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping standalone learner")
    finally:
        if learner_env is not None:
            try:
                learner_env.close()
            except Exception:
                pass
        step_logger.close()
        episode_logger.close()
        if profiling_logger is not None:
            profiling_logger.close()
        tb_writer.close()
