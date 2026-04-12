from __future__ import annotations

import logging
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, IO, Any, Dict, Optional, Tuple

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

    sys.modules["gym"] = gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.policy.base import PolicyPrefetcher
from serl_launcher.policy.factory import build_policy_backend_info
from serl_launcher.policy.factory import build_policy_client
from serl_launcher.policy.factory import build_policy_prefetcher
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.data.training_loader import load_residual_training_buffer
from serl_launcher.residual.runtime_agent import create_residual_agent_runtime
from serl_launcher.residual.train.actor.support import ActorRuntimeContext
from serl_launcher.residual.train.actor.support import (
    epsilon_gating_clock as resolve_epsilon_gating_clock,
)
from serl_launcher.residual.train.actor.support import (
    epsilon_gating_enabled as resolve_epsilon_gating_enabled,
)
from serl_launcher.residual.train.actor.support import ensure_training_runtime_started
from serl_launcher.residual.train.actor.support import initialize_actor_loop_state
from serl_launcher.residual.train.actor.support import resolve_alpha_step
from serl_launcher.residual.train.actor.support import (
    scheduled_epsilon_gating_probability,
)
from serl_launcher.training.async_runtime.bridge import AgentlaceBridgeConfig
from serl_launcher.training.async_runtime.bridge import AgentlaceBridgeState
from serl_launcher.training.async_runtime.bridge import create_agentlace_async_learner
from serl_launcher.training.async_runtime.bridge import save_actor_bootstrap
from serl_launcher.residual.train.async_eval import _init_async_eval_tb_sync_state
from serl_launcher.residual.train.async_eval import _start_async_eval_watcher
from serl_launcher.training.async_runtime.agentlace import _AsyncLearner
from serl_launcher.training.async_runtime.agentlace import _MixedBatchPrefetcher
from serl_launcher.training.async_runtime.agentlace import _ProcessAsyncLearner
from serl_launcher.training.async_runtime.agentlace import _sample_mixed_batch
from serl_launcher.residual.train.bindings import ResidualRuntimeBindings
from serl_launcher.residual.train.config import build_residual_action_transform
from serl_launcher.residual.train.config import resolve_action_mask_from_cfg
from serl_launcher.residual.train.config import resolve_control_indices_from_cfg
from serl_launcher.training.checkpoint import AsyncCheckpointWriter
from serl_launcher.residual.train.obs_utils import _obs_space_from_sample
from serl_launcher.residual.train.pretrain import _pretrain_critic_with_calql
from serl_launcher.training.profiling import _RuntimeProfiler
from serl_launcher.training.profiling import _profile_call
from serl_launcher.residual.train.step_chunk_replay import ChunkReplayBuffer
from serl_launcher.residual.train.telemetry import _new_tb_step_window
from serl_launcher.training.jsonl import JsonlWriter
from serl_launcher.residual.utils.alpha_utils import require_residual_alpha
from serl_launcher.residual.utils.alpha_utils import validate_alpha

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def build_actor_runtime_session(
    cfg: DictConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
    bindings: ResidualRuntimeBindings,
    async_eval_watcher_path: Path,
):
    env = bindings.env
    normalizer = getattr(bindings, "normalizer", None)
    image_keys = tuple(bindings.image_keys)
    obs_cache = bindings.obs_cache
    task_key = str(bindings.task_key)
    data_config = bindings.data_config
    build_residual_step_obs_profiled = bindings.build_step_obs_profiled
    build_residual_step_core = bindings.build_step_core

    agent_runtime = create_residual_agent_runtime(cfg)
    logger.info("Residual runtime: %s", agent_runtime.name)

    policy_backend_info = build_policy_backend_info(cfg)
    policy_client = build_policy_client(cfg, logger=logger)
    logger.info(
        "Chunk policy backend: type=%s id=%s",
        policy_backend_info["type"],
        policy_backend_info["id"],
    )
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    env_action_dim_cfg = cfg.get("env", {}).get("action_dim", None)
    if env_action_dim_cfg is None:
        raise ValueError("env.action_dim must be set in yaml (e.g. env.action_dim: 7)")
    env_action_dim = int(env_action_dim_cfg)
    if env_action_dim <= 0:
        raise ValueError(f"env.action_dim must be positive, got {env_action_dim}")

    control_indices = resolve_control_indices_from_cfg(
        cfg, full_action_dim=env_action_dim
    )
    step_action_dim = int(len(control_indices))
    chunk_horizon = int(cfg.residual.chunk_horizon)
    residual_alpha = require_residual_alpha(cfg.get("residual", None))
    chunk_step_cfg = cfg.get("chunk_step", None)
    chunk_step_enabled = (
        bool(chunk_step_cfg.get("enabled", False))
        if chunk_step_cfg is not None
        else False
    )
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
    chunk_step_scheduler_clock = (
        str(chunk_step_cfg.get("scheduler_clock", "env_step")).lower()
        if chunk_step_cfg is not None
        else "env_step"
    )
    if chunk_step_scheduler_clock not in {"decision_step", "env_step"}:
        raise ValueError(
            "chunk_step.scheduler_clock must be 'decision_step' or 'env_step', "
            f"got {chunk_step_scheduler_clock}"
        )
    if chunk_step_enabled:
        logger.info(
            "chunk_step step-window replay is enabled; sample_stride is interpreted in env-step "
            "units over the step stream, matching RLT-style chunk subsampling."
        )
    agent_action_dim = (
        int(step_action_dim * chunk_horizon)
        if chunk_step_enabled
        else int(step_action_dim)
    )
    critic_action_dim = (
        int(env_action_dim * chunk_horizon)
        if chunk_step_enabled
        else int(env_action_dim)
    )
    residual_action_limits_cfg = cfg.residual.get("action_limits", None)
    residual_limits = build_residual_limits(
        control_indices,
        action_limits=residual_action_limits_cfg,
        full_action_dim=env_action_dim,
    )
    action_mask = resolve_action_mask_from_cfg(cfg, full_action_dim=env_action_dim)
    epsilon_gating_enabled = resolve_epsilon_gating_enabled(cfg)
    epsilon_gating_clock = resolve_epsilon_gating_clock(cfg)
    resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.info(
        "Residual config: image_keys=%s step_action_dim=%s agent_action_dim=%s "
        "action_mask=%s control_indices=%s env_action_dim=%s chunk_horizon=%s alpha=%.4f "
        "chunk_step_enabled=%s stride=%s limits=%s",
        list(image_keys),
        step_action_dim,
        agent_action_dim,
        action_mask.astype(bool).tolist(),
        control_indices.tolist(),
        env_action_dim,
        chunk_horizon,
        residual_alpha,
        chunk_step_enabled,
        chunk_step_sample_stride,
        residual_limits.tolist(),
    )
    action_transform = build_residual_action_transform(
        control_indices=control_indices,
        residual_limits=residual_limits,
        full_action_dim=env_action_dim,
        chunk_horizon=chunk_horizon,
        chunk_step_enabled=chunk_step_enabled,
        clip_gripper=bool(cfg.residual.clip_gripper),
    )
    if epsilon_gating_enabled:
        epsilon_gating_cfg = cfg.residual.get("epsilon_gating", {})
        logger.info(
            "epsilon-gating enabled: schedule=%s clock=%s min_prob=%.4f max_prob=%.4f "
            "warmup_steps=%s ramp_steps=%s",
            str(epsilon_gating_cfg.get("schedule", "linear")),
            epsilon_gating_clock,
            float(epsilon_gating_cfg.get("min_prob", 0.0)),
            float(epsilon_gating_cfg.get("max_prob", 1.0)),
            int(epsilon_gating_cfg.get("warmup_steps", 0)),
            int(epsilon_gating_cfg.get("ramp_steps", 1)),
        )

    def _resolve_train_gate(
        *,
        phase_train_flag: bool,
        alpha_value: float,
        env_step_value: int,
        decision_step_value: int,
    ) -> tuple[float, bool]:
        if (not phase_train_flag) or (not epsilon_gating_enabled):
            return 1.0, bool(alpha_value > 0.0)

        schedule_step = (
            int(env_step_value)
            if epsilon_gating_clock == "env_step"
            else int(decision_step_value)
        )
        gate_prob = scheduled_epsilon_gating_probability(
            cfg, schedule_step=schedule_step
        )
        if alpha_value <= 0.0:
            return float(gate_prob), False
        gate_on = bool(np.random.random() < float(gate_prob))
        return float(gate_prob), gate_on

    offline_enabled = bool(cfg.offline.enabled)
    offline_dataset_paths_cfg = cfg.offline.get("dataset_paths", None)
    has_offline_dataset_paths = (
        bool(offline_dataset_paths_cfg) and len(offline_dataset_paths_cfg) > 0
    )
    offline_ratio = float(cfg.offline.ratio)
    if not (0.0 <= offline_ratio <= 1.0):
        raise ValueError(f"offline.ratio must be in [0,1], got {offline_ratio}")
    symmetric_replay = bool(cfg.offline.get("symmetric_replay", False))
    async_cfg = cfg.training.get("async", None)
    async_enabled = (
        bool(async_cfg.get("enabled", False)) if async_cfg is not None else False
    )
    async_update_frequency = (
        int(async_cfg.get("update_frequency", 1)) if async_cfg is not None else 1
    )
    async_idle_sleep_sec = (
        float(async_cfg.get("idle_sleep_sec", 0.002))
        if async_cfg is not None
        else 0.002
    )
    async_backend = (
        str(async_cfg.get("backend", "thread")).strip().lower()
        if async_cfg is not None
        else "thread"
    )
    if async_backend not in {"thread", "process", "agentlace"}:
        raise ValueError(
            "training.async.backend must be 'thread', 'process', or 'agentlace', "
            f"got {async_backend}"
        )
    async_actor_device = (
        str(async_cfg.get("actor_device")).strip()
        if async_cfg is not None and async_cfg.get("actor_device", None) is not None
        else None
    )
    async_learner_device = (
        str(async_cfg.get("learner_device")).strip()
        if async_cfg is not None and async_cfg.get("learner_device", None) is not None
        else None
    )
    async_batch_queue_size = (
        int(async_cfg.get("batch_queue_size", 2)) if async_cfg is not None else 2
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
    async_data_store_queue_size = (
        int(async_cfg.get("data_store_queue_size", 2000))
        if async_cfg is not None
        else 2000
    )
    async_stats_period_steps = (
        int(async_cfg.get("stats_period_steps", 100)) if async_cfg is not None else 100
    )
    async_agentlace_cfg = (
        async_cfg.get("agentlace", None) if async_cfg is not None else None
    )
    async_agentlace_spawn_local_worker = (
        bool(async_agentlace_cfg.get("spawn_local_worker", True))
        if async_agentlace_cfg is not None
        else True
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
    async_bounded_lag_cfg = (
        async_cfg.get("bounded_lag", None) if async_cfg is not None else None
    )
    async_bounded_lag_cfg_enabled = (
        bool(async_bounded_lag_cfg.get("enabled", False))
        if async_bounded_lag_cfg is not None
        else False
    )
    async_bounded_lag_enabled = bool(async_enabled) and bool(
        async_bounded_lag_cfg_enabled
    )
    async_bounded_lag_max_update_calls = (
        int(async_bounded_lag_cfg.get("max_update_lag_calls", 2))
        if async_bounded_lag_cfg is not None
        else 2
    )
    async_bounded_lag_poll_sec = (
        async_bounded_lag_cfg.get("poll_sec", None)
        if async_bounded_lag_cfg is not None
        else None
    )
    if async_bounded_lag_poll_sec is not None:
        async_bounded_lag_poll_sec = float(async_bounded_lag_poll_sec)
    async_bounded_lag_timeout_sec = (
        async_bounded_lag_cfg.get("timeout_sec", 30.0)
        if async_bounded_lag_cfg is not None
        else 30.0
    )
    if async_bounded_lag_timeout_sec is not None:
        async_bounded_lag_timeout_sec = float(async_bounded_lag_timeout_sec)
    async_bounded_lag_sync_on_wait = (
        bool(async_bounded_lag_cfg.get("sync_on_wait", True))
        if async_bounded_lag_cfg is not None
        else True
    )
    async_bounded_lag_log_period_steps = (
        int(async_bounded_lag_cfg.get("log_period_steps", async_stats_period_steps))
        if async_bounded_lag_cfg is not None
        else int(async_stats_period_steps)
    )
    async_bounded_lag_env_steps_per_update_call = (
        async_bounded_lag_cfg.get("env_steps_per_update_call", None)
        if async_bounded_lag_cfg is not None
        else None
    )
    if async_bounded_lag_env_steps_per_update_call is not None:
        async_bounded_lag_env_steps_per_update_call = float(
            async_bounded_lag_env_steps_per_update_call
        )
        if not async_bounded_lag_env_steps_per_update_call.is_integer():
            raise ValueError(
                "training.async.bounded_lag.env_steps_per_update_call must be "
                "an integer env-step count when set, got "
                f"{async_bounded_lag_env_steps_per_update_call}"
            )
        async_bounded_lag_env_steps_per_update_call = int(
            async_bounded_lag_env_steps_per_update_call
        )
    async_bounded_lag_manual_rate_enabled = bool(
        async_bounded_lag_enabled
        and async_bounded_lag_env_steps_per_update_call is not None
    )
    async_bounded_lag_mode = (
        "manual_env_steps_per_update_call"
        if async_bounded_lag_manual_rate_enabled
        else "sync_trigger_budget"
    )
    if async_bounded_lag_max_update_calls < 0:
        raise ValueError(
            "training.async.bounded_lag.max_update_lag_calls must be >= 0, "
            f"got {async_bounded_lag_max_update_calls}"
        )
    if (
        async_bounded_lag_poll_sec is not None
        and float(async_bounded_lag_poll_sec) <= 0.0
    ):
        raise ValueError(
            "training.async.bounded_lag.poll_sec must be positive when set, "
            f"got {async_bounded_lag_poll_sec}"
        )
    if (
        async_bounded_lag_timeout_sec is not None
        and float(async_bounded_lag_timeout_sec) < 0.0
    ):
        raise ValueError(
            "training.async.bounded_lag.timeout_sec must be >= 0 when set, "
            f"got {async_bounded_lag_timeout_sec}"
        )
    if (
        async_bounded_lag_env_steps_per_update_call is not None
        and int(async_bounded_lag_env_steps_per_update_call) <= 0
    ):
        raise ValueError(
            "training.async.bounded_lag.env_steps_per_update_call must be "
            f"positive when set, got {async_bounded_lag_env_steps_per_update_call}"
        )
    if async_bounded_lag_env_steps_per_update_call is not None and (
        not async_bounded_lag_enabled
    ):
        logger.warning(
            "training.async.bounded_lag.env_steps_per_update_call=%s is set, but "
            "bounded lag is disabled (async.enabled=%s, bounded_lag.enabled=%s); "
            "manual env-step/update-call pacing is ignored",
            int(async_bounded_lag_env_steps_per_update_call),
            bool(async_enabled),
            bool(async_bounded_lag_cfg_enabled),
        )
    if (
        async_bounded_lag_manual_rate_enabled
        and int(cfg.training.updates_per_step) != 1
    ):
        raise ValueError(
            "training.async.bounded_lag.env_steps_per_update_call requires "
            f"training.updates_per_step=1, got {int(cfg.training.updates_per_step)}"
        )
    replay_prefetch_cfg = cfg.training.get("replay_prefetch", None)
    replay_prefetch_enabled = (
        bool(replay_prefetch_cfg.get("enabled", True))
        if replay_prefetch_cfg is not None
        else True
    )
    replay_prefetch_queue_size = (
        int(replay_prefetch_cfg.get("queue_size", 2))
        if replay_prefetch_cfg is not None
        else 2
    )
    replay_prefetch_pin_memory = (
        bool(replay_prefetch_cfg.get("pin_memory", True))
        if replay_prefetch_cfg is not None
        else True
    )
    replay_prefetch_to_device = (
        bool(replay_prefetch_cfg.get("to_device", True))
        if replay_prefetch_cfg is not None
        else True
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
    if async_enabled and any(
        (not bool(phase.get("train", True))) for phase in cfg.training.phases
    ):
        logger.warning(
            "Detected non-train phase in training.phases; disable async mode to preserve phase semantics."
        )
        async_enabled = False
    external_agentlace_actor_mode = bool(
        async_enabled
        and async_backend == "agentlace"
        and (not async_agentlace_spawn_local_worker)
    )
    manage_learner_state_locally = not external_agentlace_actor_mode
    agentlace_bridge_config = AgentlaceBridgeConfig(
        external_actor_mode=bool(external_agentlace_actor_mode),
        host=str(async_trainer_host),
        trainer_port=int(async_trainer_port),
        broadcast_port=int(async_broadcast_port),
        data_store_queue_size=int(async_data_store_queue_size),
        spawn_local_worker=bool(async_agentlace_spawn_local_worker),
        connect_timeout_sec=float(async_agentlace_connect_timeout_sec),
        batch_size=int(cfg.replay.batch_size),
        offline_ratio=float(offline_ratio),
        symmetric_replay=bool(symmetric_replay),
        training_starts=int(cfg.training.training_starts),
        update_every=int(cfg.training.update_every),
        updates_per_step=int(cfg.training.updates_per_step),
        update_frequency=int(async_update_frequency),
        idle_sleep_sec=float(async_idle_sleep_sec),
        learner_device=async_learner_device,
        stats_period_steps=int(async_stats_period_steps),
        bounded_lag_enabled=bool(async_bounded_lag_enabled),
        bounded_lag_mode=str(async_bounded_lag_mode),
        bounded_lag_max_update_calls=int(async_bounded_lag_max_update_calls),
        bounded_lag_poll_sec=async_bounded_lag_poll_sec,
        bounded_lag_timeout_sec=async_bounded_lag_timeout_sec,
        bounded_lag_sync_on_wait=bool(async_bounded_lag_sync_on_wait),
        bounded_lag_log_period_steps=int(async_bounded_lag_log_period_steps),
        bounded_lag_env_steps_per_update_call=async_bounded_lag_env_steps_per_update_call,
        bounded_lag_manual_rate_enabled=bool(async_bounded_lag_manual_rate_enabled),
    )
    agentlace_bridge_state = AgentlaceBridgeState()
    if (
        offline_enabled
        and manage_learner_state_locally
        and (not has_offline_dataset_paths)
    ):
        raise ValueError(
            "offline.enabled=true requires offline.dataset_paths to be set "
            "because offline bootstrap has been removed"
        )
    logger.info(
        "Async collection-learning: enabled=%s backend=%s update_frequency=%s "
        "idle_sleep_sec=%.4f actor_device=%s learner_device=%s batch_queue_size=%s "
        "trainer_host=%s trainer_port=%s broadcast_port=%s data_store_queue_size=%s "
        "stats_period_steps=%s "
        "agentlace_spawn_local_worker=%s agentlace_bootstrap_file=%s "
        "agentlace_connect_timeout_sec=%.1f",
        async_enabled,
        async_backend,
        async_update_frequency,
        async_idle_sleep_sec,
        async_actor_device,
        async_learner_device,
        async_batch_queue_size,
        async_trainer_host,
        async_trainer_port,
        async_broadcast_port,
        async_data_store_queue_size,
        async_stats_period_steps,
        async_agentlace_spawn_local_worker,
        async_agentlace_bootstrap_file,
        async_agentlace_connect_timeout_sec,
    )
    logger.info(
        "Async bounded-lag: enabled=%s mode=%s max_update_lag_calls=%s "
        "env_steps_per_update_call=%s poll_sec=%s timeout_sec=%s sync_on_wait=%s",
        async_bounded_lag_enabled,
        async_bounded_lag_mode,
        async_bounded_lag_max_update_calls,
        (
            None
            if async_bounded_lag_env_steps_per_update_call is None
            else float(async_bounded_lag_env_steps_per_update_call)
        ),
        (
            None
            if async_bounded_lag_poll_sec is None
            else float(async_bounded_lag_poll_sec)
        ),
        (
            None
            if async_bounded_lag_timeout_sec is None
            else float(async_bounded_lag_timeout_sec)
        ),
        async_bounded_lag_sync_on_wait,
    )
    logger.info(
        "Replay batch prefetch: enabled=%s queue_size=%s pin_memory=%s to_device=%s",
        replay_prefetch_enabled,
        replay_prefetch_queue_size,
        replay_prefetch_pin_memory,
        replay_prefetch_to_device,
    )
    logger.info(
        "Profiling: enabled=%s window_size=%s log_period_steps=%s log_file=%s",
        profiling_enabled,
        profiling_window_size,
        profiling_log_period_steps,
        profiling_log_file,
    )

    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(critic_action_dim,), dtype=np.float32
    )
    agent = None
    learner_agent = None
    async_learner: Optional[Any] = None
    sync_replay_lock: Optional[threading.Lock] = None
    sync_replay_prefetcher: Optional[_MixedBatchPrefetcher] = None
    checkpoint_writer: Optional[AsyncCheckpointWriter] = None
    policy_prefetcher: Optional[PolicyPrefetcher] = None
    replay_buffer = None
    offline_buffer = None
    offline_stats: Dict[str, Any] = {
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "clipped_values": 0,
        "errors": 0,
    }
    warmstart_info: Dict[str, Any] = {"enabled": 0, "steps": 0}
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

    def _flush_external_agentlace_actor() -> None:
        if external_agentlace_actor_mode and async_learner is not None:
            async_learner.flush()

    checkpoint_cfg = cfg.training.get("checkpoint", None)
    if checkpoint_cfg is None:
        raise ValueError("training.checkpoint must be configured")
    checkpoint_every_steps = int(checkpoint_cfg.get("every_steps", 0))
    checkpoint_keep = int(checkpoint_cfg.get("keep", 0))
    checkpoint_dir = Path(str(checkpoint_cfg.get("dir", "checkpoints")))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async_eval_cfg = cfg.training.get("async_eval", None)
    async_eval_enabled = (
        bool(async_eval_cfg.get("enabled", False))
        if async_eval_cfg is not None
        else False
    )
    async_eval_every_episodes = (
        int(async_eval_cfg.get("every_episodes", 0))
        if async_eval_cfg is not None
        else 0
    )
    if async_eval_enabled and async_eval_every_episodes <= 0:
        raise ValueError(
            "training.async_eval.enabled=true requires training.async_eval.every_episodes > 0"
        )
    async_eval_alpha_mode = (
        str(
            async_eval_cfg.get(
                "alpha_mode",
                "checkpoint_schedule",
            )
        )
        .strip()
        .lower()
        if async_eval_cfg is not None
        else "checkpoint_schedule"
    )
    valid_async_eval_alpha_modes = {"checkpoint_schedule", "base", "fixed"}
    if async_eval_enabled and async_eval_alpha_mode not in valid_async_eval_alpha_modes:
        raise ValueError(
            "training.async_eval.alpha_mode must be one of "
            f"{sorted(valid_async_eval_alpha_modes)}, got {async_eval_alpha_mode!r}"
        )
    if async_eval_enabled and async_eval_alpha_mode == "fixed":
        fixed_alpha_cfg = async_eval_cfg.get("fixed_alpha", None)
        if fixed_alpha_cfg is None:
            raise ValueError(
                "training.async_eval.alpha_mode=fixed requires "
                "training.async_eval.fixed_alpha to be set"
            )
        validate_alpha(
            fixed_alpha_cfg,
            name="training.async_eval.fixed_alpha",
            allow_zero=True,
        )
    # Async eval reconstructs alpha from checkpoint env-step. Do not allow decision-step clock here.
    if (
        async_eval_enabled
        and async_eval_alpha_mode == "checkpoint_schedule"
        and chunk_step_scheduler_clock != "env_step"
    ):
        raise ValueError(
            "training.async_eval.alpha_mode=checkpoint_schedule requires "
            "chunk_step.scheduler_clock=env_step"
        )

    async_eval_proc: Optional[subprocess.Popen] = None
    async_eval_log_fp: Optional[IO[str]] = None
    async_eval_log_path: Optional[Path] = None
    async_eval_summary_path: Optional[Path] = None
    async_eval_watcher_return_code: Optional[int] = None
    async_eval_dead_reported = False
    profiler = _RuntimeProfiler(
        enabled=(profiling_enabled or external_agentlace_actor_mode),
        window_size=profiling_window_size,
    )
    checkpoint_writer = AsyncCheckpointWriter(profiler=profiler)
    profiling_logger: Optional[JsonlWriter] = None
    profiling_last_flush_step = -1
    async_eval_queue_path: Optional[Path] = None
    (
        async_eval_proc,
        async_eval_log_fp,
        async_eval_log_path,
        async_eval_summary_path,
        async_eval_queue_path,
    ) = _start_async_eval_watcher(
        watcher_path=async_eval_watcher_path,
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
    )

    step_logger = JsonlWriter(run_dir / str(cfg.logging.step_log_file))
    episode_logger = JsonlWriter(run_dir / str(cfg.logging.episode_log_file))
    policy_prefetcher = build_policy_prefetcher(cfg, logger=logger)
    if profiling_enabled:
        profiling_logger = JsonlWriter(run_dir / profiling_log_file)
    # Import TensorBoard lazily so real-robot env setup can initialize the
    # AgiBot DDS stack before TensorFlow/tensorboard side effects occur.
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    tb_step_period = int(cfg.logging.get("tb_step_period", 100))
    tb_histogram_period = max(
        tb_step_period,
        int(cfg.logging.get("tb_histogram_period", max(tb_step_period * 10, 1))),
    )
    progress_enabled = bool(cfg.logging.get("progress_bar", True))
    progress_mininterval_sec = float(cfg.logging.get("progress_mininterval_sec", 1.0))
    step_metric_window = _new_tb_step_window()
    async_eval_tb_sync_state = _init_async_eval_tb_sync_state(async_eval_summary_path)

    def _policy_input(
        obs_raw: Dict[str, Any],
        prompt: str,
        *,
        cache_key: Optional[Any] = None,
    ):
        return bindings.build_policy_input(obs_raw, prompt, cache_key=cache_key)

    def _maybe_send_agentlace_timer_stats(
        *,
        train_env_step_value: int,
        decision_step_value: int,
        train_episode_id_value: int,
        force: bool = False,
    ) -> None:
        maybe_send_agentlace_timer_stats(
            config=agentlace_bridge_config,
            state=agentlace_bridge_state,
            profiler=profiler,
            replay_buffer=replay_buffer,
            offline_buffer=offline_buffer,
            async_learner=async_learner,
            sync_replay_prefetcher=sync_replay_prefetcher,
            train_env_step=int(train_env_step_value),
            decision_step=int(decision_step_value),
            train_episode_id=int(train_episode_id_value),
            force=bool(force),
        )

    def _advance_async_target_update_calls(
        *,
        phase_train_flag: bool,
        train_step_before: int,
        train_step_after: int,
        replay_size_before: int,
        replay_size_after: int,
    ) -> int:
        return advance_async_target_update_calls(
            config=agentlace_bridge_config,
            state=agentlace_bridge_state,
            async_learner=async_learner,
            phase_train_flag=bool(phase_train_flag),
            train_step_before=int(train_step_before),
            train_step_after=int(train_step_after),
            replay_size_before=int(replay_size_before),
            replay_size_after=int(replay_size_after),
        )

    def _maybe_wait_for_async_learner_budget(
        *,
        train_env_step_value: int,
        decision_step_value: int,
    ) -> None:
        maybe_wait_for_async_learner_budget(
            config=agentlace_bridge_config,
            state=agentlace_bridge_state,
            async_learner=async_learner,
            logger=logger,
            train_env_step=int(train_env_step_value),
            decision_step=int(decision_step_value),
        )

    def _sync_async_bounded_lag_baseline_from_learner() -> None:
        sync_async_bounded_lag_baseline_from_learner(
            config=agentlace_bridge_config,
            state=agentlace_bridge_state,
            async_learner=async_learner,
            logger=logger,
        )

    task_cfg = cfg.get("task", {})
    sample_reset_kwargs = {"init_episode_idx": -1}
    if task_cfg is not None and task_cfg.get("seed_base", None) is not None:
        sample_reset_kwargs["seed"] = int(task_cfg.get("seed_base"))
    sample_obs_raw = _profile_call(
        profiler,
        "env_reset",
        env.reset,
        **sample_reset_kwargs,
    )
    sample_policy_chunk, _ = policy_client.infer(
        _policy_input(sample_obs_raw, env.current_instruction)
    )
    sample_base_chunk = select_action_chunk_window(
        sample_policy_chunk,
        horizon=chunk_horizon,
        action_dim=env_action_dim,
    )
    sample_obs = build_residual_step_obs_profiled(
        profiler,
        sample_obs_raw,
        sample_base_chunk[0],
        image_keys=image_keys,
        stack_horizon=stack_horizon,
        normalizer=normalizer,
        obs_cache=obs_cache,
        base_action_chunk=(sample_base_chunk if chunk_step_enabled else None),
        alpha=float(residual_alpha),
    )
    sample_state_core = build_residual_step_core(
        sample_obs_raw,
        image_keys=image_keys,
        normalizer=normalizer,
        obs_cache=obs_cache,
    )["state_core"]

    def _normalize_step_action(action: np.ndarray) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if normalizer is None:
            return action_arr.astype(np.float32)
        return np.asarray(normalizer.normalize_action(action_arr), dtype=np.float32)

    def _build_chunk_step_record(
        current_obs_raw: Dict[str, Any],
        *,
        base_action: np.ndarray,
        final_action: np.ndarray,
        alpha_obs: float,
        episode_id: int,
        episode_step: int,
        done: bool,
    ) -> Dict[str, Any]:
        obs_core = build_residual_step_core(
            current_obs_raw,
            image_keys=image_keys,
            normalizer=normalizer,
            obs_cache=obs_cache,
        )
        base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
        final_action_arr = np.asarray(final_action, dtype=np.float32).reshape(-1)
        return {
            "obs_core": obs_core,
            "base_action": base_action_arr,
            "base_action_norm": _normalize_step_action(base_action_arr),
            "actions": final_action_arr,
            "rewards": 0.0,
            "dones": bool(done),
            "alpha": float(alpha_obs),
            "episode_id": int(episode_id),
            "episode_step": int(episode_step),
        }

    def _replay_progress_size(buffer: Any) -> int:
        return int(getattr(buffer, "num_steps", len(buffer)))

    learner_agent_device = (
        async_learner_device if async_enabled and manage_learner_state_locally else None
    )
    actor_agent_device = (
        async_actor_device
        if async_enabled and async_backend in {"process", "agentlace"}
        else None
    )
    learner_agent = agent_runtime.create_learner_agent(
        cfg,
        sample_obs=sample_obs,
        action_dim=agent_action_dim,
        image_keys=image_keys,
        critic_action_dim=critic_action_dim,
        action_transform=action_transform,
        device=learner_agent_device,
    )
    if async_enabled and async_backend in {"process", "agentlace"}:
        agent = agent_runtime.create_actor_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=agent_action_dim,
            image_keys=image_keys,
            critic_action_dim=critic_action_dim,
            action_transform=action_transform,
            device=actor_agent_device,
        )
        agent_runtime.sync_modules(agent, learner_agent)
        if not manage_learner_state_locally:
            learner_agent = agent
    else:
        agent = learner_agent
    if chunk_step_enabled:
        replay_buffer = ChunkReplayBuffer(
            sample_observation_template=sample_obs,
            state_core_dim=int(sample_state_core.shape[0]),
            step_action_dim=env_action_dim,
            chunk_horizon=chunk_horizon,
            discount=float(cfg.sac.discount),
            capacity=int(cfg.replay.capacity),
            sample_stride=chunk_step_sample_stride,
            require_full_horizon=chunk_step_require_full_horizon,
            pad_action_to_horizon=chunk_step_pad_action,
        )
    else:
        replay_buffer = ReplayBuffer(
            observation_space=_obs_space_from_sample(sample_obs),
            action_space=action_space,
            capacity=int(cfg.replay.capacity),
        )
    if offline_enabled and manage_learner_state_locally:
        if chunk_step_enabled:
            offline_buffer = ChunkReplayBuffer(
                sample_observation_template=sample_obs,
                state_core_dim=int(sample_state_core.shape[0]),
                step_action_dim=env_action_dim,
                chunk_horizon=chunk_horizon,
                discount=float(cfg.sac.discount),
                capacity=int(cfg.offline.capacity),
                sample_stride=chunk_step_sample_stride,
                require_full_horizon=chunk_step_require_full_horizon,
                pad_action_to_horizon=chunk_step_pad_action,
            )
        else:
            offline_buffer = ReplayBuffer(
                observation_space=_obs_space_from_sample(sample_obs),
                action_space=action_space,
                capacity=int(cfg.offline.capacity),
            )
        offline_residual_alpha = resolve_alpha_step(
            cfg, base_alpha=residual_alpha, schedule_step=0
        )
        offline_stats = load_residual_training_buffer(
            cfg.offline.dataset_paths,
            sample_obs_template=sample_obs,
            replay_buffer=offline_buffer,
            action_dim=env_action_dim,
            chunk_horizon=chunk_horizon,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            chunk_step_enabled=chunk_step_enabled,
            logger=logger,
            data_config=data_config,
            normalizer=normalizer,
            profiler=profiler,
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
            "offline preload: buffer=%s files_loaded=%s/%s candidates=%s inserted=%s skipped=%s clipped=%s errors=%s",
            len(offline_buffer),
            offline_stats.get("files_loaded", 0),
            offline_stats.get("files_total", 0),
            offline_stats.get("candidates", 0),
            offline_stats.get("inserted", 0),
            offline_stats.get("skipped", 0),
            offline_stats.get("clipped_values", 0),
            offline_stats.get("errors", 0),
        )
        warmstart_info = _pretrain_critic_with_calql(
            cfg,
            agent=learner_agent,
            offline_buffer=offline_buffer,
            logger=logger,
            tb_writer=tb_writer,
        )

    async_agentlace_bootstrap_path: Optional[Path] = None
    if async_enabled and async_backend == "agentlace":
        async_agentlace_bootstrap_path = save_actor_bootstrap(
            run_dir=run_dir,
            bootstrap_file=async_agentlace_bootstrap_file,
            sample_obs=sample_obs,
            state_core_dim=int(sample_state_core.shape[0]),
            env_action_dim=int(env_action_dim),
            step_action_dim=int(step_action_dim),
            agent_action_dim=int(agent_action_dim),
            critic_action_dim=int(critic_action_dim),
            image_keys=tuple(image_keys),
            action_transform=action_transform,
            chunk_step_enabled=bool(chunk_step_enabled),
            chunk_horizon=int(chunk_horizon),
            learner_agent=learner_agent,
            logger=logger,
        )

    if external_agentlace_actor_mode:
        if offline_enabled:
            logger.info(
                "External agentlace actor mode: offline replay/pretrain will be owned by the external learner process."
            )
        async_learner = create_agentlace_async_learner(
            config=replace(agentlace_bridge_config, spawn_local_worker=False),
            algorithm=agent_runtime,
            actor_agent=agent,
            replay_buffer=replay_buffer,
            offline_buffer=None,
            cfg_dict=resolved_cfg_dict,
            sample_obs=sample_obs,
            action_dim=agent_action_dim,
            critic_action_dim=critic_action_dim,
            image_keys=image_keys,
            action_transform=action_transform,
        )
        replay_buffer = async_learner.replay_proxy
        logger.info(
            "Connected actor to external agentlace learner at %s:%s",
            async_trainer_host,
            async_trainer_port,
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
    online_prefill_loaded_episodes = 0
    if (
        external_agentlace_actor_mode
        and async_learner is not None
        and online_prefill_enabled
        and configured_warmup_episodes > 0
    ):
        online_prefill_stats = async_learner.get_online_prefill_stats()
        online_prefill_loaded_episodes = int(
            online_prefill_stats.get("episodes_loaded", 0)
        )
        if (
            online_prefill_loaded_episodes <= 0
            or int(online_prefill_stats.get("inserted", 0)) <= 0
        ):
            raise RuntimeError(
                "training.online_prefill.enabled=true in external agentlace actor mode, "
                "but the learner reported no online prefill episodes loaded into replay"
            )
        logger.info(
            "External agentlace actor mode: learner-owned online prefill "
            "episodes_loaded=%s/%s files_loaded=%s/%s inserted=%s success_episodes=%s",
            online_prefill_loaded_episodes,
            configured_warmup_episodes,
            online_prefill_stats.get("files_loaded", 0),
            online_prefill_stats.get("files_total", 0),
            online_prefill_stats.get("inserted", 0),
            online_prefill_stats.get("success_episodes", 0),
        )
    elif online_prefill_enabled and configured_warmup_episodes > 0:
        online_prefill_dataset_paths = online_prefill_cfg.get("dataset_paths", None)
        if not online_prefill_dataset_paths:
            raise ValueError(
                "training.online_prefill.enabled=true requires "
                "training.online_prefill.dataset_paths to be set when "
                "training.warmup.episodes > 0"
            )
        online_prefill_stats = load_residual_training_buffer(
            online_prefill_dataset_paths,
            replay_buffer=replay_buffer,
            sample_obs_template=sample_obs,
            action_dim=env_action_dim,
            chunk_horizon=chunk_horizon,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            chunk_step_enabled=chunk_step_enabled,
            logger=logger,
            data_config=data_config,
            normalizer=normalizer,
            profiler=profiler,
            max_episodes=configured_warmup_episodes,
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
                "episodes were loaded into the replay buffer"
            )
        logger.info(
            "online prefill: episodes_loaded=%s/%s files_loaded=%s/%s inserted=%s "
            "success_episodes=%s",
            online_prefill_loaded_episodes,
            configured_warmup_episodes,
            online_prefill_stats.get("files_loaded", 0),
            online_prefill_stats.get("files_total", 0),
            online_prefill_stats.get("inserted", 0),
            online_prefill_stats.get("success_episodes", 0),
        )
        _flush_external_agentlace_actor()
    elif online_prefill_enabled:
        logger.info(
            "training.online_prefill.enabled=true but training.warmup.episodes=%s; "
            "skipping online prefill load",
            configured_warmup_episodes,
        )

    warmup_episodes_cfg = max(
        0,
        int(configured_warmup_episodes) - int(online_prefill_loaded_episodes),
    )
    if online_prefill_loaded_episodes > 0 and warmup_episodes_cfg > 0:
        logger.info(
            "online prefill covered %s/%s warmup episodes; runtime warmup will "
            "collect the remaining %s episodes",
            online_prefill_loaded_episodes,
            configured_warmup_episodes,
            warmup_episodes_cfg,
        )
    need_warmup_first = warmup_episodes_cfg > 0

    if async_learner is None and not need_warmup_first and async_enabled:
        if async_backend == "agentlace":
            agentlace_replay_buffer = replay_buffer
            async_learner = create_agentlace_async_learner(
                config=agentlace_bridge_config,
                algorithm=agent_runtime,
                actor_agent=agent,
                replay_buffer=agentlace_replay_buffer,
                offline_buffer=(
                    offline_buffer
                    if offline_enabled and manage_learner_state_locally
                    else None
                ),
                cfg_dict=resolved_cfg_dict,
                sample_obs=sample_obs,
                action_dim=agent_action_dim,
                critic_action_dim=critic_action_dim,
                image_keys=image_keys,
                action_transform=action_transform,
            )
            replay_buffer = async_learner.replay_proxy
        elif async_backend == "process":
            async_learner = _ProcessAsyncLearner(
                algorithm=agent_runtime,
                actor_agent=agent,
                online_buffer=replay_buffer,
                offline_buffer=offline_buffer if offline_enabled else None,
                batch_size=int(cfg.replay.batch_size),
                offline_ratio=offline_ratio,
                symmetric_replay=symmetric_replay,
                training_starts=int(cfg.training.training_starts),
                update_frequency=async_update_frequency,
                idle_sleep_sec=async_idle_sleep_sec,
                cfg_dict=resolved_cfg_dict,
                sample_obs=sample_obs,
                action_dim=agent_action_dim,
                critic_action_dim=critic_action_dim,
                image_keys=image_keys,
                action_transform=action_transform,
                actor_device=async_actor_device,
                learner_device=async_learner_device,
                batch_queue_size=async_batch_queue_size,
            )
            async_learner.start()
        else:
            agent = agent_runtime.create_actor_agent(
                cfg,
                sample_obs=sample_obs,
                action_dim=agent_action_dim,
                image_keys=image_keys,
                critic_action_dim=critic_action_dim,
                action_transform=action_transform,
                device=async_actor_device,
            )
            agent_runtime.sync_modules(agent, learner_agent)
            async_learner = _AsyncLearner(
                algorithm=agent_runtime,
                learner_agent=learner_agent,
                actor_agent=agent,
                online_buffer=replay_buffer,
                offline_buffer=offline_buffer if offline_enabled else None,
                batch_size=int(cfg.replay.batch_size),
                offline_ratio=offline_ratio,
                symmetric_replay=symmetric_replay,
                training_starts=int(cfg.training.training_starts),
                utd_ratio=int(cfg.sac.utd_ratio),
                update_frequency=async_update_frequency,
                idle_sleep_sec=async_idle_sleep_sec,
                replay_prefetch_enabled=replay_prefetch_enabled,
                replay_prefetch_queue_size=replay_prefetch_queue_size,
                replay_prefetch_pin_memory=replay_prefetch_pin_memory,
                replay_prefetch_to_device=replay_prefetch_to_device,
                checkpoint_writer=checkpoint_writer,
                profiler=profiler,
            )
            async_learner.start()
    elif not need_warmup_first and replay_prefetch_enabled:
        sync_replay_lock = threading.Lock()

        def _sample_sync_prefetch_batch() -> Optional[Tuple[Dict[str, Any], int, int]]:
            assert replay_buffer is not None
            assert sync_replay_lock is not None
            with sync_replay_lock:
                if _replay_progress_size(replay_buffer) < int(
                    cfg.training.training_starts
                ):
                    return None
                return _sample_mixed_batch(
                    replay_buffer,
                    offline_buffer if offline_enabled else None,
                    batch_size=int(cfg.replay.batch_size),
                    offline_ratio=offline_ratio,
                    symmetric_replay=symmetric_replay,
                )

        sync_replay_prefetcher = _MixedBatchPrefetcher(
            sample_fn=_sample_sync_prefetch_batch,
            queue_size=replay_prefetch_queue_size,
            idle_sleep_sec=async_idle_sleep_sec,
            device=learner_agent.device,
            pin_memory=replay_prefetch_pin_memory,
            to_device=replay_prefetch_to_device,
            profiler=profiler,
        )
        sync_replay_prefetcher.start()
    logger.info("Initialized DrQ agent, replay buffer, and offline pipeline")
    max_train_env_steps = int(cfg.training.get("max_train_env_steps", 0))
    ctx = ActorRuntimeContext(
        cfg=cfg,
        run_dir=run_dir,
        logger=logger,
        bindings=bindings,
        async_eval_watcher_path=async_eval_watcher_path,
    )
    ctx.update(
        algorithm=agent_runtime,
        env=env,
        policy_backend_info=policy_backend_info,
        policy_client=policy_client,
        policy_prefetcher=policy_prefetcher,
        stack_horizon=stack_horizon,
        env_action_dim=env_action_dim,
        control_indices=control_indices,
        step_action_dim=step_action_dim,
        chunk_horizon=chunk_horizon,
        residual_alpha=residual_alpha,
        chunk_step_enabled=chunk_step_enabled,
        chunk_step_sample_stride=chunk_step_sample_stride,
        chunk_step_require_full_horizon=chunk_step_require_full_horizon,
        chunk_step_pad_action=chunk_step_pad_action,
        chunk_step_scheduler_clock=chunk_step_scheduler_clock,
        agent_action_dim=agent_action_dim,
        critic_action_dim=critic_action_dim,
        residual_limits=residual_limits,
        action_mask=action_mask,
        epsilon_gating_enabled=epsilon_gating_enabled,
        epsilon_gating_clock=epsilon_gating_clock,
        resolved_cfg_dict=resolved_cfg_dict,
        offline_enabled=offline_enabled,
        offline_ratio=offline_ratio,
        symmetric_replay=symmetric_replay,
        async_enabled=async_enabled,
        async_update_frequency=async_update_frequency,
        async_idle_sleep_sec=async_idle_sleep_sec,
        async_backend=async_backend,
        async_actor_device=async_actor_device,
        async_learner_device=async_learner_device,
        async_batch_queue_size=async_batch_queue_size,
        async_trainer_host=async_trainer_host,
        async_trainer_port=async_trainer_port,
        async_broadcast_port=async_broadcast_port,
        async_data_store_queue_size=async_data_store_queue_size,
        async_stats_period_steps=async_stats_period_steps,
        async_agentlace_spawn_local_worker=async_agentlace_spawn_local_worker,
        async_agentlace_bootstrap_file=async_agentlace_bootstrap_file,
        async_agentlace_connect_timeout_sec=async_agentlace_connect_timeout_sec,
        async_bounded_lag_enabled=async_bounded_lag_enabled,
        async_bounded_lag_max_update_calls=async_bounded_lag_max_update_calls,
        async_bounded_lag_poll_sec=async_bounded_lag_poll_sec,
        async_bounded_lag_timeout_sec=async_bounded_lag_timeout_sec,
        async_bounded_lag_sync_on_wait=async_bounded_lag_sync_on_wait,
        async_bounded_lag_log_period_steps=async_bounded_lag_log_period_steps,
        async_bounded_lag_env_steps_per_update_call=async_bounded_lag_env_steps_per_update_call,
        async_bounded_lag_manual_rate_enabled=async_bounded_lag_manual_rate_enabled,
        async_bounded_lag_mode=async_bounded_lag_mode,
        replay_prefetch_enabled=replay_prefetch_enabled,
        replay_prefetch_queue_size=replay_prefetch_queue_size,
        replay_prefetch_pin_memory=replay_prefetch_pin_memory,
        replay_prefetch_to_device=replay_prefetch_to_device,
        profiling_enabled=profiling_enabled,
        profiling_window_size=profiling_window_size,
        profiling_log_period_steps=profiling_log_period_steps,
        profiling_log_file=profiling_log_file,
        external_agentlace_actor_mode=external_agentlace_actor_mode,
        manage_learner_state_locally=manage_learner_state_locally,
        agentlace_bridge_config=agentlace_bridge_config,
        agentlace_bridge_state=agentlace_bridge_state,
        action_transform=action_transform,
        action_space=action_space,
        agent=agent,
        learner_agent=learner_agent,
        async_learner=async_learner,
        sync_replay_lock=sync_replay_lock,
        sync_replay_prefetcher=sync_replay_prefetcher,
        checkpoint_writer=checkpoint_writer,
        replay_buffer=replay_buffer,
        offline_buffer=offline_buffer,
        offline_stats=offline_stats,
        warmstart_info=warmstart_info,
        online_prefill_stats=online_prefill_stats,
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_keep=checkpoint_keep,
        checkpoint_dir=checkpoint_dir,
        async_eval_enabled=async_eval_enabled,
        async_eval_every_episodes=async_eval_every_episodes,
        async_eval_alpha_mode=async_eval_alpha_mode,
        async_eval_proc=async_eval_proc,
        async_eval_log_fp=async_eval_log_fp,
        async_eval_log_path=async_eval_log_path,
        async_eval_summary_path=async_eval_summary_path,
        async_eval_watcher_return_code=async_eval_watcher_return_code,
        async_eval_dead_reported=async_eval_dead_reported,
        profiler=profiler,
        profiling_logger=profiling_logger,
        profiling_last_flush_step=profiling_last_flush_step,
        async_eval_queue_path=async_eval_queue_path,
        step_logger=step_logger,
        episode_logger=episode_logger,
        tb_writer=tb_writer,
        tb_step_period=tb_step_period,
        tb_histogram_period=tb_histogram_period,
        progress_enabled=progress_enabled,
        progress_mininterval_sec=progress_mininterval_sec,
        step_metric_window=step_metric_window,
        async_eval_tb_sync_state=async_eval_tb_sync_state,
        sample_obs=sample_obs,
        sample_state_core=sample_state_core,
        configured_warmup_episodes=configured_warmup_episodes,
        online_prefill_enabled=online_prefill_enabled,
        online_prefill_loaded_episodes=online_prefill_loaded_episodes,
        warmup_episodes_cfg=warmup_episodes_cfg,
        need_warmup_first=need_warmup_first,
        max_train_env_steps=max_train_env_steps,
    )
    state = initialize_actor_loop_state(ctx)
    if async_learner is None and not need_warmup_first:
        ensure_training_runtime_started(ctx)
        agent = ctx.agent
        async_learner = ctx.async_learner
        replay_buffer = ctx.replay_buffer
        sync_replay_lock = ctx.sync_replay_lock
        sync_replay_prefetcher = ctx.sync_replay_prefetcher

    train_env_step = int(state.train_env_step)
    decision_step = int(state.decision_step)
    train_episode_id = int(state.train_episode_id)
    warmup_episode_id = int(state.warmup_episode_id)
    init_episode_idx = int(state.init_episode_idx)
    eval_trigger_count = int(state.eval_trigger_count)
    train_total_success = int(state.train_total_success)
    train_recent_successes = state.train_recent_successes
    warmup_total_success = int(state.warmup_total_success)
    warmup_recent_successes = state.warmup_recent_successes
    skipped_seeds = int(state.skipped_seeds)
    seed_cursor = int(state.seed_cursor)
    stopped_by_env_budget = bool(state.stopped_by_env_budget)
    last_update_info = state.last_update_info
    saved_checkpoint_steps = state.saved_checkpoint_steps
    train_progress = state.train_progress
    warmup_progress = state.warmup_progress
    phase_progress = state.phase_progress
    train_progress_last_step = int(state.train_progress_last_step)

    return ctx, state
