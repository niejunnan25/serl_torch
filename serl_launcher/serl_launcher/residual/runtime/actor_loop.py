from __future__ import annotations

"""
LIBERO residual policy training (OpenPI + DrQ-SAC).

Minimal residual RL loop:
1. OpenPI predicts a base action chunk.
2. Residual policy predicts a residual action (step mode) or residual chunk (chunk mode).
3. Final action = base action + bounded residual.
4. Step mode writes step transitions; chunk mode writes a step stream that replay assembles into chunk windows.
5. DrQ-SAC updates from online replay or mixed online/offline replay.

Recommended runtime split:
- OpenPI server + LIBERO env server: run in the `libero` conda env.
- This trainer: run in a serl_torch / newer PyTorch env.
"""

import json
import logging
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import IO, Any, Dict, Optional, Tuple

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

    # Keep legacy `gym.*` imports working when only Gymnasium is installed.
    sys.modules["gym"] = gym
import numpy as np
import torch
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
from serl_launcher.policy.openpi.client import OpenPIPolicyClient
from serl_launcher.policy.openpi.prefetch import AsyncOpenPIPolicyPrefetcher
from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.action import as_numpy_action_chunk
from serl_launcher.residual.action import compose_residual_action
from serl_launcher.residual.action import compose_residual_action_chunk
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.data.training_loader import load_residual_training_buffer
from serl_launcher.residual.runtime.async_eval import _append_async_eval_request
from serl_launcher.residual.runtime.async_eval import _init_async_eval_tb_sync_state
from serl_launcher.residual.runtime.async_eval import _start_async_eval_watcher
from serl_launcher.residual.runtime.async_eval import _stop_async_eval_watcher
from serl_launcher.residual.runtime.async_eval import _sync_async_eval_results_to_tb
from serl_launcher.residual.runtime.agentlace_bridge import advance_async_target_update_calls
from serl_launcher.residual.runtime.agentlace_bridge import AgentlaceBridgeConfig
from serl_launcher.residual.runtime.agentlace_bridge import AgentlaceBridgeState
from serl_launcher.residual.runtime.agentlace_bridge import create_agentlace_async_learner
from serl_launcher.residual.runtime.agentlace_bridge import maybe_send_agentlace_timer_stats
from serl_launcher.residual.runtime.agentlace_bridge import maybe_wait_for_async_learner_budget
from serl_launcher.residual.runtime.agentlace_bridge import save_actor_bootstrap
from serl_launcher.residual.runtime.agentlace_bridge import sync_async_bounded_lag_baseline_from_learner
from serl_launcher.residual.runtime.async_learning import _AsyncLearner
from serl_launcher.residual.runtime.async_learning import _MixedBatchPrefetcher
from serl_launcher.residual.runtime.async_learning import _ProcessAsyncLearner
from serl_launcher.residual.runtime.async_learning import _sample_mixed_batch
from serl_launcher.residual.runtime.async_learning import _sync_agent_modules_inplace
from serl_launcher.residual.runtime.actor_support import ActorLoopState
from serl_launcher.residual.runtime.actor_support import ActorRuntimeContext
from serl_launcher.residual.runtime.actor_support import ensure_training_runtime_started
from serl_launcher.residual.runtime.actor_support import initialize_actor_loop_state
from serl_launcher.residual.runtime.actor_warmup import run_base_only_warmup
from serl_launcher.residual.runtime.checkpoint import _AsyncCheckpointWriter
from serl_launcher.residual.runtime.checkpoint import _CheckpointTask
from serl_launcher.residual.runtime.checkpoint import _snapshot_agent_checkpoint_payload
from serl_launcher.residual.runtime.checkpoint import _write_checkpoint_payload
from serl_launcher.residual.runtime.config_utils import build_drq_agent
from serl_launcher.residual.runtime.config_utils import build_residual_action_transform
from serl_launcher.residual.runtime.config_utils import resolve_action_mask_from_cfg
from serl_launcher.residual.runtime.config_utils import resolve_control_indices_from_cfg
from serl_launcher.residual.runtime.config_utils import resolve_residual_observation_state_mode
from serl_launcher.residual.runtime.config_utils import sample_probing_steps
from serl_launcher.residual.runtime.config_utils import set_global_seeds
from serl_launcher.residual.runtime.obs_utils import _clone_obs_dict
from serl_launcher.residual.runtime.obs_utils import _obs_space_from_sample
from serl_launcher.residual.runtime.obs_utils import _zero_obs_like
from serl_launcher.residual.runtime.pretrain import _pretrain_critic_with_calql
from serl_launcher.residual.runtime.profiling import _RuntimeProfiler
from serl_launcher.residual.runtime.profiling import _emit_profiling_snapshot
from serl_launcher.residual.runtime.profiling import _profile_call
from serl_launcher.residual.runtime.replay_batch import _consume_prepared_replay_batch
from serl_launcher.residual.runtime.replay_batch import _prepare_replay_batch
from serl_launcher.residual.runtime.schedules import _epsilon_gating_clock
from serl_launcher.residual.runtime.schedules import _epsilon_gating_enabled
from serl_launcher.residual.runtime.schedules import _scheduled_alpha
from serl_launcher.residual.runtime.schedules import _scheduled_epsilon_gating_probability
from serl_launcher.residual.runtime.step_chunk_replay import ChunkReplayBuffer
from serl_launcher.residual.runtime.tb_metrics import _append_tb_step_window
from serl_launcher.residual.runtime.tb_metrics import _flush_tb_step_window
from serl_launcher.residual.runtime.tb_metrics import _log_update_metrics
from serl_launcher.residual.runtime.tb_metrics import _new_tb_step_window
from serl_launcher.residual.runtime.train_loop_utils import _count_env_step_update_triggers
from serl_launcher.residual.runtime.train_loop_utils import _insert_online_transition
from serl_launcher.residual.runtime.train_loop_utils import _iter_period_hits
from serl_launcher.residual.runtime.train_loop_utils import _remaining_train_budget_steps
from serl_launcher.utils.alpha_utils import require_residual_alpha
from serl_launcher.utils.alpha_utils import validate_alpha
from serl_launcher.utils.logger import JsonlLogger
from serl_launcher.utils.serialization import _to_jsonable

from torch.utils.tensorboard import SummaryWriter

from serl_launcher.data.replay_buffer import ReplayBuffer




def run_actor_loop(
    ctx: ActorRuntimeContext,
    state: ActorLoopState,
) -> None:
    cfg = ctx.cfg
    run_dir = ctx.run_dir
    logger = ctx.logger
    bindings = ctx.bindings
    env = ctx.env
    normalizer = ctx.normalizer
    image_keys = tuple(ctx.image_keys)
    obs_cache = ctx.obs_cache
    task_key = str(ctx.task_key)
    data_config = ctx.data_config
    build_residual_step_obs_profiled = ctx.build_residual_step_obs_profiled
    build_residual_step_core = ctx.build_residual_step_core
    openpi_client = ctx.openpi_client
    openpi_prefetcher = ctx.openpi_prefetcher
    stack_horizon = int(ctx.stack_horizon)
    obs_state_mode = str(ctx.obs_state_mode)
    env_action_dim = int(ctx.env_action_dim)
    control_indices = ctx.control_indices
    step_action_dim = int(ctx.step_action_dim)
    chunk_horizon = int(ctx.chunk_horizon)
    residual_alpha = float(ctx.residual_alpha)
    chunk_step_enabled = bool(ctx.chunk_step_enabled)
    chunk_step_sample_stride = int(ctx.chunk_step_sample_stride)
    chunk_step_require_full_horizon = bool(ctx.chunk_step_require_full_horizon)
    chunk_step_pad_action = bool(ctx.chunk_step_pad_action)
    chunk_step_scheduler_clock = str(ctx.chunk_step_scheduler_clock)
    agent_action_dim = int(ctx.agent_action_dim)
    critic_action_dim = int(ctx.critic_action_dim)
    residual_limits = ctx.residual_limits
    action_mask = ctx.action_mask
    epsilon_gating_enabled = bool(ctx.epsilon_gating_enabled)
    epsilon_gating_clock = str(ctx.epsilon_gating_clock)
    resolved_cfg_dict = ctx.resolved_cfg_dict
    offline_enabled = bool(ctx.offline_enabled)
    offline_ratio = float(ctx.offline_ratio)
    symmetric_replay = bool(ctx.symmetric_replay)
    async_enabled = bool(ctx.async_enabled)
    async_update_frequency = int(ctx.async_update_frequency)
    async_idle_sleep_sec = float(ctx.async_idle_sleep_sec)
    async_backend = str(ctx.async_backend)
    async_actor_device = ctx.async_actor_device
    async_learner_device = ctx.async_learner_device
    async_batch_queue_size = int(ctx.async_batch_queue_size)
    async_trainer_host = str(ctx.async_trainer_host)
    async_trainer_port = int(ctx.async_trainer_port)
    async_broadcast_port = int(ctx.async_broadcast_port)
    async_data_store_queue_size = int(ctx.async_data_store_queue_size)
    async_stats_period_steps = int(ctx.async_stats_period_steps)
    async_agentlace_spawn_local_worker = bool(ctx.async_agentlace_spawn_local_worker)
    async_agentlace_bootstrap_file = str(ctx.async_agentlace_bootstrap_file)
    async_agentlace_connect_timeout_sec = float(ctx.async_agentlace_connect_timeout_sec)
    async_bounded_lag_enabled = bool(ctx.async_bounded_lag_enabled)
    async_bounded_lag_max_update_calls = int(ctx.async_bounded_lag_max_update_calls)
    async_bounded_lag_poll_sec = ctx.async_bounded_lag_poll_sec
    async_bounded_lag_timeout_sec = ctx.async_bounded_lag_timeout_sec
    async_bounded_lag_sync_on_wait = bool(ctx.async_bounded_lag_sync_on_wait)
    async_bounded_lag_log_period_steps = int(ctx.async_bounded_lag_log_period_steps)
    async_bounded_lag_env_steps_per_update_call = ctx.async_bounded_lag_env_steps_per_update_call
    async_bounded_lag_manual_rate_enabled = bool(ctx.async_bounded_lag_manual_rate_enabled)
    async_bounded_lag_mode = str(ctx.async_bounded_lag_mode)
    replay_prefetch_enabled = bool(ctx.replay_prefetch_enabled)
    replay_prefetch_queue_size = int(ctx.replay_prefetch_queue_size)
    replay_prefetch_pin_memory = bool(ctx.replay_prefetch_pin_memory)
    replay_prefetch_to_device = bool(ctx.replay_prefetch_to_device)
    profiling_enabled = bool(ctx.profiling_enabled)
    profiling_window_size = int(ctx.profiling_window_size)
    profiling_log_period_steps = int(ctx.profiling_log_period_steps)
    profiling_log_file = str(ctx.profiling_log_file)
    external_agentlace_actor_mode = bool(ctx.external_agentlace_actor_mode)
    manage_learner_state_locally = bool(ctx.manage_learner_state_locally)
    agentlace_bridge_config = ctx.agentlace_bridge_config
    agentlace_bridge_state = ctx.agentlace_bridge_state
    action_transform = ctx.action_transform
    action_space = ctx.action_space
    agent = ctx.agent
    learner_agent = ctx.learner_agent
    async_learner = ctx.async_learner
    sync_replay_lock = ctx.sync_replay_lock
    sync_replay_prefetcher = ctx.sync_replay_prefetcher
    checkpoint_writer = ctx.checkpoint_writer
    replay_buffer = ctx.replay_buffer
    offline_buffer = ctx.offline_buffer
    offline_stats = ctx.offline_stats
    warmstart_info = ctx.warmstart_info
    online_prefill_stats = ctx.online_prefill_stats
    checkpoint_every_steps = int(ctx.checkpoint_every_steps)
    checkpoint_keep = int(ctx.checkpoint_keep)
    checkpoint_dir = ctx.checkpoint_dir
    async_eval_enabled = bool(ctx.async_eval_enabled)
    async_eval_every_episodes = int(ctx.async_eval_every_episodes)
    async_eval_alpha_mode = str(ctx.async_eval_alpha_mode)
    async_eval_proc = ctx.async_eval_proc
    async_eval_log_fp = ctx.async_eval_log_fp
    async_eval_log_path = ctx.async_eval_log_path
    async_eval_summary_path = ctx.async_eval_summary_path
    async_eval_watcher_return_code = ctx.async_eval_watcher_return_code
    async_eval_dead_reported = bool(ctx.async_eval_dead_reported)
    profiler = ctx.profiler
    profiling_logger = ctx.profiling_logger
    profiling_last_flush_step = int(ctx.profiling_last_flush_step)
    async_eval_queue_path = ctx.async_eval_queue_path
    step_logger = ctx.step_logger
    episode_logger = ctx.episode_logger
    tb_writer = ctx.tb_writer
    tb_step_period = int(ctx.tb_step_period)
    tb_histogram_period = int(ctx.tb_histogram_period)
    progress_enabled = bool(ctx.progress_enabled)
    progress_mininterval_sec = float(ctx.progress_mininterval_sec)
    step_metric_window = ctx.step_metric_window
    async_eval_tb_sync_state = ctx.async_eval_tb_sync_state
    sample_obs = ctx.sample_obs
    sample_state_core = ctx.sample_state_core
    configured_warmup_episodes = int(ctx.configured_warmup_episodes)
    online_prefill_enabled = bool(ctx.online_prefill_enabled)
    online_prefill_loaded_episodes = int(ctx.online_prefill_loaded_episodes)
    warmup_episodes_cfg = int(ctx.warmup_episodes_cfg)
    need_warmup_first = bool(ctx.need_warmup_first)
    max_train_env_steps = int(ctx.max_train_env_steps)
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
        gate_prob = _scheduled_epsilon_gating_probability(
            cfg, schedule_step=schedule_step
        )
        if alpha_value <= 0.0:
            return float(gate_prob), False
        gate_on = bool(np.random.random() < float(gate_prob))
        return float(gate_prob), gate_on

    def _flush_external_agentlace_actor() -> None:
        if external_agentlace_actor_mode and async_learner is not None:
            async_learner.flush()

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
    ctx, state = build_actor_runtime_session(
        cfg,
        run_dir=run_dir,
        logger=logger,
        bindings=bindings,
        async_eval_watcher_path=async_eval_watcher_path,
    )
    env = ctx.env
    normalizer = ctx.normalizer
    image_keys = tuple(ctx.image_keys)
    obs_cache = ctx.obs_cache
    task_key = str(ctx.task_key)
    data_config = ctx.data_config
    build_residual_step_obs_profiled = ctx.build_residual_step_obs_profiled
    build_residual_step_core = ctx.build_residual_step_core
    openpi_client = ctx.openpi_client
    openpi_prefetcher = ctx.openpi_prefetcher
    stack_horizon = int(ctx.stack_horizon)
    obs_state_mode = str(ctx.obs_state_mode)
    env_action_dim = int(ctx.env_action_dim)
    control_indices = ctx.control_indices
    step_action_dim = int(ctx.step_action_dim)
    chunk_horizon = int(ctx.chunk_horizon)
    residual_alpha = float(ctx.residual_alpha)
    chunk_step_enabled = bool(ctx.chunk_step_enabled)
    chunk_step_sample_stride = int(ctx.chunk_step_sample_stride)
    chunk_step_require_full_horizon = bool(ctx.chunk_step_require_full_horizon)
    chunk_step_pad_action = bool(ctx.chunk_step_pad_action)
    chunk_step_scheduler_clock = str(ctx.chunk_step_scheduler_clock)
    agent_action_dim = int(ctx.agent_action_dim)
    critic_action_dim = int(ctx.critic_action_dim)
    residual_limits = ctx.residual_limits
    action_mask = ctx.action_mask
    epsilon_gating_enabled = bool(ctx.epsilon_gating_enabled)
    epsilon_gating_clock = str(ctx.epsilon_gating_clock)
    resolved_cfg_dict = ctx.resolved_cfg_dict
    offline_enabled = bool(ctx.offline_enabled)
    offline_ratio = float(ctx.offline_ratio)
    symmetric_replay = bool(ctx.symmetric_replay)
    async_enabled = bool(ctx.async_enabled)
    async_update_frequency = int(ctx.async_update_frequency)
    async_idle_sleep_sec = float(ctx.async_idle_sleep_sec)
    async_backend = str(ctx.async_backend)
    async_actor_device = ctx.async_actor_device
    async_learner_device = ctx.async_learner_device
    async_batch_queue_size = int(ctx.async_batch_queue_size)
    async_trainer_host = str(ctx.async_trainer_host)
    async_trainer_port = int(ctx.async_trainer_port)
    async_broadcast_port = int(ctx.async_broadcast_port)
    async_data_store_queue_size = int(ctx.async_data_store_queue_size)
    async_stats_period_steps = int(ctx.async_stats_period_steps)
    async_agentlace_spawn_local_worker = bool(ctx.async_agentlace_spawn_local_worker)
    async_agentlace_bootstrap_file = str(ctx.async_agentlace_bootstrap_file)
    async_agentlace_connect_timeout_sec = float(ctx.async_agentlace_connect_timeout_sec)
    async_bounded_lag_enabled = bool(ctx.async_bounded_lag_enabled)
    async_bounded_lag_max_update_calls = int(ctx.async_bounded_lag_max_update_calls)
    async_bounded_lag_poll_sec = ctx.async_bounded_lag_poll_sec
    async_bounded_lag_timeout_sec = ctx.async_bounded_lag_timeout_sec
    async_bounded_lag_sync_on_wait = bool(ctx.async_bounded_lag_sync_on_wait)
    async_bounded_lag_log_period_steps = int(ctx.async_bounded_lag_log_period_steps)
    async_bounded_lag_env_steps_per_update_call = ctx.async_bounded_lag_env_steps_per_update_call
    async_bounded_lag_manual_rate_enabled = bool(ctx.async_bounded_lag_manual_rate_enabled)
    async_bounded_lag_mode = str(ctx.async_bounded_lag_mode)
    replay_prefetch_enabled = bool(ctx.replay_prefetch_enabled)
    replay_prefetch_queue_size = int(ctx.replay_prefetch_queue_size)
    replay_prefetch_pin_memory = bool(ctx.replay_prefetch_pin_memory)
    replay_prefetch_to_device = bool(ctx.replay_prefetch_to_device)
    profiling_enabled = bool(ctx.profiling_enabled)
    profiling_window_size = int(ctx.profiling_window_size)
    profiling_log_period_steps = int(ctx.profiling_log_period_steps)
    profiling_log_file = str(ctx.profiling_log_file)
    external_agentlace_actor_mode = bool(ctx.external_agentlace_actor_mode)
    manage_learner_state_locally = bool(ctx.manage_learner_state_locally)
    agentlace_bridge_config = ctx.agentlace_bridge_config
    agentlace_bridge_state = ctx.agentlace_bridge_state
    action_transform = ctx.action_transform
    action_space = ctx.action_space
    agent = ctx.agent
    learner_agent = ctx.learner_agent
    async_learner = ctx.async_learner
    sync_replay_lock = ctx.sync_replay_lock
    sync_replay_prefetcher = ctx.sync_replay_prefetcher
    checkpoint_writer = ctx.checkpoint_writer
    replay_buffer = ctx.replay_buffer
    offline_buffer = ctx.offline_buffer
    offline_stats = ctx.offline_stats
    warmstart_info = ctx.warmstart_info
    online_prefill_stats = ctx.online_prefill_stats
    checkpoint_every_steps = int(ctx.checkpoint_every_steps)
    checkpoint_keep = int(ctx.checkpoint_keep)
    checkpoint_dir = ctx.checkpoint_dir
    async_eval_enabled = bool(ctx.async_eval_enabled)
    async_eval_every_episodes = int(ctx.async_eval_every_episodes)
    async_eval_alpha_mode = str(ctx.async_eval_alpha_mode)
    async_eval_proc = ctx.async_eval_proc
    async_eval_log_fp = ctx.async_eval_log_fp
    async_eval_log_path = ctx.async_eval_log_path
    async_eval_summary_path = ctx.async_eval_summary_path
    async_eval_watcher_return_code = ctx.async_eval_watcher_return_code
    async_eval_dead_reported = bool(ctx.async_eval_dead_reported)
    profiler = ctx.profiler
    profiling_logger = ctx.profiling_logger
    profiling_last_flush_step = int(ctx.profiling_last_flush_step)
    async_eval_queue_path = ctx.async_eval_queue_path
    step_logger = ctx.step_logger
    episode_logger = ctx.episode_logger
    tb_writer = ctx.tb_writer
    tb_step_period = int(ctx.tb_step_period)
    tb_histogram_period = int(ctx.tb_histogram_period)
    progress_enabled = bool(ctx.progress_enabled)
    progress_mininterval_sec = float(ctx.progress_mininterval_sec)
    step_metric_window = ctx.step_metric_window
    async_eval_tb_sync_state = ctx.async_eval_tb_sync_state
    sample_obs = ctx.sample_obs
    sample_state_core = ctx.sample_state_core
    configured_warmup_episodes = int(ctx.configured_warmup_episodes)
    online_prefill_enabled = bool(ctx.online_prefill_enabled)
    online_prefill_loaded_episodes = int(ctx.online_prefill_loaded_episodes)
    warmup_episodes_cfg = int(ctx.warmup_episodes_cfg)
    need_warmup_first = bool(ctx.need_warmup_first)
    max_train_env_steps = int(ctx.max_train_env_steps)
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

    def _new_progress(
        *, desc: str, total: Optional[int], position: int, leave: bool
    ) -> Optional[Any]:
        if not progress_enabled:
            return None
        return tqdm(
            total=total,
            desc=desc,
            dynamic_ncols=True,
            mininterval=progress_mininterval_sec,
            position=position,
            leave=leave,
        )

    if max_train_env_steps > 0:
        train_progress = _new_progress(
            desc="train_env_step",
            total=int(max_train_env_steps),
            position=0,
            leave=True,
        )

    def _update_train_progress(*, force_postfix: bool = False) -> None:
        nonlocal train_progress_last_step
        if train_progress is None:
            return
        delta = int(train_env_step) - int(train_progress_last_step)
        if delta > 0:
            train_progress.update(delta)
            train_progress_last_step = int(train_env_step)
        if force_postfix:
            completed_eval = int(async_eval_tb_sync_state.get("processed_lines", 0))
            pending_eval = max(0, int(eval_trigger_count) - int(completed_eval))
            train_progress.set_postfix(
                {"episode": int(train_episode_id), "eval_q": int(pending_eval)},
                refresh=False,
            )

    assert agent is not None
    assert learner_agent is not None
    assert replay_buffer is not None

    def _save_checkpoint_at_step(checkpoint_step: int) -> Path:
        checkpoint_path = checkpoint_dir / f"checkpoint_{int(checkpoint_step)}.pt"
        if int(checkpoint_step) in saved_checkpoint_steps:
            return checkpoint_path
        saved_checkpoint_steps.add(int(checkpoint_step))
        if async_learner is not None:
            async_learner.save_checkpoint(
                str(checkpoint_dir),
                step=int(checkpoint_step),
                keep=checkpoint_keep,
            )
        else:
            checkpoint_payload = _snapshot_agent_checkpoint_payload(
                learner_agent,
                step=int(checkpoint_step),
            )
            if checkpoint_writer is not None:
                checkpoint_writer.submit(
                    _CheckpointTask(
                        checkpoint_dir=str(checkpoint_dir),
                        payload=checkpoint_payload,
                        step=int(checkpoint_step),
                        keep=checkpoint_keep,
                    )
                )
            else:
                _write_checkpoint_payload(
                    profiler,
                    str(checkpoint_dir),
                    checkpoint_payload,
                    step=int(checkpoint_step),
                    keep=checkpoint_keep,
                )
        return checkpoint_path

    try:
        if need_warmup_first:
            run_base_only_warmup(ctx, state)
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
            train_progress = state.train_progress
            warmup_progress = state.warmup_progress
            phase_progress = state.phase_progress
            train_progress_last_step = int(state.train_progress_last_step)

        _sync_async_bounded_lag_baseline_from_learner()

        for phase in cfg.training.phases:
            if max_train_env_steps > 0 and train_env_step >= max_train_env_steps:
                stopped_by_env_budget = True
                break
            phase_name = str(phase.name)
            phase_episodes = int(phase.episodes)
            phase_train = bool(phase.get("train", True))
            logger.info(
                "Start phase=%s episodes=%s train=%s",
                phase_name,
                phase_episodes,
                phase_train,
            )

            phase_episode_count = 0
            phase_progress = _new_progress(
                desc=f"{phase_name}:episode",
                total=int(phase_episodes),
                position=1,
                leave=False,
            )
            try:
                while phase_episode_count < phase_episodes:
                    if (
                        max_train_env_steps > 0
                        and train_env_step >= max_train_env_steps
                    ):
                        stopped_by_env_budget = True
                        break

                    seed = int(seed_cursor)
                    seed_cursor += 1
                    current_phase_episode_idx = int(phase_episode_count + 1)
                    current_train_episode_id = (
                        int(train_episode_id + 1) if phase_train else None
                    )
                    current_init_episode_idx = int(init_episode_idx)

                    if bool(cfg.training.get("expert_check", False)):
                        passed, _ = env.expert_precheck(
                            seed=seed, init_episode_idx=current_init_episode_idx
                        )
                        if not passed:
                            skipped_seeds += 1
                            logger.warning(
                                "skip seed=%s in phase=%s: expert precheck failed",
                                seed,
                                phase_name,
                            )
                            continue

                    init_episode_idx += 1
                    obs_cache.clear()
                    obs_raw = _profile_call(
                        profiler,
                        "env_reset",
                        env.reset,
                        seed=seed,
                        init_episode_idx=current_init_episode_idx,
                    )
                    max_episode_steps = int(env.step_limit)
                    if cfg.training.max_env_steps_per_episode is not None:
                        max_episode_steps = min(
                            max_episode_steps,
                            int(cfg.training.max_env_steps_per_episode),
                        )

                    episode_success = False
                    episode_return = 0.0
                    episode_steps = 0
                    episode_done = False
                    cached_base_chunk = None
                    cached_infer_info = None

                    probing_steps_target = sample_probing_steps(
                        cfg.training, episode_horizon=max_episode_steps
                    )
                    if probing_steps_target > 0:
                        probing_remaining = int(
                            min(probing_steps_target, max_episode_steps - episode_steps)
                        )
                        probe_future: Optional[
                            Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]
                        ] = None
                        while (
                            probing_remaining > 0 and episode_steps < max_episode_steps
                        ):
                            if probe_future is not None:
                                probe_chunk, probe_info = probe_future.result()
                                probe_future = None
                            else:
                                probe_chunk, probe_info = openpi_client.infer_chunk(
                                    _policy_input(obs_raw, env.current_instruction)
                                )
                            probe_base_chunk = select_action_chunk_window(
                                probe_chunk,
                                horizon=chunk_horizon,
                                action_dim=env_action_dim,
                            )
                            for probe_step in range(chunk_horizon):
                                if (
                                    probing_remaining <= 0
                                    or episode_steps >= max_episode_steps
                                ):
                                    break
                                base_action = probe_base_chunk[probe_step]
                                next_obs_raw, reward, env_done, _, info = _profile_call(
                                    profiler,
                                    "env_step",
                                    env.step,
                                    base_action,
                                )
                                episode_steps += 1
                                if phase_train:
                                    train_env_step += 1
                                    _update_train_progress()
                                probing_remaining -= 1
                                episode_return += float(reward)
                                episode_success = bool(info["success"])
                                timeout = bool(episode_steps >= max_episode_steps)
                                budget_exhausted = bool(
                                    phase_train
                                    and max_train_env_steps > 0
                                    and train_env_step >= max_train_env_steps
                                )
                                done = bool(env_done or timeout or budget_exhausted)
                                if (
                                    (not done)
                                    and probe_step == (chunk_horizon - 1)
                                    and probing_remaining > 0
                                    and openpi_prefetcher is not None
                                ):
                                    probe_future = openpi_prefetcher.submit(
                                        _policy_input(next_obs_raw, env.current_instruction)
                                    )
                                step_logger.write(
                                    {
                                        "train_env_step": int(train_env_step)
                                        if phase_train
                                        else None,
                                        "decision_step": int(decision_step)
                                        if phase_train
                                        else None,
                                        "warmup_episode_id": None,
                                        "train_episode_id": current_train_episode_id,
                                        "phase_episode_idx": current_phase_episode_idx,
                                        "phase": phase_name,
                                        "episode_step": episode_steps,
                                        "seed": int(
                                            env.last_seed
                                            if env.last_seed is not None
                                            else seed
                                        ),
                                        "init_state_idx": (
                                            int(env.current_init_state_idx)
                                            if env.current_init_state_idx is not None
                                            else None
                                        ),
                                        "is_probing": True,
                                        "replan_point": bool(probe_step == 0),
                                        "chunk_step": int(probe_step),
                                        "chunk_horizon": int(chunk_horizon),
                                        "infer_e2e_ms": probe_info.get("e2e_ms")
                                        if probe_step == 0
                                        else None,
                                        "infer_policy_ms": probe_info.get("policy_ms")
                                        if probe_step == 0
                                        else None,
                                        "infer_server_ms": probe_info.get("server_ms")
                                        if probe_step == 0
                                        else None,
                                        "a_base": base_action.tolist(),
                                        "a_res_policy": [0.0] * step_action_dim,
                                        "a_res_policy_applied": [
                                            0.0
                                        ] * step_action_dim,
                                        "a_res": np.zeros_like(
                                            base_action, dtype=np.float32
                                        ).tolist(),
                                        "a_final": base_action.tolist(),
                                        "reward": float(reward),
                                        "done": bool(done),
                                        "success": bool(episode_success),
                                    }
                                )
                                obs_raw = next_obs_raw
                                if done:
                                    episode_done = True
                                    break
                            if episode_done:
                                break

                    while (episode_steps < max_episode_steps) and (not episode_done):
                        if cached_base_chunk is None:
                            openpi_chunk, infer_info = openpi_client.infer_chunk(
                                _policy_input(obs_raw, env.current_instruction)
                            )
                            base_chunk = select_action_chunk_window(
                                openpi_chunk,
                                horizon=chunk_horizon,
                                action_dim=env_action_dim,
                            )
                        else:
                            base_chunk = cached_base_chunk
                            infer_info = cached_infer_info or {
                                "e2e_ms": None,
                                "policy_ms": None,
                                "server_ms": None,
                            }
                            cached_base_chunk = None
                            cached_infer_info = None

                        if chunk_step_enabled:
                            train_env_step_before_chunk = int(train_env_step)
                            schedule_step = (
                                train_env_step_before_chunk
                                if chunk_step_scheduler_clock == "env_step"
                                else int(decision_step)
                            )
                            alpha_step = _scheduled_alpha(
                                cfg,
                                base_alpha=residual_alpha,
                                schedule_step=schedule_step,
                            )
                            obs_input = build_residual_step_obs_profiled(
                                profiler,
                                obs_raw,
                                base_chunk[0],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                normalizer=normalizer,
                                obs_cache=obs_cache,
                                base_action_chunk=base_chunk,
                                alpha=float(alpha_step),
                                state_mode=obs_state_mode,
                            )
                            gate_prob, gate_on = _resolve_train_gate(
                                phase_train_flag=bool(phase_train),
                                alpha_value=float(alpha_step),
                                env_step_value=int(train_env_step_before_chunk),
                                decision_step_value=int(decision_step),
                            )

                            if alpha_step <= 0.0:
                                residual_chunk = np.zeros(
                                    (chunk_horizon, step_action_dim), dtype=np.float32
                                )
                            elif (not phase_train) or (
                                train_env_step_before_chunk
                                < int(cfg.training.random_steps)
                            ):
                                residual_chunk = np.random.uniform(
                                    -1.0,
                                    1.0,
                                    size=(chunk_horizon, step_action_dim),
                                ).astype(np.float32)
                            else:
                                if async_learner is not None:
                                    sampled_chunk = async_learner.sample_actor_action(
                                        obs_input, agent_action_dim
                                    )
                                else:
                                    sample_actions_start = time.perf_counter()
                                    sampled = agent.sample_actions(
                                        obs_input, deterministic=False
                                    )
                                    profiler.record_duration(
                                        "agent_sample_actions",
                                        (time.perf_counter() - sample_actions_start)
                                        * 1000.0,
                                    )
                                    sampled_chunk = as_numpy_action(
                                        sampled, agent_action_dim
                                    )
                                residual_chunk = as_numpy_action_chunk(
                                    sampled_chunk,
                                    action_dim=step_action_dim,
                                    chunk_horizon=chunk_horizon,
                                )
                            policy_residual_chunk = np.asarray(
                                residual_chunk, dtype=np.float32
                            ).copy()
                            if not gate_on:
                                residual_chunk = np.zeros_like(residual_chunk)

                            remaining_budget_steps = (
                                _remaining_train_budget_steps(
                                    max_train_env_steps=max_train_env_steps,
                                    train_env_step=train_env_step,
                                )
                                if phase_train
                                else None
                            )
                            execute_horizon = int(
                                min(chunk_horizon, max_episode_steps - episode_steps)
                            )
                            if remaining_budget_steps is not None:
                                execute_horizon = int(
                                    min(execute_horizon, remaining_budget_steps)
                                )
                            if phase_train and train_env_step_before_chunk < int(
                                cfg.training.random_steps
                            ):
                                execute_horizon = int(
                                    min(
                                        execute_horizon,
                                        int(cfg.training.random_steps)
                                        - train_env_step_before_chunk,
                                    )
                                )
                            if execute_horizon <= 0:
                                episode_done = True
                                break

                            current_decision_id = (
                                int(decision_step + 1) if phase_train else None
                            )
                            replay_size_before = int(
                                _replay_progress_size(replay_buffer)
                            )
                            executed_base_chunk = np.asarray(
                                base_chunk[:execute_horizon], dtype=np.float32
                            )
                            executed_policy_residual_chunk = np.asarray(
                                policy_residual_chunk[:execute_horizon], dtype=np.float32
                            )
                            executed_residual_chunk = np.asarray(
                                residual_chunk[:execute_horizon], dtype=np.float32
                            )
                            delta_chunk, final_chunk = compose_residual_action_chunk(
                                base_chunk=executed_base_chunk,
                                residual_chunk=executed_residual_chunk,
                                indices=control_indices,
                                limits=residual_limits,
                                alpha=alpha_step,
                                clip_gripper=bool(cfg.residual.clip_gripper),
                            )

                            chunk_result = _profile_call(
                                profiler,
                                "env_step_chunk",
                                env.step_chunk,
                                final_chunk,
                            )
                            chunk_observations = list(chunk_result["observations"])
                            chunk_rewards = [float(v) for v in chunk_result["rewards"]]
                            chunk_infos = [dict(v) for v in chunk_result["infos"]]
                            chunk_env_dones = [bool(v) for v in chunk_result["dones"]]
                            next_obs_raw = chunk_result["obs"]
                            actual_chunk_steps = int(len(chunk_rewards))
                            if actual_chunk_steps <= 0:
                                raise RuntimeError(
                                    "env.step_chunk returned zero executed steps during training"
                                )
                            executed_base_chunk = executed_base_chunk[
                                :actual_chunk_steps
                            ]
                            executed_residual_chunk = executed_residual_chunk[
                                :actual_chunk_steps
                            ]
                            delta_chunk = delta_chunk[:actual_chunk_steps]
                            final_chunk = final_chunk[:actual_chunk_steps]

                            done = False
                            current_step_obs_raw = obs_raw
                            chunk_step_payloads = []
                            for chunk_step in range(actual_chunk_steps):
                                current_episode_step = int(episode_steps)
                                reward = float(chunk_rewards[chunk_step])
                                info = chunk_infos[chunk_step]
                                episode_steps += 1
                                if phase_train:
                                    train_env_step += 1
                                    _update_train_progress()
                                episode_return += reward
                                episode_success = bool(
                                    info.get("success", episode_success)
                                )

                                timeout = bool(episode_steps >= max_episode_steps)
                                remaining_budget_steps = (
                                    _remaining_train_budget_steps(
                                        max_train_env_steps=max_train_env_steps,
                                        train_env_step=train_env_step,
                                    )
                                    if phase_train
                                    else None
                                )
                                budget_exhausted = bool(
                                    remaining_budget_steps is not None
                                    and remaining_budget_steps <= 0
                                )
                                done = bool(
                                    chunk_env_dones[chunk_step]
                                    or timeout
                                    or budget_exhausted
                                )

                                step_logger.write(
                                    {
                                        "train_env_step": int(train_env_step)
                                        if phase_train
                                        else None,
                                        "decision_step": current_decision_id,
                                        "warmup_episode_id": None,
                                        "train_episode_id": current_train_episode_id,
                                        "phase_episode_idx": current_phase_episode_idx,
                                        "phase": phase_name,
                                        "episode_step": episode_steps,
                                        "seed": int(
                                            env.last_seed
                                            if env.last_seed is not None
                                            else seed
                                        ),
                                        "init_state_idx": (
                                            int(env.current_init_state_idx)
                                            if env.current_init_state_idx is not None
                                            else None
                                        ),
                                        "is_probing": False,
                                        "replan_point": bool(chunk_step == 0),
                                        "chunk_step": int(chunk_step),
                                        "chunk_horizon": int(actual_chunk_steps),
                                        "infer_e2e_ms": infer_info.get("e2e_ms")
                                        if chunk_step == 0
                                        else None,
                                        "infer_policy_ms": infer_info.get("policy_ms")
                                        if chunk_step == 0
                                        else None,
                                        "infer_server_ms": infer_info.get("server_ms")
                                        if chunk_step == 0
                                        else None,
                                        "a_base": executed_base_chunk[
                                            chunk_step
                                        ].tolist(),
                                        "a_res_policy": executed_policy_residual_chunk[
                                            chunk_step
                                        ].tolist(),
                                        "a_res_policy_applied": executed_residual_chunk[
                                            chunk_step
                                        ].tolist(),
                                        "a_res": delta_chunk[chunk_step].tolist(),
                                        "a_final": final_chunk[chunk_step].tolist(),
                                        "alpha": float(alpha_step),
                                        "epsilon_gate_prob": float(gate_prob),
                                        "epsilon_gate_on": bool(gate_on),
                                        "reward": float(reward),
                                        "done": bool(done),
                                        "success": bool(episode_success),
                                    }
                                )
                                step_payload = _build_chunk_step_record(
                                    current_step_obs_raw,
                                    base_action=executed_base_chunk[chunk_step],
                                    final_action=final_chunk[chunk_step],
                                    alpha_obs=float(alpha_step),
                                    episode_id=current_init_episode_idx,
                                    episode_step=current_episode_step,
                                    done=bool(done),
                                )
                                step_payload["rewards"] = float(reward)
                                chunk_step_payloads.append(step_payload)

                                _append_tb_step_window(
                                    step_metric_window,
                                    reward=float(reward),
                                    alpha=float(alpha_step),
                                    gate_prob=float(gate_prob),
                                    gate_on=bool(gate_on),
                                    residual_action_raw=executed_policy_residual_chunk[
                                        chunk_step
                                    ],
                                    residual_action_applied=executed_residual_chunk[
                                        chunk_step
                                    ],
                                    delta_action=delta_chunk[chunk_step],
                                    base_action=executed_base_chunk[chunk_step],
                                    final_action=final_chunk[chunk_step],
                                    infer_info=infer_info,
                                    replan_point=bool(chunk_step == 0),
                                )
                                if phase_train and train_env_step % tb_step_period == 0:
                                    _flush_tb_step_window(
                                        tb_writer,
                                        step_window=step_metric_window,
                                        global_env_step=train_env_step,
                                        control_indices=control_indices,
                                        histogram=bool(
                                            train_env_step % tb_histogram_period == 0
                                        ),
                                    )

                                if chunk_step < (actual_chunk_steps - 1):
                                    current_step_obs_raw = chunk_observations[
                                        chunk_step
                                    ]
                                if done:
                                    episode_done = True
                                    break

                            train_env_step_after_chunk = int(train_env_step)
                            if not done:
                                (
                                    next_openpi_chunk,
                                    next_infer_info,
                                ) = openpi_client.infer_chunk(
                                    _policy_input(next_obs_raw, env.current_instruction)
                                )
                                next_base_chunk = select_action_chunk_window(
                                    next_openpi_chunk,
                                    horizon=chunk_horizon,
                                    action_dim=env_action_dim,
                                )
                                cached_base_chunk = next_base_chunk
                                cached_infer_info = next_infer_info

                            if async_learner is not None:
                                with async_learner.replay_lock:
                                    for step_payload in chunk_step_payloads:
                                        _insert_online_transition(
                                            replay_buffer,
                                            step_payload,
                                            chunk_step_enabled=chunk_step_enabled,
                                        )
                            elif sync_replay_lock is not None:
                                with sync_replay_lock:
                                    for step_payload in chunk_step_payloads:
                                        _insert_online_transition(
                                            replay_buffer,
                                            step_payload,
                                            chunk_step_enabled=chunk_step_enabled,
                                        )
                            else:
                                for step_payload in chunk_step_payloads:
                                    _insert_online_transition(
                                        replay_buffer,
                                        step_payload,
                                        chunk_step_enabled=chunk_step_enabled,
                                    )

                            replay_size_after = int(
                                _replay_progress_size(replay_buffer)
                            )
                            if async_learner is None:
                                if phase_train:
                                    trigger_count = _count_env_step_update_triggers(
                                        train_step_before=train_env_step_before_chunk,
                                        train_step_after=train_env_step_after_chunk,
                                        replay_size_before=replay_size_before,
                                        replay_size_after=replay_size_after,
                                        training_starts=int(
                                            cfg.training.training_starts
                                        ),
                                        update_every=int(cfg.training.update_every),
                                    )
                                    for _ in range(
                                        int(
                                            trigger_count
                                            * int(cfg.training.updates_per_step)
                                        )
                                    ):
                                        if sync_replay_prefetcher is not None:
                                            sampled_batch = sync_replay_prefetcher.get(
                                                timeout=async_idle_sleep_sec
                                            )
                                            if sampled_batch is None:
                                                continue
                                            batch, online_bs, offline_bs = sampled_batch
                                        else:
                                            replay_sample_start = time.perf_counter()
                                            sampled = _sample_mixed_batch(
                                                replay_buffer,
                                                offline_buffer
                                                if offline_enabled
                                                else None,
                                                batch_size=int(cfg.replay.batch_size),
                                                offline_ratio=offline_ratio,
                                                symmetric_replay=symmetric_replay,
                                            )
                                            profiler.record_duration(
                                                "replay_sample",
                                                (
                                                    time.perf_counter()
                                                    - replay_sample_start
                                                )
                                                * 1000.0,
                                            )
                                            replay_prepare_start = time.perf_counter()
                                            prepared = _prepare_replay_batch(
                                                sampled,
                                                device=learner_agent.device,
                                                pin_memory=replay_prefetch_pin_memory,
                                                to_device=replay_prefetch_to_device,
                                                profiler=profiler,
                                                cuda_stream=None,
                                            )
                                            (
                                                batch,
                                                online_bs,
                                                offline_bs,
                                            ) = _consume_prepared_replay_batch(
                                                prepared,
                                                device=learner_agent.device,
                                                profiler=profiler,
                                            )
                                            profiler.record_duration(
                                                "replay_prepare",
                                                (
                                                    time.perf_counter()
                                                    - replay_prepare_start
                                                )
                                                * 1000.0,
                                            )
                                        update_start = time.perf_counter()
                                        (
                                            learner_agent,
                                            last_update_info,
                                        ) = learner_agent.update_high_utd(
                                            batch,
                                            utd_ratio=int(cfg.sac.utd_ratio),
                                        )
                                        profiler.record_duration(
                                            "agent_update_high_utd",
                                            (time.perf_counter() - update_start)
                                            * 1000.0,
                                        )
                                        last_update_info["online_batch_size"] = int(
                                            online_bs
                                        )
                                        last_update_info["offline_batch_size"] = int(
                                            offline_bs
                                        )
                                        last_update_info["offline_fraction"] = float(
                                            offline_bs / max(1, online_bs + offline_bs)
                                        )
                                    agent = learner_agent
                            else:
                                _advance_async_target_update_calls(
                                    phase_train_flag=bool(phase_train),
                                    train_step_before=int(train_env_step_before_chunk),
                                    train_step_after=int(train_env_step_after_chunk),
                                    replay_size_before=int(replay_size_before),
                                    replay_size_after=int(replay_size_after),
                                )
                                _maybe_wait_for_async_learner_budget(
                                    train_env_step_value=int(train_env_step_after_chunk),
                                    decision_step_value=int(
                                        current_decision_id
                                        if current_decision_id is not None
                                        else decision_step
                                    ),
                                )
                                last_update_info = async_learner.get_last_update_info()

                            if (
                                phase_train
                                and train_env_step % tb_step_period == 0
                                and last_update_info
                            ):
                                _log_update_metrics(
                                    tb_writer, last_update_info, train_env_step
                                )
                                tb_writer.add_scalar(
                                    "system/online_buffer_size",
                                    int(len(replay_buffer)),
                                    train_env_step,
                                )
                                if offline_buffer is not None:
                                    tb_writer.add_scalar(
                                        "system/offline_buffer_size",
                                        int(len(offline_buffer)),
                                        train_env_step,
                                    )
                                tb_writer.add_scalar(
                                    "system/decision_step",
                                    float(
                                        current_decision_id
                                        if current_decision_id is not None
                                        else 0
                                    ),
                                    train_env_step,
                                )
                                if async_learner is not None:
                                    tb_writer.add_scalar(
                                        "system/learner_update_steps",
                                        int(async_learner.get_update_steps()),
                                        train_env_step,
                                    )
                                    tb_writer.add_scalar(
                                        "system/replay_prefetch_queue_size",
                                        int(async_learner.get_prefetch_queue_size()),
                                        train_env_step,
                                    )
                                    if async_bounded_lag_enabled:
                                        tb_writer.add_scalar(
                                            "system/async_target_update_calls",
                                            float(
                                                agentlace_bridge_state.target_update_calls
                                            ),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_bounded_lag_tracked_env_steps",
                                            float(
                                                agentlace_bridge_state.tracked_env_steps
                                            ),
                                            train_env_step,
                                        )
                                        if (
                                            async_bounded_lag_env_steps_per_update_call
                                            is not None
                                        ):
                                            tb_writer.add_scalar(
                                                "system/async_env_steps_per_update_call",
                                                float(
                                                    async_bounded_lag_env_steps_per_update_call
                                                ),
                                                train_env_step,
                                            )
                                        tb_writer.add_scalar(
                                            "system/async_required_update_steps",
                                            float(
                                                agentlace_bridge_state.last_required_update_steps
                                            ),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_update_lag_before_wait",
                                            float(
                                                agentlace_bridge_state.last_lag_before_wait
                                            ),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_update_lag_after_wait",
                                            float(
                                                agentlace_bridge_state.last_lag_after_wait
                                            ),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_wait_last_sec",
                                            float(agentlace_bridge_state.last_wait_sec),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_wait_count",
                                            float(agentlace_bridge_state.wait_count),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_wait_timeout_count",
                                            float(agentlace_bridge_state.timeout_count),
                                            train_env_step,
                                        )
                                elif sync_replay_prefetcher is not None:
                                    tb_writer.add_scalar(
                                        "system/replay_prefetch_queue_size",
                                        int(sync_replay_prefetcher.get_queue_size()),
                                        train_env_step,
                                    )

                            if phase_train:
                                decision_step = int(current_decision_id)

                            if (
                                profiling_enabled
                                and profiling_log_period_steps > 0
                                and train_env_step > 0
                                and (train_env_step - profiling_last_flush_step)
                                >= profiling_log_period_steps
                            ):
                                _emit_profiling_snapshot(
                                    profiler,
                                    profile_logger=profiling_logger,
                                    tb_writer=tb_writer,
                                    logger=logger,
                                    train_env_step=train_env_step,
                                    decision_step=decision_step,
                                    train_episode_id=train_episode_id,
                                    learner_update_steps=(
                                        int(async_learner.get_update_steps())
                                        if async_learner is not None
                                        else 0
                                    ),
                                    replay_prefetch_queue_size=(
                                        int(async_learner.get_prefetch_queue_size())
                                        if async_learner is not None
                                        else int(
                                            sync_replay_prefetcher.get_queue_size()
                                        )
                                        if sync_replay_prefetcher is not None
                                        else 0
                                    ),
                                )
                                profiling_last_flush_step = int(train_env_step)
                            _maybe_send_agentlace_timer_stats(
                                train_env_step_value=int(train_env_step),
                                decision_step_value=int(decision_step),
                                train_episode_id_value=int(train_episode_id),
                            )

                            checkpoint_hits = _iter_period_hits(
                                step_before=train_env_step_before_chunk,
                                step_after=train_env_step_after_chunk,
                                period=checkpoint_every_steps,
                            )
                            if phase_train and checkpoint_hits:
                                for checkpoint_step in checkpoint_hits:
                                    _save_checkpoint_at_step(int(checkpoint_step))

                            obs_raw = next_obs_raw
                            if episode_done:
                                break
                        else:
                            next_obs_raw = obs_raw
                            for chunk_step in range(chunk_horizon):
                                if episode_steps >= max_episode_steps:
                                    episode_done = True
                                    break

                                train_env_step_before_step = int(train_env_step)
                                alpha_step = _scheduled_alpha(
                                    cfg,
                                    base_alpha=residual_alpha,
                                    schedule_step=train_env_step_before_step,
                                )
                                obs_input = build_residual_step_obs_profiled(
                                    profiler,
                                    next_obs_raw,
                                    base_chunk[chunk_step],
                                    image_keys=image_keys,
                                    stack_horizon=stack_horizon,
                                    normalizer=normalizer,
                                    obs_cache=obs_cache,
                                    alpha=float(alpha_step),
                                    state_mode=obs_state_mode,
                                )
                                gate_prob, gate_on = _resolve_train_gate(
                                    phase_train_flag=bool(phase_train),
                                    alpha_value=float(alpha_step),
                                    env_step_value=int(train_env_step_before_step),
                                    decision_step_value=int(decision_step),
                                )

                                if alpha_step <= 0.0:
                                    residual_step_action = np.zeros(
                                        (step_action_dim,), dtype=np.float32
                                    )
                                elif (not phase_train) or (
                                    train_env_step_before_step
                                    < int(cfg.training.random_steps)
                                ):
                                    residual_step_action = np.random.uniform(
                                        -1.0, 1.0, size=(step_action_dim,)
                                    ).astype(np.float32)
                                else:
                                    if async_learner is not None:
                                        residual_step_action = (
                                            async_learner.sample_actor_action(
                                                obs_input,
                                                step_action_dim,
                                            )
                                        )
                                    else:
                                        sample_actions_start = time.perf_counter()
                                        sampled = agent.sample_actions(
                                            obs_input, deterministic=False
                                        )
                                        profiler.record_duration(
                                            "agent_sample_actions",
                                            (time.perf_counter() - sample_actions_start)
                                            * 1000.0,
                                        )
                                        residual_step_action = as_numpy_action(
                                            sampled, step_action_dim
                                        )
                                policy_residual_step_action = np.asarray(
                                    residual_step_action, dtype=np.float32
                                ).copy()
                                if not gate_on:
                                    residual_step_action = np.zeros_like(
                                        residual_step_action
                                    )

                                delta_action, final_action = compose_residual_action(
                                    base_action=base_chunk[chunk_step],
                                    residual_action=residual_step_action,
                                    indices=control_indices,
                                    limits=residual_limits,
                                    alpha=alpha_step,
                                    clip_gripper=bool(cfg.residual.clip_gripper),
                                )

                                current_decision_id = (
                                    int(decision_step + 1) if phase_train else None
                                )
                                next_obs_raw, reward, env_done, _, info = _profile_call(
                                    profiler,
                                    "env_step",
                                    env.step,
                                    final_action,
                                )
                                episode_steps += 1
                                if phase_train:
                                    train_env_step += 1
                                    _update_train_progress()
                                train_env_step_after_step = int(train_env_step)
                                next_alpha_step = _scheduled_alpha(
                                    cfg,
                                    base_alpha=residual_alpha,
                                    schedule_step=train_env_step_after_step,
                                )
                                episode_return += float(reward)
                                episode_success = bool(info["success"])
                                timeout = bool(episode_steps >= max_episode_steps)
                                budget_exhausted = bool(
                                    phase_train
                                    and max_train_env_steps > 0
                                    and train_env_step >= max_train_env_steps
                                )
                                done = bool(env_done or timeout or budget_exhausted)
                                next_chunk_future: Optional[
                                    Future[
                                        Tuple[np.ndarray, Dict[str, Optional[float]]]
                                    ]
                                ] = None
                                if (
                                    (not done)
                                    and chunk_step == (chunk_horizon - 1)
                                    and openpi_prefetcher is not None
                                ):
                                    next_chunk_future = openpi_prefetcher.submit(
                                        _policy_input(next_obs_raw, env.current_instruction)
                                    )

                                step_logger.write(
                                    {
                                        "train_env_step": int(train_env_step)
                                        if phase_train
                                        else None,
                                        "decision_step": current_decision_id,
                                        "warmup_episode_id": None,
                                        "train_episode_id": current_train_episode_id,
                                        "phase_episode_idx": current_phase_episode_idx,
                                        "phase": phase_name,
                                        "episode_step": episode_steps,
                                        "seed": int(
                                            env.last_seed
                                            if env.last_seed is not None
                                            else seed
                                        ),
                                        "init_state_idx": (
                                            int(env.current_init_state_idx)
                                            if env.current_init_state_idx is not None
                                            else None
                                        ),
                                        "is_probing": False,
                                        "replan_point": bool(chunk_step == 0),
                                        "chunk_step": int(chunk_step),
                                        "chunk_horizon": int(chunk_horizon),
                                        "infer_e2e_ms": infer_info.get("e2e_ms")
                                        if chunk_step == 0
                                        else None,
                                        "infer_policy_ms": infer_info.get("policy_ms")
                                        if chunk_step == 0
                                        else None,
                                        "infer_server_ms": infer_info.get("server_ms")
                                        if chunk_step == 0
                                        else None,
                                        "a_base": base_chunk[chunk_step].tolist(),
                                        "a_res_policy": policy_residual_step_action.tolist(),
                                        "a_res_policy_applied": residual_step_action.tolist(),
                                        "a_res": delta_action.tolist(),
                                        "a_final": final_action.tolist(),
                                        "alpha": float(alpha_step),
                                        "epsilon_gate_prob": float(gate_prob),
                                        "epsilon_gate_on": bool(gate_on),
                                        "reward": float(reward),
                                        "done": bool(done),
                                        "success": bool(episode_success),
                                    }
                                )

                                _append_tb_step_window(
                                    step_metric_window,
                                    reward=float(reward),
                                    alpha=float(alpha_step),
                                    gate_prob=float(gate_prob),
                                    gate_on=bool(gate_on),
                                    residual_action_raw=policy_residual_step_action,
                                    residual_action_applied=residual_step_action,
                                    delta_action=delta_action,
                                    base_action=base_chunk[chunk_step],
                                    final_action=final_action,
                                    infer_info=infer_info,
                                    replan_point=bool(chunk_step == 0),
                                )
                                if phase_train and train_env_step % tb_step_period == 0:
                                    _flush_tb_step_window(
                                        tb_writer,
                                        step_window=step_metric_window,
                                        global_env_step=train_env_step,
                                        control_indices=control_indices,
                                        histogram=bool(
                                            train_env_step % tb_histogram_period == 0
                                        ),
                                    )

                                if done:
                                    next_obs_input = _zero_obs_like(obs_input)
                                    mask = 0.0
                                elif chunk_step < (chunk_horizon - 1):
                                    next_obs_input = build_residual_step_obs_profiled(
                                        profiler,
                                        next_obs_raw,
                                        base_chunk[chunk_step + 1],
                                        image_keys=image_keys,
                                        stack_horizon=stack_horizon,
                                        normalizer=normalizer,
                                        obs_cache=obs_cache,
                                        alpha=float(next_alpha_step),
                                        state_mode=obs_state_mode,
                                    )
                                    mask = 1.0
                                else:
                                    if next_chunk_future is not None:
                                        (
                                            next_openpi_chunk,
                                            next_infer_info,
                                        ) = next_chunk_future.result()
                                    else:
                                        (
                                            next_openpi_chunk,
                                            next_infer_info,
                                        ) = openpi_client.infer_chunk(
                                            _policy_input(next_obs_raw, env.current_instruction)
                                        )
                                    next_base_chunk = select_action_chunk_window(
                                        next_openpi_chunk,
                                        horizon=chunk_horizon,
                                        action_dim=env_action_dim,
                                    )
                                    next_obs_input = build_residual_step_obs_profiled(
                                        profiler,
                                        next_obs_raw,
                                        next_base_chunk[0],
                                        image_keys=image_keys,
                                        stack_horizon=stack_horizon,
                                        normalizer=normalizer,
                                        obs_cache=obs_cache,
                                        alpha=float(next_alpha_step),
                                        state_mode=obs_state_mode,
                                    )
                                    cached_base_chunk = next_base_chunk
                                    cached_infer_info = next_infer_info
                                    mask = 1.0

                                transition_payload = {
                                    "observations": _clone_obs_dict(obs_input),
                                    "actions": final_action.astype(np.float32),
                                    "next_observations": _clone_obs_dict(
                                        next_obs_input
                                    ),
                                    "rewards": np.float32(reward),
                                    "masks": np.float32(mask),
                                    "dones": bool(done),
                                    "episode_id": int(current_init_episode_idx),
                                    "episode_step": int(episode_steps - 1),
                                }
                                replay_size_before = int(
                                    _replay_progress_size(replay_buffer)
                                )
                                if async_learner is not None:
                                    with async_learner.replay_lock:
                                        _insert_online_transition(
                                            replay_buffer,
                                            transition_payload,
                                            chunk_step_enabled=chunk_step_enabled,
                                        )
                                elif sync_replay_lock is not None:
                                    with sync_replay_lock:
                                        _insert_online_transition(
                                            replay_buffer,
                                            transition_payload,
                                            chunk_step_enabled=chunk_step_enabled,
                                        )
                                else:
                                    _insert_online_transition(
                                        replay_buffer,
                                        transition_payload,
                                        chunk_step_enabled=chunk_step_enabled,
                                    )
                                replay_size_after = int(
                                    _replay_progress_size(replay_buffer)
                                )

                                if async_learner is None:
                                    if (
                                        phase_train
                                        and _replay_progress_size(replay_buffer)
                                        >= int(cfg.training.training_starts)
                                        and train_env_step_before_step
                                        % int(cfg.training.update_every)
                                        == 0
                                    ):
                                        for _ in range(
                                            int(cfg.training.updates_per_step)
                                        ):
                                            if sync_replay_prefetcher is not None:
                                                sampled_batch = (
                                                    sync_replay_prefetcher.get(
                                                        timeout=async_idle_sleep_sec
                                                    )
                                                )
                                                if sampled_batch is None:
                                                    continue
                                                (
                                                    batch,
                                                    online_bs,
                                                    offline_bs,
                                                ) = sampled_batch
                                            else:
                                                replay_sample_start = (
                                                    time.perf_counter()
                                                )
                                                sampled = _sample_mixed_batch(
                                                    replay_buffer,
                                                    offline_buffer
                                                    if offline_enabled
                                                    else None,
                                                    batch_size=int(
                                                        cfg.replay.batch_size
                                                    ),
                                                    offline_ratio=offline_ratio,
                                                    symmetric_replay=symmetric_replay,
                                                )
                                                profiler.record_duration(
                                                    "replay_sample",
                                                    (
                                                        time.perf_counter()
                                                        - replay_sample_start
                                                    )
                                                    * 1000.0,
                                                )
                                                replay_prepare_start = (
                                                    time.perf_counter()
                                                )
                                                prepared = _prepare_replay_batch(
                                                    sampled,
                                                    device=learner_agent.device,
                                                    pin_memory=replay_prefetch_pin_memory,
                                                    to_device=replay_prefetch_to_device,
                                                    profiler=profiler,
                                                    cuda_stream=None,
                                                )
                                                (
                                                    batch,
                                                    online_bs,
                                                    offline_bs,
                                                ) = _consume_prepared_replay_batch(
                                                    prepared,
                                                    device=learner_agent.device,
                                                    profiler=profiler,
                                                )
                                                profiler.record_duration(
                                                    "replay_prepare",
                                                    (
                                                        time.perf_counter()
                                                        - replay_prepare_start
                                                    )
                                                    * 1000.0,
                                                )
                                            update_start = time.perf_counter()
                                            (
                                                learner_agent,
                                                last_update_info,
                                            ) = learner_agent.update_high_utd(
                                                batch,
                                                utd_ratio=int(cfg.sac.utd_ratio),
                                            )
                                            profiler.record_duration(
                                                "agent_update_high_utd",
                                                (time.perf_counter() - update_start)
                                                * 1000.0,
                                            )
                                            last_update_info["online_batch_size"] = int(
                                                online_bs
                                            )
                                            last_update_info[
                                                "offline_batch_size"
                                            ] = int(offline_bs)
                                            last_update_info[
                                                "offline_fraction"
                                            ] = float(
                                                offline_bs
                                                / max(1, online_bs + offline_bs)
                                            )
                                        agent = learner_agent
                                else:
                                    _advance_async_target_update_calls(
                                        phase_train_flag=bool(phase_train),
                                        train_step_before=int(train_env_step_before_step),
                                        train_step_after=int(train_env_step_after_step),
                                        replay_size_before=int(replay_size_before),
                                        replay_size_after=int(replay_size_after),
                                    )
                                    _maybe_wait_for_async_learner_budget(
                                        train_env_step_value=int(train_env_step_after_step),
                                        decision_step_value=int(
                                            current_decision_id
                                            if current_decision_id is not None
                                            else decision_step
                                        ),
                                    )
                                    last_update_info = (
                                        async_learner.get_last_update_info()
                                    )

                                if (
                                    phase_train
                                    and train_env_step % tb_step_period == 0
                                    and last_update_info
                                ):
                                    _log_update_metrics(
                                        tb_writer, last_update_info, train_env_step
                                    )
                                    tb_writer.add_scalar(
                                        "system/online_buffer_size",
                                        int(len(replay_buffer)),
                                        train_env_step,
                                    )
                                    if offline_buffer is not None:
                                        tb_writer.add_scalar(
                                            "system/offline_buffer_size",
                                            int(len(offline_buffer)),
                                            train_env_step,
                                        )
                                    tb_writer.add_scalar(
                                        "system/decision_step",
                                        float(
                                            current_decision_id
                                            if current_decision_id is not None
                                            else 0
                                        ),
                                        train_env_step,
                                    )
                                    if async_learner is not None:
                                        tb_writer.add_scalar(
                                            "system/learner_update_steps",
                                            int(async_learner.get_update_steps()),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/replay_prefetch_queue_size",
                                            int(
                                                async_learner.get_prefetch_queue_size()
                                            ),
                                            train_env_step,
                                        )
                                        if async_bounded_lag_enabled:
                                            tb_writer.add_scalar(
                                                "system/async_target_update_calls",
                                                float(
                                                    agentlace_bridge_state.target_update_calls
                                                ),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_bounded_lag_tracked_env_steps",
                                                float(
                                                    agentlace_bridge_state.tracked_env_steps
                                                ),
                                                train_env_step,
                                            )
                                            if (
                                                async_bounded_lag_env_steps_per_update_call
                                                is not None
                                            ):
                                                tb_writer.add_scalar(
                                                    "system/async_env_steps_per_update_call",
                                                    float(
                                                        async_bounded_lag_env_steps_per_update_call
                                                    ),
                                                    train_env_step,
                                                )
                                            tb_writer.add_scalar(
                                                "system/async_required_update_steps",
                                                float(
                                                    agentlace_bridge_state.last_required_update_steps
                                                ),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_update_lag_before_wait",
                                                float(
                                                    agentlace_bridge_state.last_lag_before_wait
                                                ),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_update_lag_after_wait",
                                                float(
                                                    agentlace_bridge_state.last_lag_after_wait
                                                ),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_wait_last_sec",
                                                float(agentlace_bridge_state.last_wait_sec),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_wait_count",
                                                float(agentlace_bridge_state.wait_count),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_wait_timeout_count",
                                                float(agentlace_bridge_state.timeout_count),
                                                train_env_step,
                                            )
                                    elif sync_replay_prefetcher is not None:
                                        tb_writer.add_scalar(
                                            "system/replay_prefetch_queue_size",
                                            int(
                                                sync_replay_prefetcher.get_queue_size()
                                            ),
                                            train_env_step,
                                        )

                                if phase_train:
                                    decision_step = int(current_decision_id)

                                if (
                                    profiling_enabled
                                    and profiling_log_period_steps > 0
                                    and train_env_step > 0
                                    and (train_env_step - profiling_last_flush_step)
                                    >= profiling_log_period_steps
                                ):
                                    _emit_profiling_snapshot(
                                        profiler,
                                        profile_logger=profiling_logger,
                                        tb_writer=tb_writer,
                                        logger=logger,
                                        train_env_step=train_env_step,
                                        decision_step=decision_step,
                                        train_episode_id=train_episode_id,
                                        learner_update_steps=(
                                            int(async_learner.get_update_steps())
                                            if async_learner is not None
                                            else 0
                                        ),
                                        replay_prefetch_queue_size=(
                                            int(async_learner.get_prefetch_queue_size())
                                            if async_learner is not None
                                            else int(
                                                sync_replay_prefetcher.get_queue_size()
                                            )
                                            if sync_replay_prefetcher is not None
                                            else 0
                                        ),
                                    )
                                    profiling_last_flush_step = int(train_env_step)
                                _maybe_send_agentlace_timer_stats(
                                    train_env_step_value=int(train_env_step),
                                    decision_step_value=int(decision_step),
                                    train_episode_id_value=int(train_episode_id),
                                )

                                if (
                                    phase_train
                                    and checkpoint_every_steps > 0
                                    and train_env_step_after_step
                                    % checkpoint_every_steps
                                    == 0
                                ):
                                    _save_checkpoint_at_step(
                                        int(train_env_step_after_step)
                                    )

                                if done:
                                    episode_done = True
                                    break

                                obs_raw = next_obs_raw
                            if episode_done:
                                break

                    _flush_tb_step_window(
                        tb_writer,
                        step_window=step_metric_window,
                        global_env_step=max(0, int(train_env_step)),
                        control_indices=control_indices,
                        histogram=bool(
                            train_env_step > 0
                            and train_env_step % tb_histogram_period == 0
                        ),
                    )

                    if phase_train:
                        train_total_success += int(episode_success)
                        train_recent_successes.append(int(episode_success))
                        running_success_rate = float(train_total_success) / float(
                            current_train_episode_id
                        )
                        recent_success_rate = float(
                            sum(train_recent_successes)
                        ) / float(len(train_recent_successes))
                        episode_logger.write(
                            {
                                "phase": phase_name,
                                "warmup_episode_id": None,
                                "train_episode_id": current_train_episode_id,
                                "phase_episode_idx": current_phase_episode_idx,
                                "seed": int(
                                    env.last_seed if env.last_seed is not None else seed
                                ),
                                "init_state_idx": (
                                    int(env.current_init_state_idx)
                                    if env.current_init_state_idx is not None
                                    else None
                                ),
                                "success": bool(episode_success),
                                "episode_steps": int(episode_steps),
                                "episode_return": float(episode_return),
                                "train_env_step": int(train_env_step),
                                "decision_step": int(decision_step),
                                "running_success_rate": running_success_rate,
                                "recent_success_rate": recent_success_rate,
                            }
                        )
                        tb_writer.add_scalar(
                            "train_episode/success",
                            int(episode_success),
                            current_train_episode_id,
                        )
                        tb_writer.add_scalar(
                            "train_episode/return",
                            float(episode_return),
                            current_train_episode_id,
                        )
                        tb_writer.add_scalar(
                            "train_episode/length",
                            int(episode_steps),
                            current_train_episode_id,
                        )
                        tb_writer.add_scalar(
                            "train_episode/running_success_rate",
                            running_success_rate,
                            current_train_episode_id,
                        )
                        tb_writer.add_scalar(
                            "train_episode/recent_success_rate_20",
                            recent_success_rate,
                            current_train_episode_id,
                        )
                        tb_writer.add_scalar(
                            "system/online_buffer_size",
                            int(len(replay_buffer)),
                            train_env_step,
                        )
                        if offline_buffer is not None:
                            tb_writer.add_scalar(
                                "system/offline_buffer_size",
                                int(len(offline_buffer)),
                                train_env_step,
                            )
                        tb_writer.add_scalar(
                            "system/decision_step", int(decision_step), train_env_step
                        )
                        if async_learner is not None:
                            tb_writer.add_scalar(
                                "system/learner_update_steps",
                                int(async_learner.get_update_steps()),
                                train_env_step,
                            )
                            tb_writer.add_scalar(
                                "system/replay_prefetch_queue_size",
                                int(async_learner.get_prefetch_queue_size()),
                                train_env_step,
                            )
                        elif sync_replay_prefetcher is not None:
                            tb_writer.add_scalar(
                                "system/replay_prefetch_queue_size",
                                int(sync_replay_prefetcher.get_queue_size()),
                                train_env_step,
                            )

                        logger.info(
                            "phase=%s train_episode=%s success=%s steps=%s return=%.2f "
                            "train_env_step=%s success_rate=%.3f recent=%.3f",
                            phase_name,
                            current_train_episode_id,
                            episode_success,
                            episode_steps,
                            episode_return,
                            train_env_step,
                            running_success_rate,
                            recent_success_rate,
                        )
                        if external_agentlace_actor_mode and async_learner is not None:
                            _maybe_send_agentlace_timer_stats(
                                train_env_step_value=int(train_env_step),
                                decision_step_value=int(decision_step),
                                train_episode_id_value=int(current_train_episode_id),
                                force=True,
                            )
                            async_learner.request_stats(
                                {
                                    "train_episode": {
                                        "phase": str(phase_name),
                                        "train_episode_id": int(
                                            current_train_episode_id
                                        ),
                                        "success": bool(episode_success),
                                        "episode_steps": int(episode_steps),
                                        "episode_return": float(episode_return),
                                        "train_env_step": int(train_env_step),
                                        "decision_step": int(decision_step),
                                        "running_success_rate": float(
                                            running_success_rate
                                        ),
                                        "recent_success_rate": float(
                                            recent_success_rate
                                        ),
                                    }
                                }
                            )
                        train_episode_id = int(current_train_episode_id)

                        if (
                            async_eval_enabled
                            and async_eval_queue_path is not None
                            and train_episode_id % async_eval_every_episodes == 0
                        ):
                            checkpoint_path = _save_checkpoint_at_step(
                                int(train_env_step)
                            )
                            eval_index = int(eval_trigger_count)
                            _append_async_eval_request(
                                async_eval_queue_path,
                                {
                                    "eval_index": eval_index,
                                    "train_episode_id": int(train_episode_id),
                                    "train_env_step": int(train_env_step),
                                    "checkpoint_step": int(train_env_step),
                                    "checkpoint_path": str(checkpoint_path),
                                },
                            )
                            eval_trigger_count += 1
                    else:
                        episode_logger.write(
                            {
                                "phase": phase_name,
                                "warmup_episode_id": None,
                                "train_episode_id": None,
                                "phase_episode_idx": current_phase_episode_idx,
                                "seed": int(
                                    env.last_seed if env.last_seed is not None else seed
                                ),
                                "init_state_idx": (
                                    int(env.current_init_state_idx)
                                    if env.current_init_state_idx is not None
                                    else None
                                ),
                                "success": bool(episode_success),
                                "episode_steps": int(episode_steps),
                                "episode_return": float(episode_return),
                                "train_env_step": int(train_env_step),
                                "decision_step": None,
                                "running_success_rate": None,
                                "recent_success_rate": None,
                            }
                        )
                        logger.info(
                            "phase=%s phase_episode=%s success=%s steps=%s return=%.2f train_env_step=%s",
                            phase_name,
                            current_phase_episode_idx,
                            episode_success,
                            episode_steps,
                            episode_return,
                            train_env_step,
                        )

                    if async_eval_proc is not None and (not async_eval_dead_reported):
                        proc_rc = async_eval_proc.poll()
                        if proc_rc is not None:
                            async_eval_dead_reported = True
                            logger.warning(
                                "Async eval watcher exited early with returncode=%s; "
                                "see %s for details",
                                proc_rc,
                                async_eval_log_path,
                            )
                    _sync_async_eval_results_to_tb(
                        tb_writer,
                        summary_jsonl_path=async_eval_summary_path,
                        sync_state=async_eval_tb_sync_state,
                        logger=logger,
                    )
                    _update_train_progress(force_postfix=True)

                    phase_episode_count += 1
                    if phase_progress is not None:
                        phase_progress.update(1)
            finally:
                if phase_progress is not None:
                    phase_progress.close()
                    phase_progress = None

            if stopped_by_env_budget:
                break

        if async_learner is not None:
            async_learner.stop()
            last_update_info = async_learner.get_last_update_info()
        if checkpoint_writer is not None:
            checkpoint_writer.close(wait=True)
            checkpoint_writer = None

        final_profiling_payload = _emit_profiling_snapshot(
            profiler,
            profile_logger=profiling_logger,
            tb_writer=tb_writer,
            logger=logger,
            train_env_step=train_env_step,
            decision_step=decision_step,
            train_episode_id=train_episode_id,
            learner_update_steps=int(async_learner.get_update_steps())
            if async_learner is not None
            else 0,
            replay_prefetch_queue_size=(
                int(async_learner.get_prefetch_queue_size())
                if async_learner is not None
                else int(sync_replay_prefetcher.get_queue_size())
                if sync_replay_prefetcher is not None
                else 0
            ),
        )

        summary = {
            "train_env_step": int(train_env_step),
            "decision_step": int(decision_step),
            "train_episode_id": int(train_episode_id),
            "configured_warmup_episodes": int(configured_warmup_episodes),
            "warmup_episode_id": int(warmup_episode_id),
            "warmup_source": (
                "online_prefill"
                if int(online_prefill_loaded_episodes) > 0
                and int(warmup_episode_id) == int(online_prefill_loaded_episodes)
                else "online_prefill+runtime"
                if int(online_prefill_loaded_episodes) > 0
                else "runtime"
                if int(warmup_episode_id) > 0
                else "disabled"
            ),
            "train_total_success": int(train_total_success),
            "train_success_rate": float(
                train_total_success / max(1, int(train_episode_id))
            ),
            "warmup_total_success": int(warmup_total_success),
            "warmup_success_rate": float(
                warmup_total_success / max(1, int(warmup_episode_id))
            ),
            "skipped_seeds": int(skipped_seeds),
            "seed_start": int(cfg.task.seed_base),
            "seed_next": int(seed_cursor),
            "stopped_by_env_budget": bool(stopped_by_env_budget),
            "max_train_env_steps": int(max_train_env_steps),
            "chunk_step": {
                "enabled": bool(chunk_step_enabled),
                "sample_stride": int(chunk_step_sample_stride),
                "require_full_horizon": bool(chunk_step_require_full_horizon),
                "pad_action_to_horizon": bool(chunk_step_pad_action),
                "scheduler_clock": str(chunk_step_scheduler_clock),
                "step_action_dim": int(step_action_dim),
                "agent_action_dim": int(agent_action_dim),
                "chunk_horizon": int(chunk_horizon),
            },
            "replay_size": int(len(replay_buffer) if replay_buffer is not None else 0),
            "offline_enabled": bool(offline_enabled),
            "offline_ratio": float(offline_ratio),
            "offline_symmetric_replay": bool(symmetric_replay),
            "offline_buffer_size": int(
                len(offline_buffer) if offline_buffer is not None else 0
            ),
            "offline_stats": offline_stats,
            "online_prefill_stats": _to_jsonable(online_prefill_stats),
            "critic_pretrain": _to_jsonable(warmstart_info),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_every_steps": int(checkpoint_every_steps),
            "checkpoint_keep": int(checkpoint_keep),
            "last_update_info": _to_jsonable(last_update_info),
            "async_enabled": bool(async_enabled),
            "async_backend": str(async_backend),
            "async_update_frequency": int(async_update_frequency),
            "async_actor_device": async_actor_device,
            "async_learner_device": async_learner_device,
            "async_batch_queue_size": int(async_batch_queue_size),
            "async_trainer_host": str(async_trainer_host),
            "async_trainer_port": int(async_trainer_port),
            "async_broadcast_port": int(async_broadcast_port),
            "async_data_store_queue_size": int(async_data_store_queue_size),
            "learner_update_steps": int(
                async_learner.get_update_steps() if async_learner is not None else 0
            ),
            "async_bounded_lag": {
                "enabled": bool(async_bounded_lag_enabled),
                "mode": str(async_bounded_lag_mode),
                "max_update_lag_calls": int(async_bounded_lag_max_update_calls),
                "env_steps_per_update_call": (
                    None
                    if async_bounded_lag_env_steps_per_update_call is None
                    else float(async_bounded_lag_env_steps_per_update_call)
                ),
                "poll_sec": (
                    None
                    if async_bounded_lag_poll_sec is None
                    else float(async_bounded_lag_poll_sec)
                ),
                "timeout_sec": (
                    None
                    if async_bounded_lag_timeout_sec is None
                    else float(async_bounded_lag_timeout_sec)
                ),
                "sync_on_wait": bool(async_bounded_lag_sync_on_wait),
                "target_update_calls": int(agentlace_bridge_state.target_update_calls),
                "tracked_env_steps": int(agentlace_bridge_state.tracked_env_steps),
                "last_required_update_steps": int(
                    agentlace_bridge_state.last_required_update_steps
                ),
                "last_lag_before_wait": int(agentlace_bridge_state.last_lag_before_wait),
                "last_lag_after_wait": int(agentlace_bridge_state.last_lag_after_wait),
                "wait_count": int(agentlace_bridge_state.wait_count),
                "wait_timeout_count": int(agentlace_bridge_state.timeout_count),
                "wait_total_sec": float(agentlace_bridge_state.wait_total_sec),
                "last_wait_sec": float(agentlace_bridge_state.last_wait_sec),
            },
            "replay_prefetch_enabled": bool(replay_prefetch_enabled),
            "replay_prefetch_queue_size": int(replay_prefetch_queue_size),
            "replay_prefetch_pin_memory": bool(replay_prefetch_pin_memory),
            "replay_prefetch_to_device": bool(replay_prefetch_to_device),
            "profiling": {
                "enabled": bool(profiling_enabled),
                "window_size": int(profiling_window_size),
                "log_period_steps": int(profiling_log_period_steps),
                "log_file": str(run_dir / profiling_log_file)
                if profiling_enabled
                else None,
                "snapshot": (
                    _to_jsonable(final_profiling_payload.get("metrics", {}))
                    if final_profiling_payload is not None
                    else {}
                ),
            },
            "async_eval": {
                "enabled": bool(async_eval_enabled),
                "every_episodes": int(async_eval_every_episodes),
                "queue_path": str(async_eval_queue_path)
                if async_eval_queue_path is not None
                else None,
                "eval_trigger_count": int(eval_trigger_count),
                "watcher_started": bool(async_eval_proc is not None),
                "watcher_log_path": str(async_eval_log_path)
                if async_eval_log_path is not None
                else None,
                "summary_jsonl_path": (
                    str(async_eval_summary_path)
                    if async_eval_summary_path is not None
                    else None
                ),
                "watcher_return_code": (
                    int(async_eval_proc.returncode)
                    if async_eval_proc is not None
                    and async_eval_proc.returncode is not None
                    else None
                ),
            },
        }
        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("training done: %s", summary)

    finally:
        if async_learner is not None:
            async_learner.stop()
        if checkpoint_writer is not None:
            checkpoint_writer.close(wait=True)
        if sync_replay_prefetcher is not None:
            sync_replay_prefetcher.stop()
        if openpi_prefetcher is not None:
            openpi_prefetcher.close()
        if phase_progress is not None:
            phase_progress.close()
        if warmup_progress is not None:
            warmup_progress.close()
        if train_progress is not None:
            train_progress.close()
        async_eval_watcher_return_code = _stop_async_eval_watcher(
            async_eval_proc,
            async_eval_log_fp,
            logger=logger,
        )
        if async_eval_proc is not None:
            logger.info(
                "Async eval watcher stopped (returncode=%s, log=%s)",
                async_eval_watcher_return_code,
                async_eval_log_path,
            )
        _sync_async_eval_results_to_tb(
            tb_writer,
            summary_jsonl_path=async_eval_summary_path,
            sync_state=async_eval_tb_sync_state,
            logger=logger,
        )
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        step_logger.close()
        episode_logger.close()
        if profiling_logger is not None:
            profiling_logger.close()
        tb_writer.close()
