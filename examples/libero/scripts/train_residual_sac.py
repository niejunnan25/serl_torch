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
from pathlib import Path
from typing import IO, Any, Dict, Optional, Tuple

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

    # Keep legacy `gym.*` imports working when only Gymnasium is installed.
    sys.modules["gym"] = gym
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from serl_launcher.data.normalizer import StateActionNormalizer, load_normalizer
from serl_launcher.residual.data.training_loader import load_residual_training_buffer

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.data.training_config import (
    LIBERO_RESIDUAL_BASE_CONFIG,
)
from serl_torch.examples.libero.env_wrappers import (
    resolve_openpi_root,
    setup_openpi_client_pythonpath,
)
from serl_torch.examples.libero.env_wrappers.factory import _create_env
from serl_torch.examples.libero.policy import (
    LiberoObservationCache,
    OpenPIChunkClient,
    as_numpy_action,
    as_numpy_action_chunk,
    build_residual_step_core,
    build_residual_limits,
    compose_residual_action,
    compose_residual_action_chunk,
    select_action_chunk_window,
)
from serl_torch.examples.libero.policy.openpi_prefetch import (
    _AsyncOpenPIChunkPrefetcher,
)
from serl_torch.examples.libero.utils.async_eval import (
    _append_async_eval_request,
    _init_async_eval_tb_sync_state,
    _start_async_eval_watcher,
    _stop_async_eval_watcher,
    _sync_async_eval_results_to_tb,
)
from serl_torch.examples.libero.utils.async_learning import (
    _AgentlaceAsyncLearner,
    _AsyncLearner,
    _MixedBatchPrefetcher,
    _ProcessAsyncLearner,
    _sample_mixed_batch,
    _sync_agent_modules_inplace,
)
from serl_torch.examples.libero.utils import (
    JsonlLogger,
    ensure_serl_launcher_importable,
)
from serl_torch.examples.libero.utils.checkpoint import (
    _AsyncCheckpointWriter,
    _CheckpointTask,
    _snapshot_agent_checkpoint_payload,
    _write_checkpoint_payload,
)
from serl_torch.examples.libero.utils.config_utils import (
    build_residual_action_transform,
    build_drq_agent,
    resolve_control_indices_from_cfg,
    resolve_image_keys,
    resolve_residual_observation_state_mode,
    sample_probing_steps,
    set_global_seeds,
)
from serl_torch.examples.libero.utils.obs_utils import (
    _clone_obs_dict,
    _obs_space_from_sample,
    _zero_obs_like,
)
from serl_torch.examples.libero.utils.alpha_utils import (
    require_residual_alpha,
    validate_alpha,
)
from serl_torch.examples.libero.utils.agentlace_io import (
    resolve_agentlace_bootstrap_path,
    save_agentlace_bootstrap,
)
from serl_torch.examples.libero.utils.profiling import (
    _RuntimeProfiler,
    _build_residual_step_obs_profiled,
    _emit_profiling_snapshot,
    _profile_call,
)
from serl_torch.examples.libero.utils.pretrain import _pretrain_critic_with_calql
from serl_torch.examples.libero.utils.replay_batch import (
    _consume_prepared_replay_batch,
    _prepare_replay_batch,
)
from serl_torch.examples.libero.utils.schedules import (
    _epsilon_gating_clock,
    _epsilon_gating_enabled,
    _scheduled_alpha,
    _scheduled_epsilon_gating_probability,
)
from serl_torch.examples.libero.utils.serialization import _to_jsonable
from serl_torch.examples.libero.utils.step_chunk_replay import ChunkReplayBuffer
from serl_torch.examples.libero.utils.train_loop_utils import (
    _count_env_step_update_triggers,
    _insert_online_transition,
    _iter_period_hits,
    _remaining_train_budget_steps,
)
from serl_torch.examples.libero.utils.tb_metrics import (
    _append_tb_step_window,
    _flush_tb_step_window,
    _log_update_metrics,
    _new_tb_step_window,
)

ensure_serl_launcher_importable()

from torch.utils.tensorboard import SummaryWriter

from serl_launcher.data.replay_buffer import ReplayBuffer


@hydra.main(version_base=None, config_path="../conf", config_name="train_residual_sac")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger()
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_global_seeds(int(cfg.seed))

    openpi_root = resolve_openpi_root(cfg.get("openpi_root", None))
    setup_openpi_client_pythonpath(openpi_root)
    logger.info("openpi root: %s", openpi_root)

    env = _create_env(cfg, logger)
    logger.info(
        "LIBERO task: suite=%s task_id=%s prompt=%s",
        cfg.task.suite_name,
        cfg.task.task_id,
        env.current_instruction,
    )

    normalizer: StateActionNormalizer | None = None
    norm_cfg = cfg.get("normalization", None)
    if norm_cfg is not None and bool(norm_cfg.get("enabled", False)):
        task_key = f"{cfg.task.suite_name}_task_{int(cfg.task.task_id)}"
        stats_dir = norm_cfg.get(
            "stats_dir",
            str(Path(__file__).resolve().parents[1] / "data" / "stats"),
        )
        normalizer = load_normalizer(
            task_key, stats_dir=stats_dir
        )
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)
    obs_state_mode = resolve_residual_observation_state_mode(cfg)
    logger.info(
        "Residual observation state_mode=%s normalization.enabled=%s",
        obs_state_mode,
        bool(norm_cfg.get("enabled", False)) if norm_cfg is not None else False,
    )

    openpi_client = OpenPIChunkClient(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )

    image_keys = resolve_image_keys(cfg)
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
    epsilon_gating_enabled = _epsilon_gating_enabled(cfg)
    epsilon_gating_clock = _epsilon_gating_clock(cfg)
    logger.info(
        "Residual config: image_keys=%s step_action_dim=%s agent_action_dim=%s "
        "action_indices=%s env_action_dim=%s chunk_horizon=%s alpha=%.4f "
        "chunk_step_enabled=%s stride=%s limits=%s",
        list(image_keys),
        step_action_dim,
        agent_action_dim,
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
        gate_prob = _scheduled_epsilon_gating_probability(
            cfg, schedule_step=schedule_step
        )
        if alpha_value <= 0.0:
            return float(gate_prob), False
        gate_on = bool(np.random.random() < float(gate_prob))
        return float(gate_prob), gate_on

    offline_enabled = bool(cfg.offline.enabled)
    offline_dataset_paths_cfg = cfg.offline.get("dataset_paths", None)
    has_offline_dataset_paths = bool(offline_dataset_paths_cfg) and len(
        offline_dataset_paths_cfg
    ) > 0
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
    async_bounded_lag_enabled = (
        bool(async_enabled) and bool(async_bounded_lag_cfg_enabled)
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
    if (
        async_bounded_lag_env_steps_per_update_call is not None
        and (not async_bounded_lag_enabled)
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
    if offline_enabled and manage_learner_state_locally and (not has_offline_dataset_paths):
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
    checkpoint_writer: Optional[_AsyncCheckpointWriter] = None
    openpi_prefetcher: Optional[_AsyncOpenPIChunkPrefetcher] = None
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
    checkpoint_writer = _AsyncCheckpointWriter(profiler=profiler)
    profiling_logger: Optional[JsonlLogger] = None
    profiling_last_flush_step = -1
    async_eval_queue_path: Optional[Path] = None
    (
        async_eval_proc,
        async_eval_log_fp,
        async_eval_log_path,
        async_eval_summary_path,
        async_eval_queue_path,
    ) = _start_async_eval_watcher(
        cfg,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
    )

    step_logger = JsonlLogger(run_dir / str(cfg.logging.step_log_file))
    episode_logger = JsonlLogger(run_dir / str(cfg.logging.episode_log_file))
    openpi_prefetcher = _AsyncOpenPIChunkPrefetcher(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )
    if profiling_enabled:
        profiling_logger = JsonlLogger(run_dir / profiling_log_file)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    tb_step_period = int(cfg.logging.get("tb_step_period", 100))
    tb_histogram_period = max(
        tb_step_period,
        int(cfg.logging.get("tb_histogram_period", max(tb_step_period * 10, 1))),
    )
    progress_enabled = bool(cfg.logging.get("progress_bar", True))
    progress_mininterval_sec = float(cfg.logging.get("progress_mininterval_sec", 1.0))
    if progress_enabled and tqdm is None:
        logger.warning(
            "progress_bar=true but tqdm is unavailable; progress bars are disabled"
        )
        progress_enabled = False
    step_metric_window = _new_tb_step_window()
    async_eval_tb_sync_state = _init_async_eval_tb_sync_state(async_eval_summary_path)
    obs_cache = LiberoObservationCache()
    agentlace_timer_last_sent_step = -1
    async_target_update_calls = 0
    async_bounded_lag_tracked_env_steps = 0
    async_bounded_lag_wait_count = 0
    async_bounded_lag_timeout_count = 0
    async_bounded_lag_wait_total_sec = 0.0
    async_bounded_lag_last_wait_sec = 0.0
    async_bounded_lag_last_required_update_steps = 0
    async_bounded_lag_last_lag_before_wait = 0
    async_bounded_lag_last_lag_after_wait = 0

    def _build_agentlace_timer_payload(
        *,
        train_env_step_value: int,
        decision_step_value: int,
        train_episode_id_value: int,
    ) -> Optional[Dict[str, Any]]:
        if profiler is None or (not profiler.enabled) or (not profiler.has_data()):
            return None
        snapshot = profiler.snapshot()
        payload: Dict[str, Any] = {
            "train_env_step": int(train_env_step_value),
            "decision_step": int(decision_step_value),
            "train_episode_id": int(train_episode_id_value),
            "online_buffer_size": int(len(replay_buffer)) if replay_buffer is not None else 0,
        }
        if offline_buffer is not None:
            payload["offline_buffer_size"] = int(len(offline_buffer))
        if async_learner is not None:
            payload["learner_update_steps"] = int(async_learner.get_update_steps())
            payload["replay_prefetch_queue_size"] = int(
                async_learner.get_prefetch_queue_size()
            )
            if async_bounded_lag_enabled:
                payload["bounded_lag_mode"] = str(async_bounded_lag_mode)
                payload["bounded_lag_target_update_calls"] = int(
                    async_target_update_calls
                )
                payload["bounded_lag_tracked_env_steps"] = int(
                    async_bounded_lag_tracked_env_steps
                )
                if async_bounded_lag_env_steps_per_update_call is not None:
                    payload["bounded_lag_env_steps_per_update_call"] = float(
                        async_bounded_lag_env_steps_per_update_call
                    )
                payload["bounded_lag_required_update_steps"] = int(
                    async_bounded_lag_last_required_update_steps
                )
                payload["bounded_lag_lag_before_wait"] = int(
                    async_bounded_lag_last_lag_before_wait
                )
                payload["bounded_lag_lag_after_wait"] = int(
                    async_bounded_lag_last_lag_after_wait
                )
                payload["bounded_lag_wait_last_sec"] = float(
                    async_bounded_lag_last_wait_sec
                )
        elif sync_replay_prefetcher is not None:
            payload["replay_prefetch_queue_size"] = int(
                sync_replay_prefetcher.get_queue_size()
            )

        def _safe_name(name: str) -> str:
            return str(name).replace(".", "_").replace("/", "_").replace(" ", "_")

        for name, stats in snapshot.get("durations", {}).items():
            mean_ms = stats.get("mean_ms", None)
            if mean_ms is not None:
                payload[f"{_safe_name(name)}_mean_ms"] = float(mean_ms)
        for name, stats in snapshot.get("values", {}).items():
            mean_value = stats.get("mean", None)
            if mean_value is not None:
                payload[f"{_safe_name(name)}_mean"] = float(mean_value)
        return payload

    def _maybe_send_agentlace_timer_stats(
        *,
        train_env_step_value: int,
        decision_step_value: int,
        train_episode_id_value: int,
        force: bool = False,
    ) -> None:
        nonlocal agentlace_timer_last_sent_step
        if (not external_agentlace_actor_mode) or async_learner is None:
            return
        current_step = int(train_env_step_value)
        period = max(1, int(async_stats_period_steps))
        if (not force) and agentlace_timer_last_sent_step >= 0:
            if (current_step - agentlace_timer_last_sent_step) < period:
                return
        payload = _build_agentlace_timer_payload(
            train_env_step_value=int(train_env_step_value),
            decision_step_value=int(decision_step_value),
            train_episode_id_value=int(train_episode_id_value),
        )
        if payload is None:
            return
        async_learner.request_stats({"timer": payload})
        agentlace_timer_last_sent_step = current_step

    def _advance_async_target_update_calls(
        *,
        phase_train_flag: bool,
        train_step_before: int,
        train_step_after: int,
        replay_size_before: int,
        replay_size_after: int,
    ) -> int:
        nonlocal async_target_update_calls
        nonlocal async_bounded_lag_tracked_env_steps
        if (
            (not async_bounded_lag_enabled)
            or async_learner is None
            or (not bool(phase_train_flag))
        ):
            return 0
        if async_bounded_lag_manual_rate_enabled:
            added_trainable_env_steps = _count_env_step_update_triggers(
                train_step_before=int(train_step_before),
                train_step_after=int(train_step_after),
                replay_size_before=int(replay_size_before),
                replay_size_after=int(replay_size_after),
                training_starts=int(cfg.training.training_starts),
                update_every=1,
            )
            if added_trainable_env_steps <= 0:
                return 0
            async_bounded_lag_tracked_env_steps += int(added_trainable_env_steps)
            previous_target_update_calls = int(async_target_update_calls)
            async_target_update_calls = int(
                async_bounded_lag_tracked_env_steps
                // float(async_bounded_lag_env_steps_per_update_call)
            )
            if async_target_update_calls < previous_target_update_calls:
                async_target_update_calls = int(previous_target_update_calls)
            return int(async_target_update_calls - previous_target_update_calls)
        trigger_count = _count_env_step_update_triggers(
            train_step_before=int(train_step_before),
            train_step_after=int(train_step_after),
            replay_size_before=int(replay_size_before),
            replay_size_after=int(replay_size_after),
            training_starts=int(cfg.training.training_starts),
            update_every=int(cfg.training.update_every),
        )
        added_update_calls = int(trigger_count * int(cfg.training.updates_per_step))
        if added_update_calls > 0:
            async_target_update_calls += int(added_update_calls)
        return int(added_update_calls)

    def _maybe_wait_for_async_learner_budget(
        *,
        train_env_step_value: int,
        decision_step_value: int,
    ) -> None:
        nonlocal async_bounded_lag_wait_count
        nonlocal async_bounded_lag_timeout_count
        nonlocal async_bounded_lag_wait_total_sec
        nonlocal async_bounded_lag_last_wait_sec
        nonlocal async_bounded_lag_last_required_update_steps
        nonlocal async_bounded_lag_last_lag_before_wait
        nonlocal async_bounded_lag_last_lag_after_wait
        if (not async_bounded_lag_enabled) or async_learner is None:
            return
        required_update_steps = max(
            0,
            int(async_target_update_calls) - int(async_bounded_lag_max_update_calls),
        )
        async_bounded_lag_last_required_update_steps = int(required_update_steps)
        current_update_steps = int(async_learner.get_update_steps())
        lag_before_wait = max(0, required_update_steps - current_update_steps)
        async_bounded_lag_last_lag_before_wait = int(lag_before_wait)
        async_bounded_lag_last_lag_after_wait = int(lag_before_wait)
        async_bounded_lag_last_wait_sec = 0.0
        if lag_before_wait <= 0:
            return

        wait_start = time.perf_counter()
        updated_steps = int(
            async_learner.wait_for_update_steps(
                required_update_steps,
                poll_interval_sec=async_bounded_lag_poll_sec,
                timeout_sec=async_bounded_lag_timeout_sec,
            )
        )
        if async_bounded_lag_sync_on_wait:
            async_learner.sync_now(timeout_sec=async_bounded_lag_timeout_sec)
            updated_steps = int(async_learner.get_update_steps())
        wait_sec = float(time.perf_counter() - wait_start)
        lag_after_wait = max(0, required_update_steps - updated_steps)

        async_bounded_lag_wait_count += 1
        async_bounded_lag_wait_total_sec += wait_sec
        async_bounded_lag_last_wait_sec = float(wait_sec)
        async_bounded_lag_last_lag_after_wait = int(lag_after_wait)

        if lag_after_wait > 0:
            async_bounded_lag_timeout_count += 1
            logger.warning(
                "Bounded async lag timeout: train_env_step=%s decision_step=%s "
                "required_update_steps=%s learner_update_steps=%s remaining_lag=%s "
                "wait_sec=%.3f",
                int(train_env_step_value),
                int(decision_step_value),
                int(required_update_steps),
                int(updated_steps),
                int(lag_after_wait),
                float(wait_sec),
            )
            return

        log_period = max(0, int(async_bounded_lag_log_period_steps))
        if log_period > 0 and int(train_env_step_value) % log_period == 0:
            logger.info(
                "Bounded async lag wait complete: train_env_step=%s decision_step=%s "
                "target_update_calls=%s required_update_steps=%s learner_update_steps=%s "
                "wait_sec=%.3f",
                int(train_env_step_value),
                int(decision_step_value),
                int(async_target_update_calls),
                int(required_update_steps),
                int(updated_steps),
                float(wait_sec),
            )

    def _sync_async_bounded_lag_baseline_from_learner() -> None:
        nonlocal async_target_update_calls
        nonlocal async_bounded_lag_tracked_env_steps
        if (not async_bounded_lag_enabled) or async_learner is None:
            return
        learner_update_steps = int(async_learner.get_update_steps())
        previous_target_update_calls = int(async_target_update_calls)
        if learner_update_steps > int(async_target_update_calls):
            async_target_update_calls = int(learner_update_steps)
        if async_bounded_lag_manual_rate_enabled:
            min_tracked_env_steps = int(async_target_update_calls) * int(
                async_bounded_lag_env_steps_per_update_call
            )
            if min_tracked_env_steps > int(async_bounded_lag_tracked_env_steps):
                async_bounded_lag_tracked_env_steps = int(min_tracked_env_steps)
        if int(async_target_update_calls) != int(previous_target_update_calls):
            logger.info(
                "Aligned bounded-lag baseline to learner update steps: "
                "target_update_calls %s -> %s learner_update_steps=%s mode=%s",
                int(previous_target_update_calls),
                int(async_target_update_calls),
                int(learner_update_steps),
                str(async_bounded_lag_mode),
            )

    sample_obs_raw = _profile_call(
        profiler,
        "env_reset",
        env.reset,
        seed=int(cfg.task.seed_base),
        init_episode_idx=-1,
    )
    sample_openpi_chunk, _ = openpi_client.infer_chunk(
        sample_obs_raw,
        env.current_instruction,
        obs_cache=obs_cache,
    )
    sample_base_chunk = select_action_chunk_window(
        sample_openpi_chunk,
        horizon=chunk_horizon,
        action_dim=env_action_dim,
    )
    sample_obs = _build_residual_step_obs_profiled(
        profiler,
        sample_obs_raw,
        sample_base_chunk[0],
        image_keys=image_keys,
        stack_horizon=stack_horizon,
        normalizer=normalizer,
        obs_cache=obs_cache,
        base_action_chunk=(sample_base_chunk if chunk_step_enabled else None),
        alpha=float(residual_alpha),
        state_mode=obs_state_mode,
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
        async_learner_device
        if async_enabled and manage_learner_state_locally
        else None
    )
    actor_agent_device = (
        async_actor_device
        if async_enabled and async_backend in {"process", "agentlace"}
        else None
    )
    learner_agent = build_drq_agent(
        cfg,
        sample_obs=sample_obs,
        action_dim=agent_action_dim,
        image_keys=image_keys,
        critic_action_dim=critic_action_dim,
        action_transform=action_transform,
        device=learner_agent_device,
    )
    if async_enabled and async_backend in {"process", "agentlace"}:
        agent = build_drq_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=agent_action_dim,
            image_keys=image_keys,
            critic_action_dim=critic_action_dim,
            action_transform=action_transform,
            device=actor_agent_device,
        )
        _sync_agent_modules_inplace(agent, learner_agent)
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
            state_mode=obs_state_mode,
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
                state_mode=obs_state_mode,
            )
        else:
            offline_buffer = ReplayBuffer(
                observation_space=_obs_space_from_sample(sample_obs),
                action_space=action_space,
                capacity=int(cfg.offline.capacity),
            )
        offline_residual_alpha = _scheduled_alpha(
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
            data_config=LIBERO_RESIDUAL_BASE_CONFIG,
            normalizer=normalizer,
            profiler=profiler,
            state_mode=obs_state_mode,
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
        async_agentlace_bootstrap_path = resolve_agentlace_bootstrap_path(
            run_dir=run_dir,
            bootstrap_file=async_agentlace_bootstrap_file,
        )
        save_agentlace_bootstrap(
            async_agentlace_bootstrap_path,
            {
                "sample_obs": sample_obs,
                "state_core_dim": int(sample_state_core.shape[0]),
                "env_action_dim": int(env_action_dim),
                "step_action_dim": int(step_action_dim),
                "agent_action_dim": int(agent_action_dim),
                "critic_action_dim": int(critic_action_dim),
                "image_keys": tuple(image_keys),
                "action_transform": action_transform,
                "chunk_step_enabled": bool(chunk_step_enabled),
                "chunk_horizon": int(chunk_horizon),
                "state_mode": str(obs_state_mode),
                "initial_agent_payload": _snapshot_agent_checkpoint_payload(
                    learner_agent,
                    step=int(learner_agent.state.step),
                ),
                "saved_at_unix": float(time.time()),
            },
        )
        logger.info("Agentlace bootstrap saved to %s", async_agentlace_bootstrap_path)

    if external_agentlace_actor_mode:
        if offline_enabled:
            logger.info(
                "External agentlace actor mode: offline replay/pretrain will be owned by the external learner process."
            )
        async_learner = _AgentlaceAsyncLearner(
            actor_agent=agent,
            replay_buffer=replay_buffer,
            offline_buffer=None,
            batch_size=int(cfg.replay.batch_size),
            offline_ratio=offline_ratio,
            symmetric_replay=symmetric_replay,
            training_starts=int(cfg.training.training_starts),
            update_frequency=async_update_frequency,
            idle_sleep_sec=async_idle_sleep_sec,
            cfg_dict=OmegaConf.to_container(cfg, resolve=True),
            sample_obs=sample_obs,
            action_dim=agent_action_dim,
            critic_action_dim=critic_action_dim,
            image_keys=image_keys,
            action_transform=action_transform,
            learner_device=async_learner_device,
            host=async_trainer_host,
            port_number=async_trainer_port,
            broadcast_port=async_broadcast_port,
            data_store_queue_size=async_data_store_queue_size,
            replay_capacity=(
                int(getattr(replay_buffer, "capacity"))
                if hasattr(replay_buffer, "capacity")
                else int(getattr(replay_buffer, "_capacity"))
                if hasattr(replay_buffer, "_capacity")
                else None
            ),
            spawn_local_worker=False,
            connect_timeout_sec=async_agentlace_connect_timeout_sec,
        )
        async_learner.start()
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
            data_config=LIBERO_RESIDUAL_BASE_CONFIG,
            normalizer=normalizer,
            profiler=profiler,
            max_episodes=configured_warmup_episodes,
            state_mode=obs_state_mode,
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
            async_learner = _AgentlaceAsyncLearner(
                actor_agent=agent,
                replay_buffer=agentlace_replay_buffer,
                offline_buffer=(
                    offline_buffer
                    if offline_enabled and manage_learner_state_locally
                    else None
                ),
                batch_size=int(cfg.replay.batch_size),
                offline_ratio=offline_ratio,
                symmetric_replay=symmetric_replay,
                training_starts=int(cfg.training.training_starts),
                update_frequency=async_update_frequency,
                idle_sleep_sec=async_idle_sleep_sec,
                cfg_dict=OmegaConf.to_container(cfg, resolve=True),
                sample_obs=sample_obs,
                action_dim=agent_action_dim,
                critic_action_dim=critic_action_dim,
                image_keys=image_keys,
                action_transform=action_transform,
                learner_device=async_learner_device,
                host=async_trainer_host,
                port_number=async_trainer_port,
                broadcast_port=async_broadcast_port,
                data_store_queue_size=async_data_store_queue_size,
                replay_capacity=(
                    int(getattr(agentlace_replay_buffer, "capacity"))
                    if hasattr(agentlace_replay_buffer, "capacity")
                    else int(getattr(agentlace_replay_buffer, "_capacity"))
                    if hasattr(agentlace_replay_buffer, "_capacity")
                    else None
                ),
                spawn_local_worker=async_agentlace_spawn_local_worker,
                connect_timeout_sec=async_agentlace_connect_timeout_sec,
            )
            async_learner.start()
            replay_buffer = async_learner.replay_proxy
        elif async_backend == "process":
            async_learner = _ProcessAsyncLearner(
                actor_agent=agent,
                online_buffer=replay_buffer,
                offline_buffer=offline_buffer if offline_enabled else None,
                batch_size=int(cfg.replay.batch_size),
                offline_ratio=offline_ratio,
                symmetric_replay=symmetric_replay,
                training_starts=int(cfg.training.training_starts),
                update_frequency=async_update_frequency,
                idle_sleep_sec=async_idle_sleep_sec,
                cfg_dict=OmegaConf.to_container(cfg, resolve=True),
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
            agent = build_drq_agent(
                cfg,
                sample_obs=sample_obs,
                action_dim=agent_action_dim,
                image_keys=image_keys,
                critic_action_dim=critic_action_dim,
                action_transform=action_transform,
                device=async_actor_device,
            )
            _sync_agent_modules_inplace(agent, learner_agent)
            async_learner = _AsyncLearner(
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

    train_env_step = 0
    decision_step = 0
    train_episode_id = 0
    warmup_episode_id = int(online_prefill_loaded_episodes)
    init_episode_idx = int(online_prefill_loaded_episodes)
    eval_trigger_count = 0
    train_total_success = 0
    train_recent_successes: deque[int] = deque(maxlen=20)
    warmup_total_success = int(online_prefill_stats.get("success_episodes", 0))
    warmup_recent_successes: deque[int] = deque(
        [
            int(v)
            for v in online_prefill_stats.get("recent_episode_successes", [])
        ],
        maxlen=20,
    )
    skipped_seeds = 0
    seed_cursor = int(cfg.task.seed_base) + int(online_prefill_loaded_episodes)
    stopped_by_env_budget = False
    last_update_info: Dict[str, Any] = {}
    saved_checkpoint_steps: set[int] = set()

    max_train_env_steps = int(cfg.training.get("max_train_env_steps", 0))
    train_progress: Optional[Any] = None
    warmup_progress: Optional[Any] = None
    phase_progress: Optional[Any] = None
    train_progress_last_step = 0

    def _new_progress(
        *, desc: str, total: Optional[int], position: int, leave: bool
    ) -> Optional[Any]:
        if (not progress_enabled) or tqdm is None:
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
            if online_prefill_loaded_episodes > 0:
                logger.info(
                    "Warmup phase: collecting remaining %s/%s base-only episodes "
                    "after loading online prefill, no actor/critic updates",
                    warmup_episodes_cfg,
                    configured_warmup_episodes,
                )
            else:
                logger.info(
                    "Warmup phase: collecting %s base-only episodes, no actor/critic updates",
                    warmup_episodes_cfg,
                )
            warmup_progress = _new_progress(
                desc="warmup_episode",
                total=int(warmup_episodes_cfg),
                position=1,
                leave=False,
            )
            for _ in range(warmup_episodes_cfg):
                current_warmup_episode_id = int(warmup_episode_id + 1)
                current_init_episode_idx = int(init_episode_idx)
                init_episode_idx += 1

                seed = int(seed_cursor)
                seed_cursor += 1
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
                while (episode_steps < max_episode_steps) and (not episode_done):
                    if cached_base_chunk is None:
                        openpi_chunk, infer_info = openpi_client.infer_chunk(
                            obs_raw,
                            env.current_instruction,
                            obs_cache=obs_cache,
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
                        alpha_step = 0.0
                        execute_horizon = int(
                            min(chunk_horizon, max_episode_steps - episode_steps)
                        )
                        executed_base_chunk = np.asarray(
                            base_chunk[:execute_horizon], dtype=np.float32
                        )
                        chunk_result = _profile_call(
                            profiler,
                            "env_step_chunk",
                            env.step_chunk,
                            executed_base_chunk,
                        )
                        chunk_observations = list(chunk_result["observations"])
                        next_obs_raw = chunk_result["obs"]
                        chunk_rewards = [float(v) for v in chunk_result["rewards"]]
                        chunk_infos = [dict(v) for v in chunk_result["infos"]]
                        chunk_dones = [bool(v) for v in chunk_result["dones"]]
                        actual_chunk_steps = int(len(chunk_rewards))
                        if actual_chunk_steps <= 0:
                            raise RuntimeError(
                                "Warmup chunk execution returned zero steps"
                            )
                        executed_base_chunk = executed_base_chunk[:actual_chunk_steps]

                        done = False
                        current_step_obs_raw = obs_raw
                        for chunk_step in range(actual_chunk_steps):
                            current_episode_step = int(episode_steps)
                            reward = float(chunk_rewards[chunk_step])
                            info = chunk_infos[chunk_step]
                            episode_steps += 1
                            episode_return += float(reward)
                            episode_success = bool(info.get("success", episode_success))
                            timeout = bool(episode_steps >= max_episode_steps)
                            done = bool(chunk_dones[chunk_step] or timeout)
                            step_logger.write(
                                {
                                    "train_env_step": None,
                                    "decision_step": None,
                                    "warmup_episode_id": current_warmup_episode_id,
                                    "train_episode_id": None,
                                    "phase_episode_idx": current_warmup_episode_id,
                                    "phase": "warmup_base_only",
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
                                    "is_warmup": True,
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
                                    "a_base": executed_base_chunk[chunk_step].tolist(),
                                    "a_res_policy": np.zeros(
                                        (step_action_dim,), dtype=np.float32
                                    ).tolist(),
                                    "a_res_policy_applied": np.zeros(
                                        (step_action_dim,), dtype=np.float32
                                    ).tolist(),
                                    "a_res": np.zeros(
                                        (env_action_dim,), dtype=np.float32
                                    ).tolist(),
                                    "a_final": executed_base_chunk[chunk_step].tolist(),
                                    "alpha": 0.0,
                                    "alpha_obs": float(alpha_step),
                                    "alpha_exec": 0.0,
                                    "reward": float(reward),
                                    "done": bool(done),
                                    "success": bool(episode_success),
                                }
                            )
                            step_payload = _build_chunk_step_record(
                                current_step_obs_raw,
                                base_action=executed_base_chunk[chunk_step],
                                final_action=executed_base_chunk[chunk_step],
                                alpha_obs=float(alpha_step),
                                episode_id=current_init_episode_idx,
                                episode_step=current_episode_step,
                                done=bool(done),
                            )
                            step_payload["rewards"] = float(reward)
                            _insert_online_transition(
                                replay_buffer,
                                step_payload,
                                chunk_step_enabled=chunk_step_enabled,
                            )
                            if chunk_step < (actual_chunk_steps - 1):
                                current_step_obs_raw = chunk_observations[chunk_step]
                            if done:
                                episode_done = True
                                break

                        if not done:
                            (
                                next_openpi_chunk,
                                next_infer_info,
                            ) = openpi_client.infer_chunk(
                                next_obs_raw,
                                env.current_instruction,
                                obs_cache=obs_cache,
                            )
                            next_base_chunk = select_action_chunk_window(
                                next_openpi_chunk,
                                horizon=chunk_horizon,
                                action_dim=env_action_dim,
                            )
                            cached_base_chunk = next_base_chunk
                            cached_infer_info = next_infer_info
                        obs_raw = next_obs_raw
                        continue

                    next_obs_raw = obs_raw
                    for chunk_step in range(chunk_horizon):
                        if episode_steps >= max_episode_steps:
                            episode_done = True
                            break
                        alpha_step = 0.0
                        obs_input = _build_residual_step_obs_profiled(
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
                        residual_step_action = np.zeros(
                            (step_action_dim,), dtype=np.float32
                        )
                        final_action = base_chunk[chunk_step].copy()
                        next_obs_raw, reward, env_done, _, info = _profile_call(
                            profiler,
                            "env_step",
                            env.step,
                            final_action,
                        )
                        episode_steps += 1
                        episode_return += float(reward)
                        episode_success = bool(info["success"])
                        timeout = bool(episode_steps >= max_episode_steps)
                        done = bool(env_done or timeout)
                        next_alpha_step = 0.0
                        next_chunk_future = None
                        if (
                            (not done)
                            and chunk_step == (chunk_horizon - 1)
                            and openpi_prefetcher is not None
                        ):
                            next_chunk_future = openpi_prefetcher.submit(
                                next_obs_raw,
                                env.current_instruction,
                                obs_cache=obs_cache,
                            )
                        if done:
                            next_obs_input = _zero_obs_like(obs_input)
                            mask = 0.0
                        elif chunk_step < (chunk_horizon - 1):
                            next_obs_input = _build_residual_step_obs_profiled(
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
                                    next_obs_raw,
                                    env.current_instruction,
                                    obs_cache=obs_cache,
                                )
                            next_base_chunk = select_action_chunk_window(
                                next_openpi_chunk,
                                horizon=chunk_horizon,
                                action_dim=env_action_dim,
                            )
                            next_obs_input = _build_residual_step_obs_profiled(
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
                            "next_observations": _clone_obs_dict(next_obs_input),
                            "rewards": np.float32(reward),
                            "masks": np.float32(mask),
                            "dones": bool(done),
                            "episode_id": int(current_init_episode_idx),
                            "episode_step": int(episode_steps - 1),
                            "decision_id": int(chunk_step == 0),
                            "is_decision_start": bool(chunk_step == 0),
                        }
                        _insert_online_transition(
                            replay_buffer,
                            transition_payload,
                            chunk_step_enabled=chunk_step_enabled,
                        )
                        step_logger.write(
                            {
                                "train_env_step": None,
                                "decision_step": None,
                                "warmup_episode_id": current_warmup_episode_id,
                                "train_episode_id": None,
                                "phase_episode_idx": current_warmup_episode_id,
                                "phase": "warmup_base_only",
                                "episode_step": episode_steps,
                                "seed": int(
                                    env.last_seed if env.last_seed is not None else seed
                                ),
                                "init_state_idx": (
                                    int(env.current_init_state_idx)
                                    if env.current_init_state_idx is not None
                                    else None
                                ),
                                "is_warmup": True,
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
                                "a_res_policy": residual_step_action.tolist(),
                                "a_res_policy_applied": np.zeros(
                                    (step_action_dim,), dtype=np.float32
                                ).tolist(),
                                "a_res": np.zeros_like(
                                    base_chunk[chunk_step], dtype=np.float32
                                ).tolist(),
                                "a_final": final_action.tolist(),
                                "alpha": 0.0,
                                "alpha_obs": float(alpha_step),
                                "alpha_exec": 0.0,
                                "reward": float(reward),
                                "done": bool(done),
                                "success": bool(episode_success),
                            }
                        )
                        if done:
                            episode_done = True
                            break
                    obs_raw = next_obs_raw

                warmup_total_success += int(episode_success)
                warmup_recent_successes.append(int(episode_success))
                warmup_running_success_rate = float(warmup_total_success) / float(
                    current_warmup_episode_id
                )
                warmup_recent_success_rate = float(
                    sum(warmup_recent_successes)
                ) / float(len(warmup_recent_successes))
                episode_logger.write(
                    {
                        "phase": "warmup_base_only",
                        "warmup_episode_id": current_warmup_episode_id,
                        "train_episode_id": None,
                        "phase_episode_idx": current_warmup_episode_id,
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
                        "train_env_step": None,
                        "decision_step": None,
                        "running_success_rate": warmup_running_success_rate,
                        "recent_success_rate": warmup_recent_success_rate,
                        "is_warmup": True,
                    }
                )
                tb_writer.add_scalar(
                    "warmup/episode/success",
                    int(episode_success),
                    current_warmup_episode_id,
                )
                tb_writer.add_scalar(
                    "warmup/episode/return",
                    float(episode_return),
                    current_warmup_episode_id,
                )
                tb_writer.add_scalar(
                    "warmup/episode/length",
                    int(episode_steps),
                    current_warmup_episode_id,
                )
                tb_writer.add_scalar(
                    "warmup/episode/running_success_rate",
                    warmup_running_success_rate,
                    current_warmup_episode_id,
                )
                tb_writer.add_scalar(
                    "warmup/episode/recent_success_rate_20",
                    warmup_recent_success_rate,
                    current_warmup_episode_id,
                )
                tb_writer.add_scalar(
                    "warmup/system/online_buffer_size",
                    int(len(replay_buffer)),
                    current_warmup_episode_id,
                )
                logger.info(
                    "warmup episode %s/%s success=%s steps=%s return=%.2f",
                    current_warmup_episode_id,
                    configured_warmup_episodes,
                    episode_success,
                    episode_steps,
                    episode_return,
                )
                _flush_external_agentlace_actor()
                warmup_episode_id = current_warmup_episode_id
                if warmup_progress is not None:
                    warmup_progress.update(1)
                    warmup_progress.set_postfix(
                        {"success": int(episode_success)},
                        refresh=False,
                    )

            logger.info(
                "Warmup complete. Warmup episodes=%s total_success=%s buffer_size=%s. "
                "Starting residual training phase.",
                warmup_episode_id,
                warmup_total_success,
                len(replay_buffer),
            )
            if warmup_progress is not None:
                warmup_progress.close()
                warmup_progress = None
            if async_learner is None and async_enabled:
                if async_backend == "agentlace":
                    agentlace_replay_buffer = replay_buffer
                    async_learner = _AgentlaceAsyncLearner(
                        actor_agent=agent,
                        replay_buffer=agentlace_replay_buffer,
                        offline_buffer=(
                            offline_buffer
                            if offline_enabled and manage_learner_state_locally
                            else None
                        ),
                        batch_size=int(cfg.replay.batch_size),
                        offline_ratio=offline_ratio,
                        symmetric_replay=symmetric_replay,
                        training_starts=int(cfg.training.training_starts),
                        update_frequency=async_update_frequency,
                        idle_sleep_sec=async_idle_sleep_sec,
                        cfg_dict=OmegaConf.to_container(cfg, resolve=True),
                        sample_obs=sample_obs,
                        action_dim=agent_action_dim,
                        critic_action_dim=critic_action_dim,
                        image_keys=image_keys,
                        action_transform=action_transform,
                        learner_device=async_learner_device,
                        host=async_trainer_host,
                        port_number=async_trainer_port,
                        broadcast_port=async_broadcast_port,
                        data_store_queue_size=async_data_store_queue_size,
                        replay_capacity=(
                            int(getattr(agentlace_replay_buffer, "capacity"))
                            if hasattr(agentlace_replay_buffer, "capacity")
                            else int(getattr(agentlace_replay_buffer, "_capacity"))
                            if hasattr(agentlace_replay_buffer, "_capacity")
                            else None
                        ),
                        spawn_local_worker=async_agentlace_spawn_local_worker,
                        connect_timeout_sec=async_agentlace_connect_timeout_sec,
                    )
                    async_learner.start()
                    replay_buffer = async_learner.replay_proxy
                elif async_backend == "process":
                    async_learner = _ProcessAsyncLearner(
                        actor_agent=agent,
                        online_buffer=replay_buffer,
                        offline_buffer=offline_buffer if offline_enabled else None,
                        batch_size=int(cfg.replay.batch_size),
                        offline_ratio=offline_ratio,
                        symmetric_replay=symmetric_replay,
                        training_starts=int(cfg.training.training_starts),
                        update_frequency=async_update_frequency,
                        idle_sleep_sec=async_idle_sleep_sec,
                        cfg_dict=OmegaConf.to_container(cfg, resolve=True),
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
                    agent = build_drq_agent(
                        cfg,
                        sample_obs=sample_obs,
                        action_dim=agent_action_dim,
                        image_keys=image_keys,
                        critic_action_dim=critic_action_dim,
                        action_transform=action_transform,
                        device=async_actor_device,
                    )
                    _sync_agent_modules_inplace(agent, learner_agent)
                    async_learner = _AsyncLearner(
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
            elif replay_prefetch_enabled:
                sync_replay_lock = threading.Lock()

                def _sample_sync_prefetch_batch() -> Optional[
                    Tuple[Dict[str, Any], int, int]
                ]:
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
                                    obs_raw,
                                    env.current_instruction,
                                    obs_cache=obs_cache,
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
                                        next_obs_raw,
                                        env.current_instruction,
                                        obs_cache=obs_cache,
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
                                obs_raw,
                                env.current_instruction,
                                obs_cache=obs_cache,
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
                            obs_input = _build_residual_step_obs_profiled(
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
                                    next_obs_raw,
                                    env.current_instruction,
                                    obs_cache=obs_cache,
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
                                            float(async_target_update_calls),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_bounded_lag_tracked_env_steps",
                                            float(async_bounded_lag_tracked_env_steps),
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
                                                async_bounded_lag_last_required_update_steps
                                            ),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_update_lag_before_wait",
                                            float(
                                                async_bounded_lag_last_lag_before_wait
                                            ),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_update_lag_after_wait",
                                            float(
                                                async_bounded_lag_last_lag_after_wait
                                            ),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_wait_last_sec",
                                            float(async_bounded_lag_last_wait_sec),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_wait_count",
                                            float(async_bounded_lag_wait_count),
                                            train_env_step,
                                        )
                                        tb_writer.add_scalar(
                                            "system/async_wait_timeout_count",
                                            float(async_bounded_lag_timeout_count),
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
                                obs_input = _build_residual_step_obs_profiled(
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
                                        next_obs_raw,
                                        env.current_instruction,
                                        obs_cache=obs_cache,
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
                                    next_obs_input = _build_residual_step_obs_profiled(
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
                                            next_obs_raw,
                                            env.current_instruction,
                                            obs_cache=obs_cache,
                                        )
                                    next_base_chunk = select_action_chunk_window(
                                        next_openpi_chunk,
                                        horizon=chunk_horizon,
                                        action_dim=env_action_dim,
                                    )
                                    next_obs_input = _build_residual_step_obs_profiled(
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
                                                float(async_target_update_calls),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_bounded_lag_tracked_env_steps",
                                                float(async_bounded_lag_tracked_env_steps),
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
                                                    async_bounded_lag_last_required_update_steps
                                                ),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_update_lag_before_wait",
                                                float(
                                                    async_bounded_lag_last_lag_before_wait
                                                ),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_update_lag_after_wait",
                                                float(
                                                    async_bounded_lag_last_lag_after_wait
                                                ),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_wait_last_sec",
                                                float(async_bounded_lag_last_wait_sec),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_wait_count",
                                                float(async_bounded_lag_wait_count),
                                                train_env_step,
                                            )
                                            tb_writer.add_scalar(
                                                "system/async_wait_timeout_count",
                                                float(async_bounded_lag_timeout_count),
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
                "target_update_calls": int(async_target_update_calls),
                "tracked_env_steps": int(async_bounded_lag_tracked_env_steps),
                "last_required_update_steps": int(
                    async_bounded_lag_last_required_update_steps
                ),
                "last_lag_before_wait": int(async_bounded_lag_last_lag_before_wait),
                "last_lag_after_wait": int(async_bounded_lag_last_lag_after_wait),
                "wait_count": int(async_bounded_lag_wait_count),
                "wait_timeout_count": int(async_bounded_lag_timeout_count),
                "wait_total_sec": float(async_bounded_lag_wait_total_sec),
                "last_wait_sec": float(async_bounded_lag_last_wait_sec),
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


if __name__ == "__main__":
    main()
