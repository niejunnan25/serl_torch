from __future__ import annotations

"""
LIBERO residual policy training (OpenPI + DrQ-SAC).

Minimal stage-1 residual RL loop:
1. OpenPI predicts a base action chunk.
2. Residual policy predicts one residual action per environment step.
3. Final action = base action + bounded residual.
4. Transitions are written step-wise into replay.
5. DrQ-SAC updates from online replay only.

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
from typing import IO, Any, Dict, List, Optional, Tuple

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

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.data import StateActionNormalizer, load_normalizer
from serl_torch.examples.libero.data.offline_bootstrap import _bootstrap_offline_with_base_success
from serl_torch.examples.libero.data.offline_residual import _load_offline_residual_buffer
from serl_torch.examples.libero.env_wrappers import (
    resolve_openpi_root,
    setup_openpi_client_pythonpath,
)
from serl_torch.examples.libero.env_wrappers.factory import _create_env
from serl_torch.examples.libero.policy import (
    LiberoObservationCache,
    OpenPIChunkClient,
    as_numpy_action,
    build_residual_limits,
    compose_residual_action,
    select_action_chunk_window,
)
from serl_torch.examples.libero.policy.openpi_prefetch import _AsyncOpenPIChunkPrefetcher
from serl_torch.examples.libero.utils.async_eval import (
    _init_async_eval_tb_sync_state,
    _start_async_eval_watcher,
    _stop_async_eval_watcher,
    _sync_async_eval_results_to_tb,
)
from serl_torch.examples.libero.utils.async_learning import (
    _AsyncLearner,
    _MixedBatchPrefetcher,
    _sample_mixed_batch,
    _sync_agent_modules_inplace,
)
from serl_torch.examples.libero.utils import JsonlLogger, ensure_serl_launcher_importable
from serl_torch.examples.libero.utils.checkpoint import (
    _AsyncCheckpointWriter,
    _CheckpointTask,
    _snapshot_agent_checkpoint_payload,
    _write_checkpoint_payload,
)
from serl_torch.examples.libero.utils.config_utils import (
    build_drq_agent,
    resolve_control_indices_from_cfg,
    resolve_image_keys,
    sample_probing_steps,
    set_global_seeds,
)
from serl_torch.examples.libero.utils.obs_utils import (
    _clone_obs_dict,
    _obs_space_from_sample,
    _zero_obs_like,
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
from serl_torch.examples.libero.utils.schedules import _scheduled_residual_scale, _scheduled_xi
from serl_torch.examples.libero.utils.serialization import _to_jsonable
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

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("libero_train_residual_sac")
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
        normalizer = load_normalizer(task_key, stats_dir=norm_cfg.get("stats_dir", None))
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)

    openpi_client = OpenPIChunkClient(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )

    image_keys = resolve_image_keys(cfg)
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    control_indices = resolve_control_indices_from_cfg(cfg)
    action_dim = int(len(control_indices))
    chunk_horizon = int(cfg.residual.chunk_horizon)
    residual_xi = float(cfg.residual.get("xi", 1.0))
    residual_limits = build_residual_limits(
        control_indices,
        arm_limit=float(cfg.residual.arm_delta_limit),
        gripper_limit=float(cfg.residual.gripper_delta_limit),
    )
    logger.info(
        "Residual config: image_keys=%s action_dim=%s action_indices=%s chunk_horizon=%s xi=%.4f",
        list(image_keys),
        action_dim,
        control_indices.tolist(),
        chunk_horizon,
        residual_xi,
    )

    offline_enabled = bool(cfg.offline.enabled)
    offline_ratio = float(cfg.offline.ratio)
    if not (0.0 <= offline_ratio <= 1.0):
        raise ValueError(f"offline.ratio must be in [0,1], got {offline_ratio}")
    symmetric_replay = bool(cfg.offline.get("symmetric_replay", False))
    async_cfg = cfg.training.get("async", None)
    async_enabled = bool(async_cfg.get("enabled", False)) if async_cfg is not None else False
    async_update_frequency = int(async_cfg.get("update_frequency", 1)) if async_cfg is not None else 1
    async_idle_sleep_sec = float(async_cfg.get("idle_sleep_sec", 0.002)) if async_cfg is not None else 0.002
    replay_prefetch_cfg = cfg.training.get("replay_prefetch", None)
    replay_prefetch_enabled = (
        bool(replay_prefetch_cfg.get("enabled", True)) if replay_prefetch_cfg is not None else True
    )
    replay_prefetch_queue_size = (
        int(replay_prefetch_cfg.get("queue_size", 2)) if replay_prefetch_cfg is not None else 2
    )
    replay_prefetch_pin_memory = (
        bool(replay_prefetch_cfg.get("pin_memory", True)) if replay_prefetch_cfg is not None else True
    )
    replay_prefetch_to_device = (
        bool(replay_prefetch_cfg.get("to_device", True)) if replay_prefetch_cfg is not None else True
    )
    profiling_cfg = cfg.training.get("profiling", None)
    profiling_enabled = bool(profiling_cfg.get("enabled", False)) if profiling_cfg is not None else False
    profiling_window_size = int(profiling_cfg.get("window_size", 2048)) if profiling_cfg is not None else 2048
    profiling_log_period_steps = (
        int(profiling_cfg.get("log_period_steps", 500)) if profiling_cfg is not None else 500
    )
    profiling_log_file = (
        str(profiling_cfg.get("log_file", "profiling_logs.jsonl"))
        if profiling_cfg is not None
        else "profiling_logs.jsonl"
    )
    if async_enabled and any((not bool(phase.get("train", True))) for phase in cfg.training.phases):
        logger.warning(
            "Detected non-train phase in training.phases; disable async mode to preserve phase semantics."
        )
        async_enabled = False
    logger.info(
        "Async collection-learning: enabled=%s update_frequency=%s idle_sleep_sec=%.4f",
        async_enabled,
        async_update_frequency,
        async_idle_sleep_sec,
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

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    agent = None
    learner_agent = None
    async_learner: Optional[_AsyncLearner] = None
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
    bootstrap_stats: Dict[str, Any] = {"enabled": 0, "inserted": 0}
    warmstart_info: Dict[str, Any] = {"enabled": 0, "steps": 0}

    checkpoint_dir = Path(str(cfg.training.checkpoint_dir))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async_eval_proc: Optional[subprocess.Popen] = None
    async_eval_log_fp: Optional[IO[str]] = None
    async_eval_log_path: Optional[Path] = None
    async_eval_summary_path: Optional[Path] = None
    async_eval_watcher_return_code: Optional[int] = None
    async_eval_dead_reported = False
    profiler = _RuntimeProfiler(enabled=profiling_enabled, window_size=profiling_window_size)
    checkpoint_writer = _AsyncCheckpointWriter(profiler=profiler)
    profiling_logger: Optional[JsonlLogger] = None
    profiling_last_flush_step = -1
    (
        async_eval_proc,
        async_eval_log_fp,
        async_eval_log_path,
        async_eval_summary_path,
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
    step_metric_window = _new_tb_step_window()
    async_eval_tb_sync_state = _init_async_eval_tb_sync_state(async_eval_summary_path)
    obs_cache = LiberoObservationCache()

    sample_obs_raw = _profile_call(profiler, "env_reset", env.reset, seed=int(cfg.task.seed_base), episode_id=-1)
    sample_openpi_chunk, _ = openpi_client.infer_chunk(
        sample_obs_raw,
        env.current_instruction,
        obs_cache=obs_cache,
    )
    sample_base_chunk = select_action_chunk_window(sample_openpi_chunk, horizon=chunk_horizon)
    sample_obs = _build_residual_step_obs_profiled(
        profiler,
        sample_obs_raw,
        sample_base_chunk[0],
        image_keys=image_keys,
        stack_horizon=stack_horizon,
        normalizer=normalizer,
        obs_cache=obs_cache,
    )

    learner_agent = build_drq_agent(
        cfg,
        sample_obs=sample_obs,
        action_dim=action_dim,
        image_keys=image_keys,
    )
    agent = learner_agent
    replay_buffer = ReplayBuffer(
        observation_space=_obs_space_from_sample(sample_obs),
        action_space=action_space,
        capacity=int(cfg.replay.capacity),
    )
    if offline_enabled:
        offline_buffer = ReplayBuffer(
            observation_space=_obs_space_from_sample(sample_obs),
            action_space=action_space,
            capacity=int(cfg.offline.capacity),
        )
        bootstrap_stats = _bootstrap_offline_with_base_success(
            cfg,
            env=env,
            openpi_client=openpi_client,
            offline_buffer=offline_buffer,
            sample_obs_template=sample_obs,
            action_dim=action_dim,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            chunk_horizon=chunk_horizon,
            logger=logger,
            normalizer=normalizer,
            profiler=profiler,
        )
        offline_stats = _load_offline_residual_buffer(
            cfg,
            sample_obs_template=sample_obs,
            offline_buffer=offline_buffer,
            action_dim=action_dim,
            full_action_dim=action_dim,
            chunk_horizon=chunk_horizon,
            control_indices=control_indices,
            residual_limits=residual_limits,
            residual_xi=residual_xi,
            openpi_client=openpi_client,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            logger=logger,
            normalizer=normalizer,
            profiler=profiler,
        )
        logger.info(
            "offline bootstrap: success_episodes=%s collected=%s inserted=%s attempts=%s",
            bootstrap_stats.get("success_episodes", 0),
            bootstrap_stats.get("episodes_collected", 0),
            bootstrap_stats.get("inserted", 0),
            bootstrap_stats.get("attempts", 0),
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

    warmup_separate = bool(cfg.training.get("warmup_separate", False))
    warmup_base_episodes_cfg = int(cfg.training.get("warmup_base_episodes", 0))
    need_warmup_first = warmup_separate and warmup_base_episodes_cfg > 0

    if not need_warmup_first and async_enabled:
        agent = build_drq_agent(
            cfg,
            sample_obs=sample_obs,
            action_dim=action_dim,
            image_keys=image_keys,
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
                if len(replay_buffer) < int(cfg.training.training_starts):
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

    global_env_step = 0
    global_policy_step = 0
    episode_id = 0
    total_success = 0
    recent_successes: deque[int] = deque(maxlen=20)
    skipped_seeds = 0
    seed_cursor = int(cfg.task.seed_base)
    stopped_by_env_budget = False
    last_update_info: Dict[str, Any] = {}

    max_online_env_steps = int(cfg.training.get("max_online_env_steps", 0))
    warmup_base_episodes = int(cfg.training.get("warmup_base_episodes", 0))
    warmup_base_steps = int(cfg.training.get("warmup_base_steps", 0))
    warmup_separate = bool(cfg.training.get("warmup_separate", False))

    assert agent is not None
    assert learner_agent is not None
    assert replay_buffer is not None

    residual_env_step = 0  # Only incremented during residual training (after warmup when warmup_separate)

    try:
        # === Separate warmup phase: base-only data collection, NO policy updates ===
        if need_warmup_first:
            logger.info(
                "Warmup phase (separate): collecting %s base-only episodes, no actor/critic updates",
                warmup_base_episodes_cfg,
            )
            for warmup_ep_idx in range(warmup_base_episodes_cfg):
                seed = int(seed_cursor)
                seed_cursor += 1
                obs_cache.clear()
                obs_raw = _profile_call(
                    profiler, "env_reset", env.reset, seed=seed, episode_id=episode_id
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
                            openpi_chunk, horizon=chunk_horizon
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
                    next_obs_raw = obs_raw
                    for chunk_step in range(chunk_horizon):
                        if episode_steps >= max_episode_steps:
                            episode_done = True
                            break
                        obs_input = _build_residual_step_obs_profiled(
                            profiler,
                            next_obs_raw,
                            base_chunk[chunk_step],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            normalizer=normalizer,
                            obs_cache=obs_cache,
                        )
                        residual_step_action = np.zeros(
                            (action_dim,), dtype=np.float32
                        )
                        final_action = base_chunk[chunk_step].copy()
                        next_obs_raw, reward, env_done, _, info = _profile_call(
                            profiler, "env_step", env.step, final_action
                        )
                        episode_steps += 1
                        global_env_step += 1
                        episode_return += float(reward)
                        episode_success = bool(info["success"])
                        timeout = bool(episode_steps >= max_episode_steps)
                        done = bool(env_done or timeout)
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
                            )
                            mask = 1.0
                        else:
                            if next_chunk_future is not None:
                                (
                                    next_openpi_chunk,
                                    next_infer_info,
                                ) = next_chunk_future.result()
                            else:
                                next_openpi_chunk, next_infer_info = (
                                    openpi_client.infer_chunk(
                                        next_obs_raw,
                                        env.current_instruction,
                                        obs_cache=obs_cache,
                                    )
                                )
                            next_base_chunk = select_action_chunk_window(
                                next_openpi_chunk, horizon=chunk_horizon
                            )
                            next_obs_input = _build_residual_step_obs_profiled(
                                profiler,
                                next_obs_raw,
                                next_base_chunk[0],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                normalizer=normalizer,
                                obs_cache=obs_cache,
                            )
                            cached_base_chunk = next_base_chunk
                            cached_infer_info = next_infer_info
                            mask = 1.0
                        transition_payload = {
                            "observations": _clone_obs_dict(obs_input),
                            "actions": residual_step_action.astype(np.float32),
                            "next_observations": _clone_obs_dict(next_obs_input),
                            "rewards": np.float32(reward),
                            "masks": np.float32(mask),
                            "dones": bool(done),
                        }
                        replay_buffer.insert(transition_payload)
                        step_logger.write(
                            {
                                "global_env_step": int(global_env_step),
                                "global_policy_step": int(global_policy_step),
                                "episode_id": episode_id,
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
                                "infer_e2e_ms": (
                                    infer_info.get("e2e_ms")
                                    if chunk_step == 0
                                    else None
                                ),
                                "infer_policy_ms": (
                                    infer_info.get("policy_ms")
                                    if chunk_step == 0
                                    else None
                                ),
                                "infer_server_ms": (
                                    infer_info.get("server_ms")
                                    if chunk_step == 0
                                    else None
                                ),
                                "a_base": base_chunk[chunk_step].tolist(),
                                "a_res_policy": residual_step_action.tolist(),
                                "a_res": np.zeros_like(
                                    base_chunk[chunk_step], dtype=np.float32
                                ).tolist(),
                                "a_final": final_action.tolist(),
                                "residual_scale": 0.0,
                                "xi": 0.0,
                                "reward": float(reward),
                                "done": bool(done),
                                "success": bool(episode_success),
                            }
                        )
                        if done:
                            episode_done = True
                            break
                    obs_raw = next_obs_raw
                total_success += int(episode_success)
                recent_successes.append(int(episode_success))
                running_success_rate = float(total_success) / float(episode_id + 1)
                recent_success_rate = float(sum(recent_successes)) / float(
                    len(recent_successes)
                )
                episode_logger.write(
                    {
                        "episode_id": episode_id,
                        "phase": "warmup_base_only",
                        "seed": int(env.last_seed if env.last_seed is not None else seed),
                        "init_state_idx": (
                            int(env.current_init_state_idx)
                            if env.current_init_state_idx is not None
                            else None
                        ),
                        "success": bool(episode_success),
                        "episode_steps": int(episode_steps),
                        "episode_return": float(episode_return),
                        "global_env_step": int(global_env_step),
                        "global_policy_step": int(global_policy_step),
                        "running_success_rate": running_success_rate,
                        "recent_success_rate": recent_success_rate,
                        "is_warmup": True,
                    }
                )
                tb_writer.add_scalar("episode/success", int(episode_success), episode_id)
                tb_writer.add_scalar("episode/return", float(episode_return), episode_id)
                tb_writer.add_scalar("episode/length", int(episode_steps), episode_id)
                tb_writer.add_scalar(
                    "episode/running_success_rate", running_success_rate, episode_id
                )
                tb_writer.add_scalar(
                    "episode/recent_success_rate_20", recent_success_rate, episode_id
                )
                tb_writer.add_scalar(
                    "system/online_buffer_size", int(len(replay_buffer)), global_env_step
                )
                logger.info(
                    "warmup episode %s/%s success=%s steps=%s return=%.2f",
                    warmup_ep_idx + 1,
                    warmup_base_episodes_cfg,
                    episode_success,
                    episode_steps,
                    episode_return,
                )
                episode_id += 1
            logger.info(
                "Warmup complete. Episodes=%s total_success=%s buffer_size=%s. "
                "Starting residual training phase.",
                episode_id,
                total_success,
                len(replay_buffer),
            )
            # Start async learner / prefetcher after warmup
            if async_enabled:
                agent = build_drq_agent(
                    cfg,
                    sample_obs=sample_obs,
                    action_dim=action_dim,
                    image_keys=image_keys,
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
                        if len(replay_buffer) < int(cfg.training.training_starts):
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

        for phase in cfg.training.phases:
            if warmup_separate and max_online_env_steps > 0:
                budget_exceeded = residual_env_step >= max_online_env_steps
            else:
                budget_exceeded = (
                    max_online_env_steps > 0
                    and global_env_step >= max_online_env_steps
                )
            if budget_exceeded:
                stopped_by_env_budget = True
                break
            phase_name = str(phase.name)
            phase_episodes = int(phase.episodes)
            phase_train = bool(phase.get("train", True))
            phase_residual_scale = float(phase.residual_scale)
            logger.info(
                "Start phase=%s episodes=%s train=%s residual_scale=%.4f",
                phase_name,
                phase_episodes,
                phase_train,
                phase_residual_scale,
            )

            phase_episode_count = 0
            while phase_episode_count < phase_episodes:
                if warmup_separate and max_online_env_steps > 0:
                    inner_budget_exceeded = (
                        residual_env_step >= max_online_env_steps
                    )
                else:
                    inner_budget_exceeded = (
                        max_online_env_steps > 0
                        and global_env_step >= max_online_env_steps
                    )
                if inner_budget_exceeded:
                    stopped_by_env_budget = True
                    break

                seed = int(seed_cursor)
                seed_cursor += 1

                if bool(cfg.training.get("expert_check", False)):
                    passed, _ = env.expert_precheck(seed=seed, episode_id=episode_id)
                    if not passed:
                        skipped_seeds += 1
                        logger.warning("skip seed=%s in phase=%s: expert precheck failed", seed, phase_name)
                        continue

                obs_cache.clear()
                obs_raw = _profile_call(profiler, "env_reset", env.reset, seed=seed, episode_id=episode_id)
                max_episode_steps = int(env.step_limit)
                if cfg.training.max_env_steps_per_episode is not None:
                    max_episode_steps = min(max_episode_steps, int(cfg.training.max_env_steps_per_episode))

                episode_success = False
                episode_return = 0.0
                episode_steps = 0
                episode_done = False
                cached_base_chunk = None
                cached_infer_info = None

                probing_steps_target = sample_probing_steps(cfg.training, episode_horizon=max_episode_steps)
                if probing_steps_target > 0:
                    probing_remaining = int(min(probing_steps_target, max_episode_steps - episode_steps))
                    probe_future: Optional[Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]] = None
                    while probing_remaining > 0 and episode_steps < max_episode_steps:
                        if probe_future is not None:
                            probe_chunk, probe_info = probe_future.result()
                            probe_future = None
                        else:
                            probe_chunk, probe_info = openpi_client.infer_chunk(
                                obs_raw,
                                env.current_instruction,
                                obs_cache=obs_cache,
                            )
                        probe_base_chunk = select_action_chunk_window(probe_chunk, horizon=chunk_horizon)
                        for probe_step in range(chunk_horizon):
                            if probing_remaining <= 0 or episode_steps >= max_episode_steps:
                                break
                            base_action = probe_base_chunk[probe_step]
                            next_obs_raw, reward, env_done, _, info = _profile_call(
                                profiler,
                                "env_step",
                                env.step,
                                base_action,
                            )
                            episode_steps += 1
                            global_env_step += 1
                            if warmup_separate:
                                residual_env_step += 1
                            probing_remaining -= 1
                            episode_return += float(reward)
                            episode_success = bool(info["success"])
                            timeout = bool(episode_steps >= max_episode_steps)
                            if warmup_separate and max_online_env_steps > 0:
                                budget_exhausted = (
                                    residual_env_step >= max_online_env_steps
                                )
                            else:
                                budget_exhausted = bool(
                                    max_online_env_steps > 0
                                    and global_env_step >= max_online_env_steps
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
                                    "global_env_step": int(global_env_step),
                                    "global_policy_step": int(global_policy_step),
                                    "episode_id": episode_id,
                                    "phase": phase_name,
                                    "episode_step": episode_steps,
                                    "seed": int(env.last_seed if env.last_seed is not None else seed),
                                    "init_state_idx": (
                                        int(env.current_init_state_idx)
                                        if env.current_init_state_idx is not None
                                        else None
                                    ),
                                    "is_probing": True,
                                    "replan_point": bool(probe_step == 0),
                                    "chunk_step": int(probe_step),
                                    "chunk_horizon": int(chunk_horizon),
                                    "infer_e2e_ms": probe_info.get("e2e_ms") if probe_step == 0 else None,
                                    "infer_policy_ms": probe_info.get("policy_ms") if probe_step == 0 else None,
                                    "infer_server_ms": probe_info.get("server_ms") if probe_step == 0 else None,
                                    "a_base": base_action.tolist(),
                                    "a_res_policy": [0.0] * action_dim,
                                    "a_res": np.zeros_like(base_action, dtype=np.float32).tolist(),
                                    "a_final": base_action.tolist(),
                                    "residual_scale": 0.0,
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
                        base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
                    else:
                        base_chunk = cached_base_chunk
                        infer_info = cached_infer_info or {
                            "e2e_ms": None,
                            "policy_ms": None,
                            "server_ms": None,
                        }
                        cached_base_chunk = None
                        cached_infer_info = None

                    next_obs_raw = obs_raw
                    for chunk_step in range(chunk_horizon):
                        if episode_steps >= max_episode_steps:
                            episode_done = True
                            break

                        obs_input = _build_residual_step_obs_profiled(
                            profiler,
                            next_obs_raw,
                            base_chunk[chunk_step],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            normalizer=normalizer,
                            obs_cache=obs_cache,
                        )

                        schedule_step = (
                            residual_env_step if warmup_separate else global_policy_step
                        )
                        residual_scale_step = _scheduled_residual_scale(
                            cfg,
                            phase_scale=phase_residual_scale,
                            global_policy_step=schedule_step,
                        )
                        xi_step = _scheduled_xi(
                            cfg,
                            base_xi=residual_xi,
                            global_policy_step=schedule_step,
                        )

                        in_warmup_episode = bool(
                            not warmup_separate and episode_id < warmup_base_episodes
                        )
                        in_warmup_step = bool(
                            not warmup_separate
                            and warmup_base_steps > 0
                            and global_policy_step < warmup_base_steps
                        )
                        if phase_train and (in_warmup_episode or in_warmup_step):
                            residual_step_action = np.zeros((action_dim,), dtype=np.float32)
                        elif residual_scale_step <= 0.0:
                            residual_step_action = np.zeros((action_dim,), dtype=np.float32)
                        elif (not phase_train) or (global_policy_step < int(cfg.training.random_steps)):
                            residual_step_action = np.random.uniform(
                                -1.0, 1.0, size=(action_dim,)
                            ).astype(np.float32)
                            residual_step_action *= float(cfg.training.random_action_scale)
                        else:
                            if async_learner is not None:
                                residual_step_action = async_learner.sample_actor_action(obs_input, action_dim)
                            else:
                                sample_actions_start = time.perf_counter()
                                sampled = agent.sample_actions(obs_input, deterministic=False)
                                profiler.record_duration(
                                    "agent_sample_actions",
                                    (time.perf_counter() - sample_actions_start) * 1000.0,
                                )
                                residual_step_action = as_numpy_action(sampled, action_dim)

                        delta_action, final_action = compose_residual_action(
                            base_action=base_chunk[chunk_step],
                            residual_action=residual_step_action,
                            indices=control_indices,
                            limits=residual_limits,
                            residual_scale=residual_scale_step,
                            xi=xi_step,
                            clip_gripper=bool(cfg.residual.clip_gripper),
                        )

                        next_obs_raw, reward, env_done, _, info = _profile_call(
                            profiler,
                            "env_step",
                            env.step,
                            final_action,
                        )
                        episode_steps += 1
                        global_env_step += 1
                        if warmup_separate:
                            residual_env_step += 1
                        episode_return += float(reward)
                        episode_success = bool(info["success"])
                        timeout = bool(episode_steps >= max_episode_steps)
                        if warmup_separate and max_online_env_steps > 0:
                            budget_exhausted = (
                                residual_env_step >= max_online_env_steps
                            )
                        else:
                            budget_exhausted = bool(
                                max_online_env_steps > 0
                                and global_env_step >= max_online_env_steps
                            )
                        done = bool(env_done or timeout or budget_exhausted)
                        next_chunk_future: Optional[Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]] = None
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
                                "global_env_step": int(global_env_step),
                                "global_policy_step": int(global_policy_step),
                                "episode_id": episode_id,
                                "phase": phase_name,
                                "episode_step": episode_steps,
                                "seed": int(env.last_seed if env.last_seed is not None else seed),
                                "init_state_idx": (
                                    int(env.current_init_state_idx)
                                    if env.current_init_state_idx is not None
                                    else None
                                ),
                                "is_probing": False,
                                "replan_point": bool(chunk_step == 0),
                                "chunk_step": int(chunk_step),
                                "chunk_horizon": int(chunk_horizon),
                                "infer_e2e_ms": infer_info.get("e2e_ms") if chunk_step == 0 else None,
                                "infer_policy_ms": infer_info.get("policy_ms") if chunk_step == 0 else None,
                                "infer_server_ms": infer_info.get("server_ms") if chunk_step == 0 else None,
                                "a_base": base_chunk[chunk_step].tolist(),
                                "a_res_policy": residual_step_action.tolist(),
                                "a_res": delta_action.tolist(),
                                "a_final": final_action.tolist(),
                                "residual_scale": float(residual_scale_step),
                                "xi": float(xi_step),
                                "reward": float(reward),
                                "done": bool(done),
                                "success": bool(episode_success),
                            }
                        )

                        _append_tb_step_window(
                            step_metric_window,
                            reward=float(reward),
                            residual_scale=float(residual_scale_step),
                            xi=float(xi_step),
                            residual_action=residual_step_action,
                            delta_action=delta_action,
                            base_action=base_chunk[chunk_step],
                            final_action=final_action,
                            infer_info=infer_info,
                            replan_point=bool(chunk_step == 0),
                        )
                        if global_env_step % tb_step_period == 0:
                            _flush_tb_step_window(
                                tb_writer,
                                step_window=step_metric_window,
                                global_env_step=global_env_step,
                                control_indices=control_indices,
                                histogram=bool(global_env_step % tb_histogram_period == 0),
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
                            )
                            mask = 1.0
                        else:
                            if next_chunk_future is not None:
                                next_openpi_chunk, next_infer_info = next_chunk_future.result()
                            else:
                                next_openpi_chunk, next_infer_info = openpi_client.infer_chunk(
                                    next_obs_raw,
                                    env.current_instruction,
                                    obs_cache=obs_cache,
                                )
                            next_base_chunk = select_action_chunk_window(
                                next_openpi_chunk,
                                horizon=chunk_horizon,
                            )
                            next_obs_input = _build_residual_step_obs_profiled(
                                profiler,
                                next_obs_raw,
                                next_base_chunk[0],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                normalizer=normalizer,
                                obs_cache=obs_cache,
                            )
                            cached_base_chunk = next_base_chunk
                            cached_infer_info = next_infer_info
                            mask = 1.0

                        transition_payload = {
                            "observations": _clone_obs_dict(obs_input),
                            "actions": residual_step_action.astype(np.float32),
                            "next_observations": _clone_obs_dict(next_obs_input),
                            "rewards": np.float32(reward),
                            "masks": np.float32(mask),
                            "dones": bool(done),
                        }
                        if async_learner is not None:
                            with async_learner.replay_lock:
                                replay_buffer.insert(transition_payload)
                        elif sync_replay_lock is not None:
                            with sync_replay_lock:
                                replay_buffer.insert(transition_payload)
                        else:
                            replay_buffer.insert(transition_payload)

                        if async_learner is None:
                            if (
                                phase_train
                                and len(replay_buffer) >= int(cfg.training.training_starts)
                                and global_policy_step % int(cfg.training.update_every) == 0
                            ):
                                for _ in range(int(cfg.training.updates_per_step)):
                                    if sync_replay_prefetcher is not None:
                                        sampled_batch = sync_replay_prefetcher.get(timeout=async_idle_sleep_sec)
                                        if sampled_batch is None:
                                            continue
                                        batch, online_bs, offline_bs = sampled_batch
                                    else:
                                        replay_sample_start = time.perf_counter()
                                        sampled = _sample_mixed_batch(
                                            replay_buffer,
                                            offline_buffer if offline_enabled else None,
                                            batch_size=int(cfg.replay.batch_size),
                                            offline_ratio=offline_ratio,
                                            symmetric_replay=symmetric_replay,
                                        )
                                        profiler.record_duration(
                                            "replay_sample",
                                            (time.perf_counter() - replay_sample_start) * 1000.0,
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
                                        batch, online_bs, offline_bs = _consume_prepared_replay_batch(
                                            prepared,
                                            device=learner_agent.device,
                                            profiler=profiler,
                                        )
                                        profiler.record_duration(
                                            "replay_prepare",
                                            (time.perf_counter() - replay_prepare_start) * 1000.0,
                                        )
                                    update_start = time.perf_counter()
                                    learner_agent, last_update_info = learner_agent.update_high_utd(
                                        batch,
                                        utd_ratio=int(cfg.sac.utd_ratio),
                                    )
                                    profiler.record_duration(
                                        "agent_update_high_utd",
                                        (time.perf_counter() - update_start) * 1000.0,
                                    )
                                    last_update_info["online_batch_size"] = int(online_bs)
                                    last_update_info["offline_batch_size"] = int(offline_bs)
                                    last_update_info["offline_fraction"] = float(
                                        offline_bs / max(1, online_bs + offline_bs)
                                    )
                                agent = learner_agent
                        else:
                            last_update_info = async_learner.get_last_update_info()

                        if global_env_step % tb_step_period == 0 and last_update_info:
                            _log_update_metrics(tb_writer, last_update_info, global_env_step)
                            tb_writer.add_scalar(
                                "system/online_buffer_size",
                                int(len(replay_buffer)),
                                global_env_step,
                            )
                            if offline_buffer is not None:
                                tb_writer.add_scalar(
                                    "system/offline_buffer_size",
                                    int(len(offline_buffer)),
                                    global_env_step,
                                )
                            tb_writer.add_scalar(
                                "system/global_policy_step",
                                int(global_policy_step),
                                global_env_step,
                            )
                            if warmup_separate:
                                tb_writer.add_scalar(
                                    "system/residual_env_step",
                                    int(residual_env_step),
                                    global_env_step,
                                )
                            if async_learner is not None:
                                tb_writer.add_scalar(
                                    "system/learner_update_steps",
                                    int(async_learner.get_update_steps()),
                                    global_env_step,
                                )
                                tb_writer.add_scalar(
                                    "system/replay_prefetch_queue_size",
                                    int(async_learner.get_prefetch_queue_size()),
                                    global_env_step,
                                )
                            elif sync_replay_prefetcher is not None:
                                tb_writer.add_scalar(
                                    "system/replay_prefetch_queue_size",
                                    int(sync_replay_prefetcher.get_queue_size()),
                                    global_env_step,
                                )

                        global_policy_step += 1

                        if (
                            profiling_enabled
                            and profiling_log_period_steps > 0
                            and global_env_step > 0
                            and (global_env_step - profiling_last_flush_step) >= profiling_log_period_steps
                        ):
                            _emit_profiling_snapshot(
                                profiler,
                                profile_logger=profiling_logger,
                                tb_writer=tb_writer,
                                logger=logger,
                                global_env_step=global_env_step,
                                global_policy_step=global_policy_step,
                                episode_id=episode_id,
                                learner_update_steps=(
                                    int(async_learner.get_update_steps()) if async_learner is not None else 0
                                ),
                                replay_prefetch_queue_size=(
                                    int(async_learner.get_prefetch_queue_size())
                                    if async_learner is not None
                                    else int(sync_replay_prefetcher.get_queue_size())
                                    if sync_replay_prefetcher is not None
                                    else 0
                                ),
                            )
                            profiling_last_flush_step = int(global_env_step)

                        if (
                            phase_train
                            and int(cfg.training.checkpoint_period) > 0
                            and global_policy_step % int(cfg.training.checkpoint_period) == 0
                        ):
                            if async_learner is not None:
                                async_learner.save_checkpoint(
                                    str(checkpoint_dir),
                                    step=global_policy_step,
                                    keep=int(cfg.training.keep_checkpoints),
                                )
                            else:
                                checkpoint_payload = _snapshot_agent_checkpoint_payload(
                                    learner_agent,
                                    step=global_policy_step,
                                )
                                if checkpoint_writer is not None:
                                    checkpoint_writer.submit(
                                        _CheckpointTask(
                                            checkpoint_dir=str(checkpoint_dir),
                                            payload=checkpoint_payload,
                                            step=int(global_policy_step),
                                            keep=int(cfg.training.keep_checkpoints),
                                        )
                                    )
                                else:
                                    _write_checkpoint_payload(
                                        profiler,
                                        str(checkpoint_dir),
                                        checkpoint_payload,
                                        step=global_policy_step,
                                        keep=int(cfg.training.keep_checkpoints),
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
                    global_env_step=global_env_step,
                    control_indices=control_indices,
                    histogram=bool(global_env_step > 0 and global_env_step % tb_histogram_period == 0),
                )
                total_success += int(episode_success)
                recent_successes.append(int(episode_success))
                running_success_rate = float(total_success) / float(episode_id + 1)
                recent_success_rate = float(sum(recent_successes)) / float(len(recent_successes))

                episode_logger.write(
                    {
                        "episode_id": episode_id,
                        "phase": phase_name,
                        "seed": int(env.last_seed if env.last_seed is not None else seed),
                        "init_state_idx": (
                            int(env.current_init_state_idx)
                            if env.current_init_state_idx is not None
                            else None
                        ),
                        "success": bool(episode_success),
                        "episode_steps": int(episode_steps),
                        "episode_return": float(episode_return),
                        "global_env_step": int(global_env_step),
                        "global_policy_step": int(global_policy_step),
                        "running_success_rate": running_success_rate,
                        "recent_success_rate": recent_success_rate,
                    }
                )
                tb_writer.add_scalar("episode/success", int(episode_success), episode_id)
                tb_writer.add_scalar("episode/return", float(episode_return), episode_id)
                tb_writer.add_scalar("episode/length", int(episode_steps), episode_id)
                tb_writer.add_scalar("episode/running_success_rate", running_success_rate, episode_id)
                tb_writer.add_scalar("episode/recent_success_rate_20", recent_success_rate, episode_id)
                tb_writer.add_scalar("system/online_buffer_size", int(len(replay_buffer)), global_env_step)
                if offline_buffer is not None:
                    tb_writer.add_scalar("system/offline_buffer_size", int(len(offline_buffer)), global_env_step)
                tb_writer.add_scalar("system/global_policy_step", int(global_policy_step), global_env_step)
                if async_learner is not None:
                    tb_writer.add_scalar(
                        "system/learner_update_steps",
                        int(async_learner.get_update_steps()),
                        global_env_step,
                    )
                    tb_writer.add_scalar(
                        "system/replay_prefetch_queue_size",
                        int(async_learner.get_prefetch_queue_size()),
                        global_env_step,
                    )
                elif sync_replay_prefetcher is not None:
                    tb_writer.add_scalar(
                        "system/replay_prefetch_queue_size",
                        int(sync_replay_prefetcher.get_queue_size()),
                        global_env_step,
                    )

                logger.info(
                    "phase=%s episode=%s success=%s steps=%s return=%.2f success_rate=%.3f recent=%.3f",
                    phase_name,
                    episode_id,
                    episode_success,
                    episode_steps,
                    episode_return,
                    running_success_rate,
                    recent_success_rate,
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

                episode_id += 1
                phase_episode_count += 1

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
            global_env_step=global_env_step,
            global_policy_step=global_policy_step,
            episode_id=episode_id,
            learner_update_steps=int(async_learner.get_update_steps()) if async_learner is not None else 0,
            replay_prefetch_queue_size=(
                int(async_learner.get_prefetch_queue_size())
                if async_learner is not None
                else int(sync_replay_prefetcher.get_queue_size())
                if sync_replay_prefetcher is not None
                else 0
            ),
        )

        summary = {
            "episodes": int(episode_id),
            "global_env_step": int(global_env_step),
            "global_policy_step": int(global_policy_step),
            "total_success": int(total_success),
            "success_rate": float(total_success / max(1, int(episode_id))),
            "skipped_seeds": int(skipped_seeds),
            "seed_start": int(cfg.task.seed_base),
            "seed_next": int(seed_cursor),
            "stopped_by_env_budget": bool(stopped_by_env_budget),
            "max_online_env_steps": int(max_online_env_steps),
            "warmup_separate": bool(warmup_separate),
            "warmup_base_episodes": int(warmup_base_episodes_cfg),
            "residual_env_step": int(residual_env_step) if warmup_separate else None,
            "replay_size": int(len(replay_buffer) if replay_buffer is not None else 0),
            "offline_enabled": bool(offline_enabled),
            "offline_ratio": float(offline_ratio),
            "offline_symmetric_replay": bool(symmetric_replay),
            "offline_buffer_size": int(len(offline_buffer) if offline_buffer is not None else 0),
            "offline_stats": offline_stats,
            "bootstrap_stats": bootstrap_stats,
            "critic_pretrain": _to_jsonable(warmstart_info),
            "checkpoint_dir": str(checkpoint_dir),
            "last_update_info": _to_jsonable(last_update_info),
            "async_enabled": bool(async_enabled),
            "async_update_frequency": int(async_update_frequency),
            "learner_update_steps": int(async_learner.get_update_steps() if async_learner is not None else 0),
            "replay_prefetch_enabled": bool(replay_prefetch_enabled),
            "replay_prefetch_queue_size": int(replay_prefetch_queue_size),
            "replay_prefetch_pin_memory": bool(replay_prefetch_pin_memory),
            "replay_prefetch_to_device": bool(replay_prefetch_to_device),
            "profiling": {
                "enabled": bool(profiling_enabled),
                "window_size": int(profiling_window_size),
                "log_period_steps": int(profiling_log_period_steps),
                "log_file": str(run_dir / profiling_log_file) if profiling_enabled else None,
                "snapshot": (
                    _to_jsonable(final_profiling_payload.get("metrics", {}))
                    if final_profiling_payload is not None
                    else {}
                ),
            },
            "async_eval": {
                "enabled": bool(cfg.training.get("async_eval", {}).get("enabled", False)),
                "watcher_started": bool(async_eval_proc is not None),
                "watcher_log_path": str(async_eval_log_path) if async_eval_log_path is not None else None,
                "summary_jsonl_path": (
                    str(async_eval_summary_path) if async_eval_summary_path is not None else None
                ),
                "watcher_return_code": (
                    int(async_eval_proc.returncode)
                    if async_eval_proc is not None and async_eval_proc.returncode is not None
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
