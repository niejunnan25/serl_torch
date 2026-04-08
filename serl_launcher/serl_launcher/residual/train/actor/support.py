from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from serl_launcher.residual.train.bindings import ResidualRuntimeBindings
from serl_launcher.training.async_runtime.agentlace import _AsyncLearner
from serl_launcher.training.async_runtime.agentlace import _MixedBatchPrefetcher
from serl_launcher.training.async_runtime.agentlace import _ProcessAsyncLearner
from serl_launcher.training.async_runtime.agentlace import _sample_mixed_batch
from serl_launcher.training.async_runtime.bridge import advance_async_target_update_calls
from serl_launcher.training.async_runtime.bridge import create_agentlace_async_learner
from serl_launcher.training.async_runtime.bridge import maybe_send_agentlace_timer_stats
from serl_launcher.training.async_runtime.bridge import maybe_wait_for_async_learner_budget
from serl_launcher.training.async_runtime.bridge import sync_async_bounded_lag_baseline_from_learner
from serl_launcher.training.checkpoint import _CheckpointTask
from serl_launcher.training.checkpoint import _write_checkpoint_payload
from serl_launcher.residual.train.schedules import _scheduled_epsilon_gating_probability


_CORE_CONTEXT_FIELDS = {
    "cfg",
    "run_dir",
    "logger",
    "bindings",
    "async_eval_watcher_path",
    "values",
}


@dataclass
class ActorRuntimeContext:
    cfg: Any
    run_dir: Path
    logger: Any
    bindings: ResidualRuntimeBindings
    async_eval_watcher_path: Path
    values: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _CORE_CONTEXT_FIELDS:
            object.__setattr__(self, name, value)
            return
        self.values[name] = value

    def update(self, **kwargs: Any) -> None:
        self.values.update(kwargs)


@dataclass
class ActorLoopState:
    train_env_step: int
    decision_step: int
    train_episode_id: int
    warmup_episode_id: int
    init_episode_idx: int
    eval_trigger_count: int
    train_total_success: int
    warmup_total_success: int
    skipped_seeds: int
    seed_cursor: int
    stopped_by_env_budget: bool
    last_update_info: Dict[str, Any] = field(default_factory=dict)
    saved_checkpoint_steps: set[int] = field(default_factory=set)
    train_recent_successes: deque[int] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    warmup_recent_successes: deque[int] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    train_progress: Optional[Any] = None
    warmup_progress: Optional[Any] = None
    phase_progress: Optional[Any] = None
    train_progress_last_step: int = 0


def initialize_actor_loop_state(ctx: ActorRuntimeContext) -> ActorLoopState:
    warmup_recent_successes = deque(
        [int(v) for v in ctx.online_prefill_stats.get("recent_episode_successes", [])],
        maxlen=20,
    )
    state = ActorLoopState(
        train_env_step=0,
        decision_step=0,
        train_episode_id=0,
        warmup_episode_id=int(ctx.online_prefill_loaded_episodes),
        init_episode_idx=int(ctx.online_prefill_loaded_episodes),
        eval_trigger_count=0,
        train_total_success=0,
        warmup_total_success=int(ctx.online_prefill_stats.get("success_episodes", 0)),
        skipped_seeds=0,
        seed_cursor=int(ctx.cfg.task.seed_base) + int(ctx.online_prefill_loaded_episodes),
        stopped_by_env_budget=False,
        warmup_recent_successes=warmup_recent_successes,
    )
    if int(ctx.max_train_env_steps) > 0:
        state.train_progress = new_progress(
            ctx,
            desc="train_env_step",
            total=int(ctx.max_train_env_steps),
            position=0,
            leave=True,
        )
    return state


def build_policy_input(
    ctx: ActorRuntimeContext,
    obs_raw: Dict[str, Any],
    prompt: str,
    *,
    cache_key: Optional[Any] = None,
) -> Any:
    return ctx.bindings.build_policy_input(obs_raw, prompt, cache_key=cache_key)


def runtime_image_keys(ctx: ActorRuntimeContext) -> tuple[str, ...]:
    return tuple(ctx.bindings.image_keys)


def runtime_normalizer(ctx: ActorRuntimeContext):
    return ctx.bindings.normalizer


def runtime_obs_cache(ctx: ActorRuntimeContext):
    return ctx.bindings.obs_cache


def runtime_task_key(ctx: ActorRuntimeContext) -> str:
    return str(ctx.bindings.task_key)


def runtime_data_config(ctx: ActorRuntimeContext) -> Any:
    return ctx.bindings.data_config


def clear_obs_cache(ctx: ActorRuntimeContext) -> None:
    runtime_obs_cache(ctx).clear()


def build_step_core(
    ctx: ActorRuntimeContext,
    obs_raw: Dict[str, Any],
    *,
    cache_key: Optional[Any] = None,
) -> dict[str, Any]:
    return ctx.bindings.build_step_core(obs_raw, cache_key=cache_key)


def build_step_obs_profiled(
    ctx: ActorRuntimeContext,
    obs_raw: Dict[str, Any],
    base_action: Any,
    *,
    cache_key: Optional[Any] = None,
    action_dim: Optional[int] = None,
    base_action_chunk: Any = None,
    alpha: Optional[float] = None,
) -> dict[str, Any]:
    return ctx.bindings.build_step_obs_profiled(
        ctx.profiler,
        obs_raw,
        base_action,
        stack_horizon=int(ctx.stack_horizon),
        cache_key=cache_key,
        action_dim=action_dim,
        base_action_chunk=base_action_chunk,
        alpha=alpha,
        state_mode=str(ctx.obs_state_mode),
    )


def resolve_train_gate(
    ctx: ActorRuntimeContext,
    *,
    phase_train_flag: bool,
    alpha_value: float,
    env_step_value: int,
    decision_step_value: int,
) -> Tuple[float, bool]:
    if (not phase_train_flag) or (not ctx.epsilon_gating_enabled):
        return 1.0, bool(alpha_value > 0.0)

    schedule_step = (
        int(env_step_value)
        if ctx.epsilon_gating_clock == "env_step"
        else int(decision_step_value)
    )
    gate_prob = _scheduled_epsilon_gating_probability(
        ctx.cfg, schedule_step=schedule_step
    )
    if alpha_value <= 0.0:
        return float(gate_prob), False
    gate_on = bool(np.random.random() < float(gate_prob))
    return float(gate_prob), gate_on


def normalize_step_action(ctx: ActorRuntimeContext, action: np.ndarray) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    normalizer = runtime_normalizer(ctx)
    if normalizer is None:
        return action_arr.astype(np.float32)
    return np.asarray(normalizer.normalize_action(action_arr), dtype=np.float32)


def build_chunk_step_record(
    ctx: ActorRuntimeContext,
    current_obs_raw: Dict[str, Any],
    *,
    base_action: np.ndarray,
    final_action: np.ndarray,
    alpha_obs: float,
    episode_id: int,
    episode_step: int,
    done: bool,
) -> Dict[str, Any]:
    obs_core = build_step_core(ctx, current_obs_raw)
    base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    final_action_arr = np.asarray(final_action, dtype=np.float32).reshape(-1)
    return {
        "obs_core": obs_core,
        "base_action": base_action_arr,
        "base_action_norm": normalize_step_action(ctx, base_action_arr),
        "actions": final_action_arr,
        "rewards": 0.0,
        "dones": bool(done),
        "alpha": float(alpha_obs),
        "episode_id": int(episode_id),
        "episode_step": int(episode_step),
    }


def replay_progress_size(buffer: Any) -> int:
    return int(getattr(buffer, "num_steps", len(buffer)))


def flush_external_agentlace_actor(ctx: ActorRuntimeContext) -> None:
    if ctx.external_agentlace_actor_mode and ctx.async_learner is not None:
        ctx.async_learner.flush()


def new_progress(
    ctx: ActorRuntimeContext,
    *,
    desc: str,
    total: Optional[int],
    position: int,
    leave: bool,
) -> Optional[Any]:
    if not ctx.progress_enabled:
        return None
    return tqdm(
        total=total,
        desc=desc,
        dynamic_ncols=True,
        mininterval=ctx.progress_mininterval_sec,
        position=position,
        leave=leave,
    )


def update_train_progress(
    ctx: ActorRuntimeContext,
    state: ActorLoopState,
    *,
    force_postfix: bool = False,
) -> None:
    if state.train_progress is None:
        return
    delta = int(state.train_env_step) - int(state.train_progress_last_step)
    if delta > 0:
        state.train_progress.update(delta)
        state.train_progress_last_step = int(state.train_env_step)
    if force_postfix:
        completed_eval = int(ctx.async_eval_tb_sync_state.get("processed_lines", 0))
        pending_eval = max(0, int(state.eval_trigger_count) - int(completed_eval))
        state.train_progress.set_postfix(
            {
                "episode": int(state.train_episode_id),
                "eval_q": int(pending_eval),
            },
            refresh=False,
        )


def save_checkpoint_at_step(
    ctx: ActorRuntimeContext,
    state: ActorLoopState,
    checkpoint_step: int,
) -> Path:
    checkpoint_path = ctx.checkpoint_dir / f"checkpoint_{int(checkpoint_step)}.pt"
    if int(checkpoint_step) in state.saved_checkpoint_steps:
        return checkpoint_path
    state.saved_checkpoint_steps.add(int(checkpoint_step))
    if ctx.async_learner is not None:
        ctx.async_learner.save_checkpoint(
            str(ctx.checkpoint_dir),
            step=int(checkpoint_step),
            keep=ctx.checkpoint_keep,
        )
    else:
        checkpoint_payload = ctx.algorithm.snapshot_checkpoint_payload(
            ctx.learner_agent,
            step=int(checkpoint_step),
        )
        if ctx.checkpoint_writer is not None:
            ctx.checkpoint_writer.submit(
                _CheckpointTask(
                    checkpoint_dir=str(ctx.checkpoint_dir),
                    payload=checkpoint_payload,
                    step=int(checkpoint_step),
                    keep=ctx.checkpoint_keep,
                )
            )
        else:
            _write_checkpoint_payload(
                ctx.profiler,
                str(ctx.checkpoint_dir),
                checkpoint_payload,
                step=int(checkpoint_step),
                keep=ctx.checkpoint_keep,
            )
    return checkpoint_path


def send_agentlace_timer_stats(
    ctx: ActorRuntimeContext,
    state: ActorLoopState,
    *,
    train_episode_id_value: Optional[int] = None,
    force: bool = False,
) -> None:
    maybe_send_agentlace_timer_stats(
        config=ctx.agentlace_bridge_config,
        state=ctx.agentlace_bridge_state,
        profiler=ctx.profiler,
        replay_buffer=ctx.replay_buffer,
        offline_buffer=ctx.offline_buffer,
        async_learner=ctx.async_learner,
        sync_replay_prefetcher=ctx.sync_replay_prefetcher,
        train_env_step=int(state.train_env_step),
        decision_step=int(state.decision_step),
        train_episode_id=int(
            state.train_episode_id
            if train_episode_id_value is None
            else train_episode_id_value
        ),
        force=bool(force),
    )


def advance_async_update_calls(
    ctx: ActorRuntimeContext,
    *,
    phase_train_flag: bool,
    train_step_before: int,
    train_step_after: int,
    replay_size_before: int,
    replay_size_after: int,
) -> int:
    return advance_async_target_update_calls(
        config=ctx.agentlace_bridge_config,
        state=ctx.agentlace_bridge_state,
        async_learner=ctx.async_learner,
        phase_train_flag=bool(phase_train_flag),
        train_step_before=int(train_step_before),
        train_step_after=int(train_step_after),
        replay_size_before=int(replay_size_before),
        replay_size_after=int(replay_size_after),
    )


def wait_for_async_learner_budget(
    ctx: ActorRuntimeContext,
    state: ActorLoopState,
) -> None:
    maybe_wait_for_async_learner_budget(
        config=ctx.agentlace_bridge_config,
        state=ctx.agentlace_bridge_state,
        async_learner=ctx.async_learner,
        logger=ctx.logger,
        train_env_step=int(state.train_env_step),
        decision_step=int(state.decision_step),
    )


def sync_async_bounded_lag_baseline(ctx: ActorRuntimeContext) -> None:
    sync_async_bounded_lag_baseline_from_learner(
        config=ctx.agentlace_bridge_config,
        state=ctx.agentlace_bridge_state,
        async_learner=ctx.async_learner,
        logger=ctx.logger,
    )


def ensure_training_runtime_started(ctx: ActorRuntimeContext) -> None:
    if ctx.async_learner is not None:
        return

    if ctx.async_enabled:
        if ctx.async_backend == "agentlace":
            agentlace_replay_buffer = ctx.replay_buffer
            ctx.async_learner = create_agentlace_async_learner(
                config=ctx.agentlace_bridge_config,
                algorithm=ctx.algorithm,
                actor_agent=ctx.agent,
                replay_buffer=agentlace_replay_buffer,
                offline_buffer=(
                    ctx.offline_buffer
                    if ctx.offline_enabled and ctx.manage_learner_state_locally
                    else None
                ),
                cfg_dict=ctx.resolved_cfg_dict,
                sample_obs=ctx.sample_obs,
                action_dim=ctx.agent_action_dim,
                critic_action_dim=ctx.critic_action_dim,
                image_keys=runtime_image_keys(ctx),
                action_transform=ctx.action_transform,
            )
            ctx.replay_buffer = ctx.async_learner.replay_proxy
            return

        if ctx.async_backend == "process":
            ctx.async_learner = _ProcessAsyncLearner(
                algorithm=ctx.algorithm,
                actor_agent=ctx.agent,
                online_buffer=ctx.replay_buffer,
                offline_buffer=ctx.offline_buffer if ctx.offline_enabled else None,
                batch_size=int(ctx.cfg.replay.batch_size),
                offline_ratio=ctx.offline_ratio,
                symmetric_replay=ctx.symmetric_replay,
                training_starts=int(ctx.cfg.training.training_starts),
                update_frequency=ctx.async_update_frequency,
                idle_sleep_sec=ctx.async_idle_sleep_sec,
                cfg_dict=ctx.resolved_cfg_dict,
                sample_obs=ctx.sample_obs,
                action_dim=ctx.agent_action_dim,
                critic_action_dim=ctx.critic_action_dim,
                image_keys=runtime_image_keys(ctx),
                action_transform=ctx.action_transform,
                actor_device=ctx.async_actor_device,
                learner_device=ctx.async_learner_device,
                batch_queue_size=ctx.async_batch_queue_size,
            )
            ctx.async_learner.start()
            return

        ctx.agent = ctx.algorithm.build_actor_agent(
            ctx.cfg,
            sample_obs=ctx.sample_obs,
            action_dim=ctx.agent_action_dim,
            image_keys=runtime_image_keys(ctx),
            critic_action_dim=ctx.critic_action_dim,
            action_transform=ctx.action_transform,
            device=ctx.async_actor_device,
        )
        ctx.algorithm.sync_modules(ctx.agent, ctx.learner_agent)
        ctx.async_learner = _AsyncLearner(
            algorithm=ctx.algorithm,
            learner_agent=ctx.learner_agent,
            actor_agent=ctx.agent,
            online_buffer=ctx.replay_buffer,
            offline_buffer=ctx.offline_buffer if ctx.offline_enabled else None,
            batch_size=int(ctx.cfg.replay.batch_size),
            offline_ratio=ctx.offline_ratio,
            symmetric_replay=ctx.symmetric_replay,
            training_starts=int(ctx.cfg.training.training_starts),
            utd_ratio=int(ctx.cfg.sac.utd_ratio),
            update_frequency=ctx.async_update_frequency,
            idle_sleep_sec=ctx.async_idle_sleep_sec,
            replay_prefetch_enabled=ctx.replay_prefetch_enabled,
            replay_prefetch_queue_size=ctx.replay_prefetch_queue_size,
            replay_prefetch_pin_memory=ctx.replay_prefetch_pin_memory,
            replay_prefetch_to_device=ctx.replay_prefetch_to_device,
            checkpoint_writer=ctx.checkpoint_writer,
            profiler=ctx.profiler,
        )
        ctx.async_learner.start()
        return

    if (not ctx.replay_prefetch_enabled) or (ctx.sync_replay_prefetcher is not None):
        return

    ctx.sync_replay_lock = threading.Lock()

    def _sample_sync_prefetch_batch() -> Optional[Tuple[Dict[str, Any], int, int]]:
        assert ctx.replay_buffer is not None
        assert ctx.sync_replay_lock is not None
        with ctx.sync_replay_lock:
            if replay_progress_size(ctx.replay_buffer) < int(ctx.cfg.training.training_starts):
                return None
            return _sample_mixed_batch(
                ctx.replay_buffer,
                ctx.offline_buffer if ctx.offline_enabled else None,
                batch_size=int(ctx.cfg.replay.batch_size),
                offline_ratio=ctx.offline_ratio,
                symmetric_replay=ctx.symmetric_replay,
            )

    ctx.sync_replay_prefetcher = _MixedBatchPrefetcher(
        sample_fn=_sample_sync_prefetch_batch,
        queue_size=ctx.replay_prefetch_queue_size,
        idle_sleep_sec=ctx.async_idle_sleep_sec,
        device=ctx.learner_agent.device,
        pin_memory=ctx.replay_prefetch_pin_memory,
        to_device=ctx.replay_prefetch_to_device,
        profiler=ctx.profiler,
    )
    ctx.sync_replay_prefetcher.start()
