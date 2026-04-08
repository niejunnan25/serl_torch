from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from serl_launcher.residual.runtime.async_learning import _sample_mixed_batch
from serl_launcher.residual.runtime.profiling import _emit_profiling_snapshot
from serl_launcher.residual.runtime.replay_batch import _consume_prepared_replay_batch
from serl_launcher.residual.runtime.replay_batch import _prepare_replay_batch
from serl_launcher.residual.runtime.tb_metrics import _flush_tb_step_window
from serl_launcher.residual.runtime.tb_metrics import _log_update_metrics
from serl_launcher.residual.runtime.train_loop_utils import _insert_online_transition


@dataclass(frozen=True)
class EpisodeSpec:
    phase_name: str
    phase_train: bool
    phase_episode_idx: int
    train_episode_id: Optional[int]
    seed: int
    init_episode_idx: int
    max_episode_steps: int


@dataclass
class EpisodeState:
    obs_raw: Dict[str, Any]
    train_env_step: int
    decision_step: int
    last_update_info: Dict[str, Any]
    agent: Any
    learner_agent: Any
    async_learner: Optional[Any]
    replay_buffer: Any
    sync_replay_lock: Optional[Any]
    sync_replay_prefetcher: Optional[Any]
    profiling_last_flush_step: int
    episode_success: bool = False
    episode_return: float = 0.0
    episode_steps: int = 0
    episode_done: bool = False
    cached_base_chunk: Optional[np.ndarray] = None
    cached_infer_info: Optional[Dict[str, Optional[float]]] = None


@dataclass
class EpisodeResult:
    episode_success: bool
    episode_return: float
    episode_steps: int
    train_env_step: int
    decision_step: int
    last_update_info: Dict[str, Any]
    agent: Any
    learner_agent: Any
    async_learner: Optional[Any]
    replay_buffer: Any
    sync_replay_lock: Optional[Any]
    sync_replay_prefetcher: Optional[Any]
    profiling_last_flush_step: int


def insert_online_transitions(
    state: EpisodeState,
    transition_payloads: list[Dict[str, Any]],
    *,
    chunk_step_enabled: bool,
) -> None:
    if state.async_learner is not None:
        with state.async_learner.replay_lock:
            for payload in transition_payloads:
                _insert_online_transition(
                    state.replay_buffer,
                    payload,
                    chunk_step_enabled=chunk_step_enabled,
                )
        return
    if state.sync_replay_lock is not None:
        with state.sync_replay_lock:
            for payload in transition_payloads:
                _insert_online_transition(
                    state.replay_buffer,
                    payload,
                    chunk_step_enabled=chunk_step_enabled,
                )
        return
    for payload in transition_payloads:
        _insert_online_transition(
            state.replay_buffer,
            payload,
            chunk_step_enabled=chunk_step_enabled,
        )


def sample_replay_batch(
    ctx: Any,
    state: EpisodeState,
) -> Optional[tuple[Any, int, int]]:
    profiler = ctx.profiler
    if state.sync_replay_prefetcher is not None:
        sampled_batch = state.sync_replay_prefetcher.get(timeout=ctx.async_idle_sleep_sec)
        if sampled_batch is None:
            return None
        return sampled_batch
    replay_sample_start = time.perf_counter()
    sampled = _sample_mixed_batch(
        state.replay_buffer,
        ctx.offline_buffer if bool(ctx.offline_enabled) else None,
        batch_size=int(ctx.cfg.replay.batch_size),
        offline_ratio=float(ctx.offline_ratio),
        symmetric_replay=bool(ctx.symmetric_replay),
    )
    profiler.record_duration(
        "replay_sample",
        (time.perf_counter() - replay_sample_start) * 1000.0,
    )
    replay_prepare_start = time.perf_counter()
    prepared = _prepare_replay_batch(
        sampled,
        device=state.learner_agent.device,
        pin_memory=bool(ctx.replay_prefetch_pin_memory),
        to_device=bool(ctx.replay_prefetch_to_device),
        profiler=profiler,
        cuda_stream=None,
    )
    batch, online_bs, offline_bs = _consume_prepared_replay_batch(
        prepared,
        device=state.learner_agent.device,
        profiler=profiler,
    )
    profiler.record_duration(
        "replay_prepare",
        (time.perf_counter() - replay_prepare_start) * 1000.0,
    )
    return batch, online_bs, offline_bs


def run_sync_training_updates(
    ctx: Any,
    state: EpisodeState,
    *,
    num_updates: int,
) -> None:
    if num_updates <= 0:
        return
    algorithm = ctx.algorithm
    profiler = ctx.profiler
    for _ in range(int(num_updates)):
        sampled_batch = sample_replay_batch(ctx, state)
        if sampled_batch is None:
            continue
        batch, online_bs, offline_bs = sampled_batch
        update_start = time.perf_counter()
        state.learner_agent, state.last_update_info = algorithm.update_high_utd(
            state.learner_agent,
            batch,
            utd_ratio=int(ctx.cfg.sac.utd_ratio),
        )
        profiler.record_duration(
            "agent_update_high_utd",
            (time.perf_counter() - update_start) * 1000.0,
        )
        state.last_update_info["online_batch_size"] = int(online_bs)
        state.last_update_info["offline_batch_size"] = int(offline_bs)
        state.last_update_info["offline_fraction"] = float(
            offline_bs / max(1, online_bs + offline_bs)
        )
    state.agent = state.learner_agent


def log_training_system_metrics(
    ctx: Any,
    state: EpisodeState,
    *,
    current_decision_id: Optional[int],
) -> None:
    tb_writer = ctx.tb_writer
    train_env_step = int(state.train_env_step)
    _log_update_metrics(tb_writer, state.last_update_info, train_env_step)
    tb_writer.add_scalar(
        "system/online_buffer_size",
        int(len(state.replay_buffer)),
        train_env_step,
    )
    if ctx.offline_buffer is not None:
        tb_writer.add_scalar(
            "system/offline_buffer_size",
            int(len(ctx.offline_buffer)),
            train_env_step,
        )
    tb_writer.add_scalar(
        "system/decision_step",
        float(current_decision_id if current_decision_id is not None else 0),
        train_env_step,
    )
    if state.async_learner is not None:
        tb_writer.add_scalar(
            "system/learner_update_steps",
            int(state.async_learner.get_update_steps()),
            train_env_step,
        )
        tb_writer.add_scalar(
            "system/replay_prefetch_queue_size",
            int(state.async_learner.get_prefetch_queue_size()),
            train_env_step,
        )
        if bool(ctx.async_bounded_lag_enabled):
            bridge_state = ctx.agentlace_bridge_state
            tb_writer.add_scalar(
                "system/async_target_update_calls",
                float(bridge_state.target_update_calls),
                train_env_step,
            )
            tb_writer.add_scalar(
                "system/async_bounded_lag_tracked_env_steps",
                float(bridge_state.tracked_env_steps),
                train_env_step,
            )
            if ctx.async_bounded_lag_env_steps_per_update_call is not None:
                tb_writer.add_scalar(
                    "system/async_env_steps_per_update_call",
                    float(ctx.async_bounded_lag_env_steps_per_update_call),
                    train_env_step,
                )
            tb_writer.add_scalar(
                "system/async_required_update_steps",
                float(bridge_state.last_required_update_steps),
                train_env_step,
            )
            tb_writer.add_scalar(
                "system/async_update_lag_before_wait",
                float(bridge_state.last_lag_before_wait),
                train_env_step,
            )
            tb_writer.add_scalar(
                "system/async_update_lag_after_wait",
                float(bridge_state.last_lag_after_wait),
                train_env_step,
            )
            tb_writer.add_scalar(
                "system/async_wait_last_sec",
                float(bridge_state.last_wait_sec),
                train_env_step,
            )
            tb_writer.add_scalar(
                "system/async_wait_count",
                float(bridge_state.wait_count),
                train_env_step,
            )
            tb_writer.add_scalar(
                "system/async_wait_timeout_count",
                float(bridge_state.timeout_count),
                train_env_step,
            )
        return
    if state.sync_replay_prefetcher is not None:
        tb_writer.add_scalar(
            "system/replay_prefetch_queue_size",
            int(state.sync_replay_prefetcher.get_queue_size()),
            train_env_step,
        )


def maybe_emit_episode_profiling(
    ctx: Any,
    state: EpisodeState,
    *,
    timer_train_episode_id: int,
) -> None:
    if not bool(ctx.profiling_enabled):
        return
    profiling_log_period_steps = int(ctx.profiling_log_period_steps)
    if profiling_log_period_steps <= 0:
        return
    if state.train_env_step <= 0:
        return
    if (state.train_env_step - state.profiling_last_flush_step) < profiling_log_period_steps:
        return
    _emit_profiling_snapshot(
        ctx.profiler,
        profile_logger=ctx.profiling_logger,
        tb_writer=ctx.tb_writer,
        logger=ctx.logger,
        train_env_step=int(state.train_env_step),
        decision_step=int(state.decision_step),
        train_episode_id=int(timer_train_episode_id),
        learner_update_steps=(
            int(state.async_learner.get_update_steps())
            if state.async_learner is not None
            else 0
        ),
        replay_prefetch_queue_size=(
            int(state.async_learner.get_prefetch_queue_size())
            if state.async_learner is not None
            else int(state.sync_replay_prefetcher.get_queue_size())
            if state.sync_replay_prefetcher is not None
            else 0
        ),
    )
    state.profiling_last_flush_step = int(state.train_env_step)


def apply_training_updates_and_runtime_hooks(
    ctx: Any,
    spec: EpisodeSpec,
    state: EpisodeState,
    *,
    train_step_before: int,
    train_step_after: int,
    replay_size_before: int,
    replay_size_after: int,
    current_decision_id: Optional[int],
    num_sync_updates: int,
    advance_async_update_calls: Callable[..., None],
    maybe_wait_for_async_learner_budget: Callable[..., None],
    maybe_send_agentlace_timer_stats: Callable[..., None],
    timer_train_episode_id: int,
) -> None:
    current_decision_value = int(
        current_decision_id if current_decision_id is not None else state.decision_step
    )
    if state.async_learner is None:
        if bool(spec.phase_train) and int(num_sync_updates) > 0:
            run_sync_training_updates(
                ctx,
                state,
                num_updates=int(num_sync_updates),
            )
    else:
        advance_async_update_calls(
            phase_train_flag=bool(spec.phase_train),
            train_step_before=int(train_step_before),
            train_step_after=int(train_step_after),
            replay_size_before=int(replay_size_before),
            replay_size_after=int(replay_size_after),
        )
        maybe_wait_for_async_learner_budget(
            train_env_step_value=int(train_step_after),
            decision_step_value=current_decision_value,
        )
        state.last_update_info = state.async_learner.get_last_update_info()
    if (
        bool(spec.phase_train)
        and int(state.train_env_step) % int(ctx.tb_step_period) == 0
        and state.last_update_info
    ):
        log_training_system_metrics(
            ctx,
            state,
            current_decision_id=current_decision_id,
        )
    if bool(spec.phase_train):
        state.decision_step = int(current_decision_value)
    maybe_emit_episode_profiling(
        ctx,
        state,
        timer_train_episode_id=int(timer_train_episode_id),
    )
    maybe_send_agentlace_timer_stats(
        train_env_step_value=int(state.train_env_step),
        decision_step_value=int(state.decision_step),
        train_episode_id_value=int(timer_train_episode_id),
    )


def flush_episode_step_window(ctx: Any, state: EpisodeState) -> None:
    _flush_tb_step_window(
        ctx.tb_writer,
        step_window=ctx.step_metric_window,
        global_env_step=max(0, int(state.train_env_step)),
        control_indices=ctx.control_indices,
        histogram=bool(
            state.train_env_step > 0
            and state.train_env_step % int(ctx.tb_histogram_period) == 0
        ),
    )


def build_episode_result(state: EpisodeState) -> EpisodeResult:
    return EpisodeResult(
        episode_success=bool(state.episode_success),
        episode_return=float(state.episode_return),
        episode_steps=int(state.episode_steps),
        train_env_step=int(state.train_env_step),
        decision_step=int(state.decision_step),
        last_update_info=dict(state.last_update_info),
        agent=state.agent,
        learner_agent=state.learner_agent,
        async_learner=state.async_learner,
        replay_buffer=state.replay_buffer,
        sync_replay_lock=state.sync_replay_lock,
        sync_replay_prefetcher=state.sync_replay_prefetcher,
        profiling_last_flush_step=int(state.profiling_last_flush_step),
    )
