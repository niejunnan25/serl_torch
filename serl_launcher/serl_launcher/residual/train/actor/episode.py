from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from concurrent.futures import Future

from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.train.actor.episode_chunk import execute_chunk_decision
from serl_launcher.residual.train.actor.episode_shared import build_episode_result
from serl_launcher.residual.train.actor.episode_shared import EpisodeResult
from serl_launcher.residual.train.actor.episode_shared import EpisodeSpec
from serl_launcher.residual.train.actor.episode_shared import EpisodeState
from serl_launcher.residual.train.actor.episode_shared import flush_episode_step_window
from serl_launcher.residual.train.actor.episode_step import execute_step_decision
from serl_launcher.residual.train.actor.support import build_policy_input
from serl_launcher.residual.train.config import sample_probing_steps
from serl_launcher.training.profiling import _profile_call


def _run_probing_steps(
    ctx: Any,
    spec: EpisodeSpec,
    state: EpisodeState,
    *,
    update_train_progress: Callable[..., None],
) -> None:
    env = ctx.env
    policy_client = ctx.policy_client
    policy_prefetcher = ctx.policy_prefetcher
    profiler = ctx.profiler
    step_logger = ctx.step_logger

    chunk_horizon = int(ctx.chunk_horizon)
    env_action_dim = int(ctx.env_action_dim)
    step_action_dim = int(ctx.step_action_dim)
    max_train_env_steps = int(ctx.max_train_env_steps)

    probing_steps_target = sample_probing_steps(
        ctx.cfg.training,
        episode_horizon=int(spec.max_episode_steps),
    )
    if probing_steps_target <= 0:
        return

    probing_remaining = int(min(probing_steps_target, spec.max_episode_steps))
    probe_future: Optional[Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]] = None
    while probing_remaining > 0 and state.episode_steps < spec.max_episode_steps:
        if probe_future is not None:
            probe_chunk, probe_info = probe_future.result()
            probe_future = None
        else:
            probe_chunk, probe_info = policy_client.infer_chunk(
                build_policy_input(ctx, state.obs_raw, env.current_instruction)
            )
        probe_base_chunk = select_action_chunk_window(
            probe_chunk,
            horizon=chunk_horizon,
            action_dim=env_action_dim,
        )
        for probe_step in range(chunk_horizon):
            if probing_remaining <= 0 or state.episode_steps >= spec.max_episode_steps:
                break
            base_action = probe_base_chunk[probe_step]
            next_obs_raw, reward, env_done, _, info = _profile_call(
                profiler,
                "env_step",
                env.step,
                base_action,
            )
            state.episode_steps += 1
            if spec.phase_train:
                state.train_env_step += 1
                update_train_progress(train_env_step_value=state.train_env_step)
            probing_remaining -= 1
            state.episode_return += float(reward)
            state.episode_success = bool(info["success"])
            timeout = bool(state.episode_steps >= spec.max_episode_steps)
            budget_exhausted = bool(
                spec.phase_train
                and max_train_env_steps > 0
                and state.train_env_step >= max_train_env_steps
            )
            done = bool(env_done or timeout or budget_exhausted)
            if (
                (not done)
                and probe_step == (chunk_horizon - 1)
                and probing_remaining > 0
                and policy_prefetcher is not None
            ):
                probe_future = policy_prefetcher.submit(
                    build_policy_input(ctx, next_obs_raw, env.current_instruction)
                )
            step_logger.write(
                {
                    "train_env_step": int(state.train_env_step)
                    if spec.phase_train
                    else None,
                    "decision_step": int(state.decision_step)
                    if spec.phase_train
                    else None,
                    "warmup_episode_id": None,
                    "train_episode_id": spec.train_episode_id,
                    "phase_episode_idx": int(spec.phase_episode_idx),
                    "phase": str(spec.phase_name),
                    "episode_step": state.episode_steps,
                    "seed": int(
                        getattr(env, "last_seed", None)
                        if getattr(env, "last_seed", None) is not None
                        else spec.seed
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
                    "a_res_policy_applied": [0.0] * step_action_dim,
                    "a_res": np.zeros_like(base_action, dtype=np.float32).tolist(),
                    "a_final": base_action.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(state.episode_success),
                }
            )
            state.obs_raw = next_obs_raw
            if done:
                state.episode_done = True
                break
        if state.episode_done:
            return


def run_policy_episode(
    ctx: Any,
    spec: EpisodeSpec,
    obs_raw: Dict[str, Any],
    *,
    agent: Any,
    learner_agent: Any,
    async_learner: Optional[Any],
    replay_buffer: Any,
    sync_replay_lock: Optional[Any],
    sync_replay_prefetcher: Optional[Any],
    train_env_step: int,
    decision_step: int,
    last_update_info: Dict[str, Any],
    profiling_last_flush_step: int,
    update_train_progress: Callable[..., None],
    advance_async_update_calls: Callable[..., None],
    maybe_send_agentlace_timer_stats: Callable[..., None],
    maybe_wait_for_async_learner_budget: Callable[..., None],
    save_checkpoint_at_step: Callable[[int], Path],
    timer_train_episode_id: int,
) -> EpisodeResult:
    policy_client = ctx.policy_client

    state = EpisodeState(
        obs_raw=obs_raw,
        train_env_step=int(train_env_step),
        decision_step=int(decision_step),
        last_update_info=dict(last_update_info),
        agent=agent,
        learner_agent=learner_agent,
        async_learner=async_learner,
        replay_buffer=replay_buffer,
        sync_replay_lock=sync_replay_lock,
        sync_replay_prefetcher=sync_replay_prefetcher,
        profiling_last_flush_step=int(profiling_last_flush_step),
    )

    _run_probing_steps(
        ctx,
        spec,
        state,
        update_train_progress=update_train_progress,
    )

    while state.episode_steps < spec.max_episode_steps and not state.episode_done:
        if state.cached_base_chunk is None:
            policy_chunk, infer_info = policy_client.infer_chunk(
                build_policy_input(ctx, state.obs_raw, ctx.env.current_instruction)
            )
            base_chunk = select_action_chunk_window(
                policy_chunk,
                horizon=int(ctx.chunk_horizon),
                action_dim=int(ctx.env_action_dim),
            )
        else:
            base_chunk = state.cached_base_chunk
            infer_info = state.cached_infer_info or {
                "e2e_ms": None,
                "policy_ms": None,
                "server_ms": None,
            }
            state.cached_base_chunk = None
            state.cached_infer_info = None

        if bool(ctx.chunk_step_enabled):
            execute_chunk_decision(
                ctx,
                spec,
                state,
                base_chunk=base_chunk,
                infer_info=infer_info,
                update_train_progress=update_train_progress,
                advance_async_update_calls=advance_async_update_calls,
                maybe_send_agentlace_timer_stats=maybe_send_agentlace_timer_stats,
                maybe_wait_for_async_learner_budget=maybe_wait_for_async_learner_budget,
                save_checkpoint_at_step=save_checkpoint_at_step,
                timer_train_episode_id=int(timer_train_episode_id),
            )
        else:
            execute_step_decision(
                ctx,
                spec,
                state,
                base_chunk=base_chunk,
                infer_info=infer_info,
                update_train_progress=update_train_progress,
                advance_async_update_calls=advance_async_update_calls,
                maybe_send_agentlace_timer_stats=maybe_send_agentlace_timer_stats,
                maybe_wait_for_async_learner_budget=maybe_wait_for_async_learner_budget,
                save_checkpoint_at_step=save_checkpoint_at_step,
                timer_train_episode_id=int(timer_train_episode_id),
            )

    flush_episode_step_window(ctx, state)
    return build_episode_result(state)
