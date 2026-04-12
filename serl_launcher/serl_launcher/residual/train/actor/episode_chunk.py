from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.action import compose_residual_action_chunk
from serl_launcher.residual.action import reshape_flat_action_to_chunk
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.train.actor.episode_shared import (
    apply_training_updates_and_runtime_hooks,
)
from serl_launcher.residual.train.actor.episode_shared import EpisodeSpec
from serl_launcher.residual.train.actor.episode_shared import EpisodeState
from serl_launcher.residual.train.actor.episode_shared import insert_online_transitions
from serl_launcher.residual.train.actor.support import build_chunk_step_record
from serl_launcher.residual.train.actor.support import build_policy_input
from serl_launcher.residual.train.actor.support import build_step_obs_profiled
from serl_launcher.residual.train.actor.support import replay_progress_size
from serl_launcher.residual.train.actor.support import resolve_alpha_step
from serl_launcher.residual.train.actor.support import resolve_train_gate
from serl_launcher.training.profiling import _profile_call
from serl_launcher.residual.train.telemetry import _append_tb_step_window
from serl_launcher.residual.train.telemetry import _flush_tb_step_window
from serl_launcher.training.loop_utils import _count_env_step_update_triggers
from serl_launcher.training.loop_utils import _iter_period_hits
from serl_launcher.training.loop_utils import _remaining_train_budget_steps


def execute_chunk_decision(
    ctx: Any,
    spec: EpisodeSpec,
    state: EpisodeState,
    *,
    base_chunk: np.ndarray,
    infer_info: Dict[str, Optional[float]],
    update_train_progress: Callable[..., None],
    advance_async_update_calls: Callable[..., None],
    maybe_send_agentlace_timer_stats: Callable[..., None],
    maybe_wait_for_async_learner_budget: Callable[..., None],
    save_checkpoint_at_step: Callable[[int], Path],
    timer_train_episode_id: int,
) -> None:
    cfg = ctx.cfg
    env = ctx.env
    policy_client = ctx.policy_client
    policy_prefetcher = ctx.policy_prefetcher
    algorithm = ctx.algorithm
    profiler = ctx.profiler
    tb_writer = ctx.tb_writer
    step_logger = ctx.step_logger
    step_metric_window = ctx.step_metric_window

    chunk_horizon = int(ctx.chunk_horizon)
    env_action_dim = int(ctx.env_action_dim)
    control_indices = ctx.control_indices
    step_action_dim = int(ctx.step_action_dim)
    residual_limits = ctx.residual_limits
    residual_alpha = float(ctx.residual_alpha)
    chunk_step_scheduler_clock = str(ctx.chunk_step_scheduler_clock)
    agent_action_dim = int(ctx.agent_action_dim)
    max_train_env_steps = int(ctx.max_train_env_steps)
    tb_step_period = int(ctx.tb_step_period)
    tb_histogram_period = int(ctx.tb_histogram_period)

    train_env_step_before_chunk = int(state.train_env_step)
    schedule_step = (
        train_env_step_before_chunk
        if chunk_step_scheduler_clock == "env_step"
        else int(state.decision_step)
    )
    alpha_step = resolve_alpha_step(
        cfg,
        base_alpha=residual_alpha,
        schedule_step=schedule_step,
    )
    obs_input = build_step_obs_profiled(
        ctx,
        state.obs_raw,
        base_chunk[0],
        base_action_chunk=base_chunk,
        alpha=float(alpha_step),
    )
    gate_prob, gate_on = resolve_train_gate(
        ctx,
        phase_train_flag=bool(spec.phase_train),
        alpha_value=float(alpha_step),
        env_step_value=int(train_env_step_before_chunk),
        decision_step_value=int(state.decision_step),
    )

    if alpha_step <= 0.0:
        residual_chunk = np.zeros((chunk_horizon, step_action_dim), dtype=np.float32)
    elif (not spec.phase_train) or (
        train_env_step_before_chunk < int(cfg.training.random_steps)
    ):
        residual_chunk = np.random.uniform(
            -1.0,
            1.0,
            size=(chunk_horizon, step_action_dim),
        ).astype(np.float32)
    else:
        if state.async_learner is not None:
            sampled_chunk = state.async_learner.sample_actor_action(
                obs_input,
                agent_action_dim,
            )
        else:
            sample_actions_start = time.perf_counter()
            sampled = algorithm.sample_actions(
                state.agent,
                obs_input,
                deterministic=False,
            )
            profiler.record_duration(
                "agent_sample_actions",
                (time.perf_counter() - sample_actions_start) * 1000.0,
            )
            sampled_chunk = as_numpy_action(sampled, agent_action_dim)
        residual_chunk = reshape_flat_action_to_chunk(
            sampled_chunk,
            action_dim=step_action_dim,
            chunk_horizon=chunk_horizon,
        )
    policy_residual_chunk = np.asarray(residual_chunk, dtype=np.float32).copy()
    if not gate_on:
        residual_chunk = np.zeros_like(residual_chunk)

    remaining_budget_steps = (
        _remaining_train_budget_steps(
            max_train_env_steps=max_train_env_steps,
            train_env_step=state.train_env_step,
        )
        if spec.phase_train
        else None
    )
    execute_horizon = int(
        min(chunk_horizon, spec.max_episode_steps - state.episode_steps)
    )
    if remaining_budget_steps is not None:
        execute_horizon = int(min(execute_horizon, remaining_budget_steps))
    if spec.phase_train and train_env_step_before_chunk < int(
        cfg.training.random_steps
    ):
        execute_horizon = int(
            min(
                execute_horizon,
                int(cfg.training.random_steps) - train_env_step_before_chunk,
            )
        )
    if execute_horizon <= 0:
        state.episode_done = True
        return

    current_decision_id = int(state.decision_step + 1) if spec.phase_train else None
    replay_size_before = int(replay_progress_size(state.replay_buffer))
    executed_base_chunk = np.asarray(base_chunk[:execute_horizon], dtype=np.float32)
    executed_policy_residual_chunk = np.asarray(
        policy_residual_chunk[:execute_horizon],
        dtype=np.float32,
    )
    executed_residual_chunk = np.asarray(
        residual_chunk[:execute_horizon],
        dtype=np.float32,
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
    executed_base_chunk = executed_base_chunk[:actual_chunk_steps]
    executed_residual_chunk = executed_residual_chunk[:actual_chunk_steps]
    delta_chunk = delta_chunk[:actual_chunk_steps]
    final_chunk = final_chunk[:actual_chunk_steps]

    done = False
    current_step_obs_raw = state.obs_raw
    chunk_step_payloads = []
    for chunk_step in range(actual_chunk_steps):
        current_episode_step = int(state.episode_steps)
        reward = float(chunk_rewards[chunk_step])
        info = chunk_infos[chunk_step]
        state.episode_steps += 1
        if spec.phase_train:
            state.train_env_step += 1
            update_train_progress(train_env_step_value=state.train_env_step)
        state.episode_return += reward
        state.episode_success = bool(info.get("success", state.episode_success))

        timeout = bool(state.episode_steps >= spec.max_episode_steps)
        remaining_budget_steps = (
            _remaining_train_budget_steps(
                max_train_env_steps=max_train_env_steps,
                train_env_step=state.train_env_step,
            )
            if spec.phase_train
            else None
        )
        budget_exhausted = bool(
            remaining_budget_steps is not None and remaining_budget_steps <= 0
        )
        done = bool(chunk_env_dones[chunk_step] or timeout or budget_exhausted)

        step_logger.write(
            {
                "train_env_step": int(state.train_env_step)
                if spec.phase_train
                else None,
                "decision_step": current_decision_id,
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
                "is_probing": False,
                "replan_point": bool(chunk_step == 0),
                "chunk_step": int(chunk_step),
                "chunk_horizon": int(actual_chunk_steps),
                "infer_e2e_ms": infer_info.get("e2e_ms") if chunk_step == 0 else None,
                "infer_policy_ms": infer_info.get("policy_ms")
                if chunk_step == 0
                else None,
                "infer_server_ms": infer_info.get("server_ms")
                if chunk_step == 0
                else None,
                "a_base": executed_base_chunk[chunk_step].tolist(),
                "a_res_policy": executed_policy_residual_chunk[chunk_step].tolist(),
                "a_res_policy_applied": executed_residual_chunk[chunk_step].tolist(),
                "a_res": delta_chunk[chunk_step].tolist(),
                "a_final": final_chunk[chunk_step].tolist(),
                "alpha": float(alpha_step),
                "epsilon_gate_prob": float(gate_prob),
                "epsilon_gate_on": bool(gate_on),
                "reward": float(reward),
                "done": bool(done),
                "success": bool(state.episode_success),
            }
        )
        step_payload = build_chunk_step_record(
            ctx,
            current_step_obs_raw,
            base_action=executed_base_chunk[chunk_step],
            final_action=final_chunk[chunk_step],
            alpha_obs=float(alpha_step),
            episode_id=int(spec.init_episode_idx),
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
            residual_action_raw=executed_policy_residual_chunk[chunk_step],
            residual_action_applied=executed_residual_chunk[chunk_step],
            delta_action=delta_chunk[chunk_step],
            base_action=executed_base_chunk[chunk_step],
            final_action=final_chunk[chunk_step],
            infer_info=infer_info,
            replan_point=bool(chunk_step == 0),
        )
        if spec.phase_train and state.train_env_step % tb_step_period == 0:
            _flush_tb_step_window(
                tb_writer,
                step_window=step_metric_window,
                global_env_step=state.train_env_step,
                control_indices=control_indices,
                histogram=bool(state.train_env_step % tb_histogram_period == 0),
            )

        if chunk_step < (actual_chunk_steps - 1):
            current_step_obs_raw = chunk_observations[chunk_step]
        if done:
            state.episode_done = True
            break

    train_env_step_after_chunk = int(state.train_env_step)
    if not done:
        next_policy_chunk, next_infer_info = policy_client.infer(
            build_policy_input(ctx, next_obs_raw, env.current_instruction)
        )
        next_base_chunk = select_action_chunk_window(
            next_policy_chunk,
            horizon=chunk_horizon,
            action_dim=env_action_dim,
        )
        state.cached_base_chunk = next_base_chunk
        state.cached_infer_info = next_infer_info

    insert_online_transitions(
        state,
        chunk_step_payloads,
        chunk_step_enabled=bool(ctx.chunk_step_enabled),
    )

    replay_size_after = int(replay_progress_size(state.replay_buffer))
    trigger_count = 0
    if state.async_learner is None and spec.phase_train:
        trigger_count = _count_env_step_update_triggers(
            train_step_before=train_env_step_before_chunk,
            train_step_after=train_env_step_after_chunk,
            replay_size_before=replay_size_before,
            replay_size_after=replay_size_after,
            training_starts=int(cfg.training.training_starts),
            update_every=int(cfg.training.update_every),
        )
    apply_training_updates_and_runtime_hooks(
        ctx,
        spec,
        state,
        train_step_before=int(train_env_step_before_chunk),
        train_step_after=int(train_env_step_after_chunk),
        replay_size_before=int(replay_size_before),
        replay_size_after=int(replay_size_after),
        current_decision_id=current_decision_id,
        num_sync_updates=int(trigger_count * int(cfg.training.updates_per_step)),
        advance_async_update_calls=advance_async_update_calls,
        maybe_wait_for_async_learner_budget=maybe_wait_for_async_learner_budget,
        maybe_send_agentlace_timer_stats=maybe_send_agentlace_timer_stats,
        timer_train_episode_id=int(timer_train_episode_id),
    )

    checkpoint_hits = _iter_period_hits(
        step_before=train_env_step_before_chunk,
        step_after=train_env_step_after_chunk,
        period=int(ctx.checkpoint_every_steps),
    )
    if spec.phase_train and checkpoint_hits:
        for checkpoint_step in checkpoint_hits:
            save_checkpoint_at_step(int(checkpoint_step))

    state.obs_raw = next_obs_raw
