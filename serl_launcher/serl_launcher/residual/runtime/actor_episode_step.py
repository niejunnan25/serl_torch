from __future__ import annotations

import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.action import compose_residual_action
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.runtime.actor_episode_shared import apply_training_updates_and_runtime_hooks
from serl_launcher.residual.runtime.actor_episode_shared import EpisodeSpec
from serl_launcher.residual.runtime.actor_episode_shared import EpisodeState
from serl_launcher.residual.runtime.actor_episode_shared import insert_online_transitions
from serl_launcher.residual.runtime.actor_support import build_policy_input
from serl_launcher.residual.runtime.actor_support import build_step_obs_profiled
from serl_launcher.residual.runtime.actor_support import replay_progress_size
from serl_launcher.residual.runtime.actor_support import resolve_train_gate
from serl_launcher.residual.runtime.obs_utils import _clone_obs_dict
from serl_launcher.residual.runtime.obs_utils import _zero_obs_like
from serl_launcher.residual.runtime.profiling import _profile_call
from serl_launcher.residual.runtime.schedules import _scheduled_alpha
from serl_launcher.residual.runtime.tb_metrics import _append_tb_step_window
from serl_launcher.residual.runtime.tb_metrics import _flush_tb_step_window


def execute_step_decision(
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
    max_train_env_steps = int(ctx.max_train_env_steps)
    tb_step_period = int(ctx.tb_step_period)
    tb_histogram_period = int(ctx.tb_histogram_period)

    next_obs_raw = state.obs_raw
    for chunk_step in range(chunk_horizon):
        if state.episode_steps >= spec.max_episode_steps:
            state.episode_done = True
            break

        train_env_step_before_step = int(state.train_env_step)
        alpha_step = _scheduled_alpha(
            cfg,
            base_alpha=residual_alpha,
            schedule_step=train_env_step_before_step,
        )
        obs_input = build_step_obs_profiled(
            ctx,
            next_obs_raw,
            base_chunk[chunk_step],
            alpha=float(alpha_step),
        )
        gate_prob, gate_on = resolve_train_gate(
            ctx,
            phase_train_flag=bool(spec.phase_train),
            alpha_value=float(alpha_step),
            env_step_value=int(train_env_step_before_step),
            decision_step_value=int(state.decision_step),
        )

        if alpha_step <= 0.0:
            residual_step_action = np.zeros((step_action_dim,), dtype=np.float32)
        elif (not spec.phase_train) or (
            train_env_step_before_step < int(cfg.training.random_steps)
        ):
            residual_step_action = np.random.uniform(
                -1.0,
                1.0,
                size=(step_action_dim,),
            ).astype(np.float32)
        else:
            if state.async_learner is not None:
                residual_step_action = state.async_learner.sample_actor_action(
                    obs_input,
                    step_action_dim,
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
                residual_step_action = as_numpy_action(sampled, step_action_dim)
        policy_residual_step_action = np.asarray(
            residual_step_action,
            dtype=np.float32,
        ).copy()
        if not gate_on:
            residual_step_action = np.zeros_like(residual_step_action)

        delta_action, final_action = compose_residual_action(
            base_action=base_chunk[chunk_step],
            residual_action=residual_step_action,
            indices=control_indices,
            limits=residual_limits,
            alpha=alpha_step,
            clip_gripper=bool(cfg.residual.clip_gripper),
        )

        current_decision_id = int(state.decision_step + 1) if spec.phase_train else None
        next_obs_raw, reward, env_done, _, info = _profile_call(
            profiler,
            "env_step",
            env.step,
            final_action,
        )
        state.episode_steps += 1
        if spec.phase_train:
            state.train_env_step += 1
            update_train_progress()
        train_env_step_after_step = int(state.train_env_step)
        next_alpha_step = _scheduled_alpha(
            cfg,
            base_alpha=residual_alpha,
            schedule_step=train_env_step_after_step,
        )
        state.episode_return += float(reward)
        state.episode_success = bool(info["success"])
        timeout = bool(state.episode_steps >= spec.max_episode_steps)
        budget_exhausted = bool(
            spec.phase_train
            and max_train_env_steps > 0
            and state.train_env_step >= max_train_env_steps
        )
        done = bool(env_done or timeout or budget_exhausted)
        next_chunk_future: Optional[Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]] = None
        if (
            (not done)
            and chunk_step == (chunk_horizon - 1)
            and policy_prefetcher is not None
        ):
            next_chunk_future = policy_prefetcher.submit(
                build_policy_input(ctx, next_obs_raw, env.current_instruction)
            )

        step_logger.write(
            {
                "train_env_step": int(state.train_env_step) if spec.phase_train else None,
                "decision_step": current_decision_id,
                "warmup_episode_id": None,
                "train_episode_id": spec.train_episode_id,
                "phase_episode_idx": int(spec.phase_episode_idx),
                "phase": str(spec.phase_name),
                "episode_step": state.episode_steps,
                "seed": int(env.last_seed if env.last_seed is not None else spec.seed),
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
                "a_res_policy": policy_residual_step_action.tolist(),
                "a_res_policy_applied": residual_step_action.tolist(),
                "a_res": delta_action.tolist(),
                "a_final": final_action.tolist(),
                "alpha": float(alpha_step),
                "epsilon_gate_prob": float(gate_prob),
                "epsilon_gate_on": bool(gate_on),
                "reward": float(reward),
                "done": bool(done),
                "success": bool(state.episode_success),
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
        if spec.phase_train and state.train_env_step % tb_step_period == 0:
            _flush_tb_step_window(
                tb_writer,
                step_window=step_metric_window,
                global_env_step=state.train_env_step,
                control_indices=control_indices,
                histogram=bool(state.train_env_step % tb_histogram_period == 0),
            )

        if done:
            next_obs_input = _zero_obs_like(obs_input)
            mask = 0.0
        elif chunk_step < (chunk_horizon - 1):
            next_obs_input = build_step_obs_profiled(
                ctx,
                next_obs_raw,
                base_chunk[chunk_step + 1],
                alpha=float(next_alpha_step),
            )
            mask = 1.0
        else:
            if next_chunk_future is not None:
                next_policy_chunk, next_infer_info = next_chunk_future.result()
            else:
                next_policy_chunk, next_infer_info = policy_client.infer_chunk(
                    build_policy_input(ctx, next_obs_raw, env.current_instruction)
                )
            next_base_chunk = select_action_chunk_window(
                next_policy_chunk,
                horizon=chunk_horizon,
                action_dim=env_action_dim,
            )
            next_obs_input = build_step_obs_profiled(
                ctx,
                next_obs_raw,
                next_base_chunk[0],
                alpha=float(next_alpha_step),
            )
            state.cached_base_chunk = next_base_chunk
            state.cached_infer_info = next_infer_info
            mask = 1.0

        transition_payload = {
            "observations": _clone_obs_dict(obs_input),
            "actions": final_action.astype(np.float32),
            "next_observations": _clone_obs_dict(next_obs_input),
            "rewards": np.float32(reward),
            "masks": np.float32(mask),
            "dones": bool(done),
            "episode_id": int(spec.init_episode_idx),
            "episode_step": int(state.episode_steps - 1),
        }
        replay_size_before = int(replay_progress_size(state.replay_buffer))
        insert_online_transitions(
            state,
            [transition_payload],
            chunk_step_enabled=bool(ctx.chunk_step_enabled),
        )
        replay_size_after = int(replay_progress_size(state.replay_buffer))

        num_sync_updates = 0
        if (
            state.async_learner is None
            and spec.phase_train
            and replay_progress_size(state.replay_buffer) >= int(cfg.training.training_starts)
            and train_env_step_before_step % int(cfg.training.update_every) == 0
        ):
            num_sync_updates = int(cfg.training.updates_per_step)
        apply_training_updates_and_runtime_hooks(
            ctx,
            spec,
            state,
            train_step_before=int(train_env_step_before_step),
            train_step_after=int(train_env_step_after_step),
            replay_size_before=int(replay_size_before),
            replay_size_after=int(replay_size_after),
            current_decision_id=current_decision_id,
            num_sync_updates=int(num_sync_updates),
            advance_async_update_calls=advance_async_update_calls,
            maybe_wait_for_async_learner_budget=maybe_wait_for_async_learner_budget,
            maybe_send_agentlace_timer_stats=maybe_send_agentlace_timer_stats,
            timer_train_episode_id=int(timer_train_episode_id),
        )

        if (
            spec.phase_train
            and int(ctx.checkpoint_every_steps) > 0
            and train_env_step_after_step % int(ctx.checkpoint_every_steps) == 0
        ):
            save_checkpoint_at_step(int(train_env_step_after_step))

        if done:
            state.episode_done = True
            break

        state.obs_raw = next_obs_raw
    if not state.episode_done:
        state.obs_raw = next_obs_raw
