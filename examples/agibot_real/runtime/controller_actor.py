"""Controller-driven AgiBot actor runtime."""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Deque
from typing import Dict
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.action import as_numpy_action_chunk
from serl_launcher.residual.action import compose_residual_action_chunk
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.train.actor.episode import EpisodeSpec
from serl_launcher.residual.train.actor.episode_shared import (
    EpisodeResult,
)
from serl_launcher.residual.train.actor.episode_shared import EpisodeState
from serl_launcher.residual.train.actor.episode_shared import (
    apply_training_updates_and_runtime_hooks,
)
from serl_launcher.residual.train.actor.episode_shared import build_episode_result
from serl_launcher.residual.train.actor.episode_shared import flush_episode_step_window
from serl_launcher.residual.train.actor.episode_shared import insert_online_transitions
from serl_launcher.residual.train.actor.setup import build_actor_runtime_session
from serl_launcher.residual.train.actor.support import build_chunk_step_record
from serl_launcher.residual.train.actor.support import build_policy_input
from serl_launcher.residual.train.actor.support import clear_obs_cache
from serl_launcher.residual.train.actor.support import flush_external_agentlace_actor
from serl_launcher.residual.train.actor.support import new_progress
from serl_launcher.residual.train.actor.support import replay_progress_size
from serl_launcher.residual.train.actor.support import resolve_train_gate
from serl_launcher.residual.train.actor.support import save_checkpoint_at_step
from serl_launcher.residual.train.actor.support import send_agentlace_timer_stats
from serl_launcher.residual.train.actor.support import sync_async_bounded_lag_baseline
from serl_launcher.residual.train.actor.support import update_train_progress
from serl_launcher.residual.train.actor.support import wait_for_async_learner_budget
from serl_launcher.residual.train.actor.support import (
    advance_async_update_calls as advance_async_update_calls_impl,
)
from serl_launcher.residual.train.async_eval import _sync_async_eval_results_to_tb
from serl_launcher.residual.train.schedules import _scheduled_alpha
from serl_launcher.residual.train.telemetry import _append_tb_step_window
from serl_launcher.training.profiling import _emit_profiling_snapshot
from serl_launcher.utils.serialization import _to_jsonable


STATE_RUNNING = "RUNNING"
STATE_WAIT_READY = "WAIT_READY"
STATE_PAUSED = "PAUSED"
STATE_RESETTING = "RESETTING"
STATE_EPISODE_DONE = "EPISODE_DONE"

TERMINAL_SUCCESS = "success"
TERMINAL_FAIL = "fail"
TERMINAL_RESET = "reset"
TERMINAL_TIMEOUT = "timeout"


@dataclass
class _PlannedStep:
    sequence_id: int
    obs_before: Optional[Dict[str, Any]]
    base_action: np.ndarray
    policy_residual: np.ndarray
    applied_residual: np.ndarray
    delta_action: np.ndarray
    final_action: np.ndarray
    alpha: float
    gate_prob: float
    gate_on: bool
    infer_info: Dict[str, Optional[float]]
    chunk_step: int
    executed_horizon: int
    current_decision_id: Optional[int]


@dataclass
class _BufferedStep:
    planned: _PlannedStep
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]


def _validate_controller_runtime(ctx: Any) -> None:
    if not bool(getattr(ctx.env, "controller_enabled", False)):
        raise ValueError(
            "controller actor runtime requires controller.enabled=true on the env"
        )
    if not bool(ctx.chunk_step_enabled):
        raise ValueError(
            "controller actor runtime requires chunk_step.enabled=true"
        )
    if bool(ctx.need_warmup_first):
        raise ValueError(
            "controller actor runtime does not support runtime warmup episodes yet; "
            "set training.warmup.episodes=0"
        )
    if bool(ctx.async_eval_enabled):
        raise ValueError(
            "controller actor runtime does not support async eval yet; "
            "set training.async_eval.enabled=false"
        )
    for phase in ctx.cfg.training.phases:
        if not bool(phase.get("train", True)):
            raise ValueError(
                "controller actor runtime currently supports only train=true phases"
            )


def _terminal_grace_sec(cfg: DictConfig) -> float:
    controller_cfg = cfg.get("controller", {})
    return max(0.0, float(controller_cfg.get("terminal_grace_sec", 0.15)))


def _controller_poll_sec(cfg: DictConfig) -> float:
    controller_cfg = cfg.get("controller", {})
    return max(0.01, float(controller_cfg.get("poll_interval_sec", 0.05)))


def _controller_inflight_matches_planned_step(
    meta: Dict[str, Any],
    planned_steps: Deque[_PlannedStep],
) -> bool:
    if not planned_steps:
        return False
    inflight_sequence_id = meta.get("inflight_sequence_id", None)
    if inflight_sequence_id is None:
        return False
    try:
        inflight_sequence_id = int(inflight_sequence_id)
    except (TypeError, ValueError):
        return False
    return int(planned_steps[0].sequence_id) == inflight_sequence_id


def _override_buffered_step_from_meta(
    buffered: _BufferedStep,
    meta: Dict[str, Any],
) -> _BufferedStep:
    info = dict(buffered.info)
    terminal_signal = meta.get("terminal_signal", None)
    terminal_info = meta.get("terminal_info", {})
    if isinstance(terminal_info, dict):
        info.update(terminal_info)
    reward = float(buffered.reward)
    done = bool(buffered.done)
    truncated = bool(buffered.truncated)
    if terminal_signal == TERMINAL_SUCCESS:
        reward = 1.0
        done = True
        truncated = False
        info["success"] = True
        info["human_success"] = True
    elif terminal_signal == TERMINAL_FAIL:
        reward = 0.0
        done = True
        truncated = False
        info["success"] = False
        info["human_fail"] = True
    elif terminal_signal == TERMINAL_RESET:
        reward = 0.0
        done = True
        truncated = True
        info["success"] = False
        info["human_reset"] = True
    elif terminal_signal == TERMINAL_TIMEOUT:
        reward = 0.0
        done = True
        truncated = True
        info["success"] = False
        info["time_limit_reached"] = True
    return _BufferedStep(
        planned=buffered.planned,
        reward=float(reward),
        done=bool(done),
        truncated=bool(truncated),
        info=info,
    )


def _flush_buffered_step(
    ctx: Any,
    spec: EpisodeSpec,
    state: EpisodeState,
    *,
    buffered: _BufferedStep,
    update_train_progress_fn,
    advance_async_update_calls_fn,
    maybe_send_agentlace_timer_stats_fn,
    maybe_wait_for_async_learner_budget_fn,
    save_checkpoint_at_step_fn,
    timer_train_episode_id: int,
) -> None:
    replay_size_before = int(replay_progress_size(state.replay_buffer))
    train_step_before = int(state.train_env_step)
    step_number = int(buffered.planned.chunk_step)
    state.episode_steps = int(state.episode_steps + 1)
    if spec.phase_train:
        state.train_env_step = int(state.train_env_step + 1)
        update_train_progress_fn(train_env_step_value=int(state.train_env_step))
    state.episode_return += float(buffered.reward)
    state.episode_success = bool(buffered.info.get("success", state.episode_success))

    ctx.step_logger.write(
        {
            "train_env_step": int(state.train_env_step) if spec.phase_train else None,
            "decision_step": buffered.planned.current_decision_id,
            "warmup_episode_id": None,
            "train_episode_id": spec.train_episode_id,
            "phase_episode_idx": int(spec.phase_episode_idx),
            "phase": str(spec.phase_name),
            "episode_step": int(state.episode_steps),
            "seed": int(ctx.env.last_seed if ctx.env.last_seed is not None else spec.seed),
            "init_state_idx": (
                int(ctx.env.current_init_state_idx)
                if ctx.env.current_init_state_idx is not None
                else None
            ),
            "is_probing": False,
            "replan_point": bool(buffered.planned.chunk_step == 0),
            "chunk_step": int(step_number),
            "chunk_horizon": int(buffered.planned.executed_horizon),
            "infer_e2e_ms": buffered.planned.infer_info.get("e2e_ms")
            if step_number == 0
            else None,
            "infer_policy_ms": buffered.planned.infer_info.get("policy_ms")
            if step_number == 0
            else None,
            "infer_server_ms": buffered.planned.infer_info.get("server_ms")
            if step_number == 0
            else None,
            "a_base": buffered.planned.base_action.tolist(),
            "a_res_policy": buffered.planned.policy_residual.tolist(),
            "a_res_policy_applied": buffered.planned.applied_residual.tolist(),
            "a_res": buffered.planned.delta_action.tolist(),
            "a_final": buffered.planned.final_action.tolist(),
            "alpha": float(buffered.planned.alpha),
            "epsilon_gate_prob": float(buffered.planned.gate_prob),
            "epsilon_gate_on": bool(buffered.planned.gate_on),
            "reward": float(buffered.reward),
            "done": bool(buffered.done),
            "success": bool(buffered.info.get("success", False)),
        }
    )

    _append_tb_step_window(
        ctx.step_metric_window,
        reward=float(buffered.reward),
        alpha=float(buffered.planned.alpha),
        gate_prob=float(buffered.planned.gate_prob),
        gate_on=bool(buffered.planned.gate_on),
        residual_action_raw=buffered.planned.policy_residual,
        residual_action_applied=buffered.planned.applied_residual,
        delta_action=buffered.planned.delta_action,
        base_action=buffered.planned.base_action,
        final_action=buffered.planned.final_action,
        infer_info=buffered.planned.infer_info,
        replan_point=bool(buffered.planned.chunk_step == 0),
    )

    step_payload = build_chunk_step_record(
        ctx,
        buffered.planned.obs_before or ctx.env.get_latest_obs(),
        base_action=buffered.planned.base_action,
        final_action=buffered.planned.final_action,
        alpha_obs=float(buffered.planned.alpha),
        episode_id=int(spec.init_episode_idx),
        episode_step=int(state.episode_steps - 1),
        done=bool(buffered.done),
    )
    step_payload["rewards"] = float(buffered.reward)
    insert_online_transitions(
        state,
        [step_payload],
        chunk_step_enabled=True,
    )
    replay_size_after = int(replay_progress_size(state.replay_buffer))

    num_sync_updates = 0
    if (
        state.async_learner is None
        and spec.phase_train
        and replay_progress_size(state.replay_buffer)
        >= int(ctx.cfg.training.training_starts)
        and train_step_before % int(ctx.cfg.training.update_every) == 0
    ):
        num_sync_updates = int(ctx.cfg.training.updates_per_step)

    apply_training_updates_and_runtime_hooks(
        ctx,
        spec,
        state,
        train_step_before=int(train_step_before),
        train_step_after=int(state.train_env_step),
        replay_size_before=int(replay_size_before),
        replay_size_after=int(replay_size_after),
        current_decision_id=buffered.planned.current_decision_id,
        num_sync_updates=int(num_sync_updates),
        advance_async_update_calls=advance_async_update_calls_fn,
        maybe_wait_for_async_learner_budget=maybe_wait_for_async_learner_budget_fn,
        maybe_send_agentlace_timer_stats=maybe_send_agentlace_timer_stats_fn,
        timer_train_episode_id=int(timer_train_episode_id),
    )

    if (
        spec.phase_train
        and int(ctx.checkpoint_every_steps) > 0
        and int(state.train_env_step) % int(ctx.checkpoint_every_steps) == 0
    ):
        save_checkpoint_at_step_fn(int(state.train_env_step))

    if bool(buffered.done):
        state.episode_done = True


def _plan_chunk(
    ctx: Any,
    spec: EpisodeSpec,
    state: EpisodeState,
    obs_raw: Dict[str, Any],
) -> Deque[_PlannedStep]:
    cfg = ctx.cfg
    env = ctx.env
    action_chunk, infer_info = ctx.policy_client.infer_chunk(
        build_policy_input(ctx, obs_raw, env.current_instruction)
    )
    base_chunk = select_action_chunk_window(
        action_chunk,
        horizon=int(ctx.chunk_horizon),
        action_dim=int(ctx.env_action_dim),
    )

    schedule_step = (
        int(state.train_env_step)
        if str(ctx.chunk_step_scheduler_clock) == "env_step"
        else int(state.decision_step)
    )
    alpha_step = _scheduled_alpha(
        cfg,
        base_alpha=float(ctx.residual_alpha),
        schedule_step=int(schedule_step),
    )
    gate_prob, gate_on = resolve_train_gate(
        ctx,
        phase_train_flag=bool(spec.phase_train),
        alpha_value=float(alpha_step),
        env_step_value=int(state.train_env_step),
        decision_step_value=int(state.decision_step),
    )

    if alpha_step <= 0.0:
        residual_chunk = np.zeros((int(ctx.chunk_horizon), int(ctx.step_action_dim)), dtype=np.float32)
    elif (not spec.phase_train) or (
        int(state.train_env_step) < int(cfg.training.random_steps)
    ):
        residual_chunk = np.random.uniform(
            -1.0,
            1.0,
            size=(int(ctx.chunk_horizon), int(ctx.step_action_dim)),
        ).astype(np.float32)
    else:
        obs_input = ctx.bindings.build_step_obs_profiled(
            ctx.profiler,
            obs_raw,
            base_chunk[0],
            stack_horizon=int(ctx.stack_horizon),
            action_dim=int(ctx.env_action_dim),
            base_action_chunk=base_chunk,
            alpha=float(alpha_step),
            state_mode=str(ctx.obs_state_mode),
        )
        if state.async_learner is not None:
            sampled_chunk = state.async_learner.sample_actor_action(
                obs_input,
                int(ctx.agent_action_dim),
            )
        else:
            sampled = ctx.algorithm.sample_actions(
                state.agent,
                obs_input,
                deterministic=False,
            )
            sampled_chunk = as_numpy_action(sampled, int(ctx.agent_action_dim))
        residual_chunk = as_numpy_action_chunk(
            sampled_chunk,
            action_dim=int(ctx.step_action_dim),
            chunk_horizon=int(ctx.chunk_horizon),
        )
    policy_residual_chunk = np.asarray(residual_chunk, dtype=np.float32).copy()
    if not gate_on:
        residual_chunk = np.zeros_like(residual_chunk)

    execute_horizon = int(
        min(int(ctx.chunk_horizon), int(spec.max_episode_steps) - int(state.episode_steps))
    )
    if spec.phase_train and int(ctx.max_train_env_steps) > 0:
        remaining_budget = int(ctx.max_train_env_steps) - int(state.train_env_step)
        execute_horizon = int(min(execute_horizon, remaining_budget))
    if spec.phase_train and int(state.train_env_step) < int(cfg.training.random_steps):
        execute_horizon = int(
            min(
                execute_horizon,
                int(cfg.training.random_steps) - int(state.train_env_step),
            )
        )
    if execute_horizon <= 0:
        return deque()

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
        indices=ctx.control_indices,
        limits=ctx.residual_limits,
        alpha=float(alpha_step),
        clip_gripper=bool(cfg.residual.clip_gripper),
    )

    sequence_ids = env.enqueue_action_chunk(final_chunk)
    if len(sequence_ids) != execute_horizon:
        raise RuntimeError(
            "controller enqueue_action_chunk accepted "
            f"{len(sequence_ids)} steps, expected {execute_horizon}"
        )

    current_decision_id = int(state.decision_step + 1) if spec.phase_train else None
    planned_steps: Deque[_PlannedStep] = deque()
    for chunk_step, sequence_id in enumerate(sequence_ids):
        planned_steps.append(
            _PlannedStep(
                sequence_id=int(sequence_id),
                obs_before=(obs_raw if chunk_step == 0 else None),
                base_action=np.asarray(executed_base_chunk[chunk_step], dtype=np.float32),
                policy_residual=np.asarray(
                    executed_policy_residual_chunk[chunk_step], dtype=np.float32
                ),
                applied_residual=np.asarray(
                    executed_residual_chunk[chunk_step], dtype=np.float32
                ),
                delta_action=np.asarray(delta_chunk[chunk_step], dtype=np.float32),
                final_action=np.asarray(final_chunk[chunk_step], dtype=np.float32),
                alpha=float(alpha_step),
                gate_prob=float(gate_prob),
                gate_on=bool(gate_on),
                infer_info=dict(infer_info),
                chunk_step=int(chunk_step),
                executed_horizon=int(execute_horizon),
                current_decision_id=current_decision_id,
            )
        )
    return planned_steps


def _run_controller_episode(
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
    update_train_progress_fn,
    advance_async_update_calls_fn,
    maybe_send_agentlace_timer_stats_fn,
    maybe_wait_for_async_learner_budget_fn,
    save_checkpoint_at_step_fn,
    timer_train_episode_id: int,
) -> EpisodeResult:
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

    env = ctx.env
    poll_sec = _controller_poll_sec(ctx.cfg)
    terminal_grace_sec = _terminal_grace_sec(ctx.cfg)
    planned_steps: Deque[_PlannedStep] = deque()
    buffered_step: Optional[_BufferedStep] = None
    queue_empty_since: Optional[float] = None
    current_obs_raw = obs_raw

    while not state.episode_done and int(state.episode_steps) < int(spec.max_episode_steps):
        meta = env.get_controller_meta()
        controller_state = str(meta.get("state", None))
        terminal_signal = meta.get("terminal_signal", None)

        if (
            terminal_signal
            not in {
                TERMINAL_SUCCESS,
                TERMINAL_FAIL,
                TERMINAL_RESET,
                TERMINAL_TIMEOUT,
            }
            and buffered_step is not None
            and (not planned_steps)
        ):
            if queue_empty_since is None:
                queue_empty_since = time.time()
            if (time.time() - float(queue_empty_since)) >= float(terminal_grace_sec):
                _flush_buffered_step(
                    ctx,
                    spec,
                    state,
                    buffered=buffered_step,
                    update_train_progress_fn=update_train_progress_fn,
                    advance_async_update_calls_fn=advance_async_update_calls_fn,
                    maybe_send_agentlace_timer_stats_fn=maybe_send_agentlace_timer_stats_fn,
                    maybe_wait_for_async_learner_budget_fn=maybe_wait_for_async_learner_budget_fn,
                    save_checkpoint_at_step_fn=save_checkpoint_at_step_fn,
                    timer_train_episode_id=int(timer_train_episode_id),
                )
                buffered_step = None
                if state.episode_done:
                    break
        else:
            queue_empty_since = None

        polled = env.poll_controller_transitions(max_items=int(ctx.chunk_horizon))
        if polled:
            queue_empty_since = None
            for payload in polled:
                if not planned_steps:
                    ctx.logger.warning(
                        "Controller transition received without a planned step: seq=%s",
                        payload.get("sequence_id", None),
                    )
                    continue
                planned = planned_steps.popleft()
                if int(planned.sequence_id) != int(payload["sequence_id"]):
                    raise RuntimeError(
                        "Controller transition sequence mismatch: "
                        f"planned={planned.sequence_id} observed={payload['sequence_id']}"
                    )
                if buffered_step is not None:
                    _flush_buffered_step(
                        ctx,
                        spec,
                        state,
                        buffered=buffered_step,
                        update_train_progress_fn=update_train_progress_fn,
                        advance_async_update_calls_fn=advance_async_update_calls_fn,
                        maybe_send_agentlace_timer_stats_fn=maybe_send_agentlace_timer_stats_fn,
                        maybe_wait_for_async_learner_budget_fn=maybe_wait_for_async_learner_budget_fn,
                        save_checkpoint_at_step_fn=save_checkpoint_at_step_fn,
                        timer_train_episode_id=int(timer_train_episode_id),
                    )
                    buffered_step = None

                current_obs_raw = payload["obs"]
                buffered_step = _BufferedStep(
                    planned=planned,
                    reward=float(payload["reward"]),
                    done=bool(payload["done"] or payload["truncated"]),
                    truncated=bool(payload["truncated"]),
                    info=dict(payload["info"]),
                )
                if planned_steps:
                    next_planned = planned_steps[0]
                    next_planned.obs_before = current_obs_raw
                if bool(buffered_step.done):
                    _flush_buffered_step(
                        ctx,
                        spec,
                        state,
                        buffered=buffered_step,
                        update_train_progress_fn=update_train_progress_fn,
                        advance_async_update_calls_fn=advance_async_update_calls_fn,
                        maybe_send_agentlace_timer_stats_fn=maybe_send_agentlace_timer_stats_fn,
                        maybe_wait_for_async_learner_budget_fn=maybe_wait_for_async_learner_budget_fn,
                        save_checkpoint_at_step_fn=save_checkpoint_at_step_fn,
                        timer_train_episode_id=int(timer_train_episode_id),
                    )
                    buffered_step = None
                    planned_steps.clear()
                    break
            if state.episode_done:
                break
            continue

        if (
            terminal_signal in {
                TERMINAL_SUCCESS,
                TERMINAL_FAIL,
                TERMINAL_RESET,
                TERMINAL_TIMEOUT,
            }
            and controller_state != STATE_RUNNING
        ):
            if _controller_inflight_matches_planned_step(meta, planned_steps):
                time.sleep(poll_sec)
                continue
            if buffered_step is not None:
                patched = _override_buffered_step_from_meta(buffered_step, meta)
                _flush_buffered_step(
                    ctx,
                    spec,
                    state,
                    buffered=patched,
                    update_train_progress_fn=update_train_progress_fn,
                    advance_async_update_calls_fn=advance_async_update_calls_fn,
                    maybe_send_agentlace_timer_stats_fn=maybe_send_agentlace_timer_stats_fn,
                    maybe_wait_for_async_learner_budget_fn=maybe_wait_for_async_learner_budget_fn,
                    save_checkpoint_at_step_fn=save_checkpoint_at_step_fn,
                    timer_train_episode_id=int(timer_train_episode_id),
                )
                buffered_step = None
            else:
                state.episode_success = bool(terminal_signal == TERMINAL_SUCCESS)
                state.episode_done = True
            if planned_steps:
                ctx.logger.warning(
                    "Controller terminal=%s dropped %s unexecuted planned steps",
                    terminal_signal,
                    len(planned_steps),
                )
            planned_steps.clear()
            break

        if controller_state == STATE_RUNNING and (not planned_steps):
            current_obs_raw = env.get_latest_obs()
            planned_steps = _plan_chunk(ctx, spec, state, current_obs_raw)
            queue_empty_since = None
            if planned_steps:
                continue
            if spec.phase_train and int(ctx.max_train_env_steps) > 0 and int(
                state.train_env_step
            ) >= int(ctx.max_train_env_steps):
                state.episode_done = True
                break

        if controller_state in {STATE_WAIT_READY, STATE_PAUSED}:
            current_obs_raw = env.get_latest_obs()
        time.sleep(poll_sec)

    if buffered_step is not None and not state.episode_done:
        _flush_buffered_step(
            ctx,
            spec,
            state,
            buffered=buffered_step,
            update_train_progress_fn=update_train_progress_fn,
            advance_async_update_calls_fn=advance_async_update_calls_fn,
            maybe_send_agentlace_timer_stats_fn=maybe_send_agentlace_timer_stats_fn,
            maybe_wait_for_async_learner_budget_fn=maybe_wait_for_async_learner_budget_fn,
            save_checkpoint_at_step_fn=save_checkpoint_at_step_fn,
            timer_train_episode_id=int(timer_train_episode_id),
        )

    flush_episode_step_window(ctx, state)
    return build_episode_result(state)


def run_agibot_controller_actor_loop(
    cfg: DictConfig,
    *,
    run_dir: Path,
    logger: logging.Logger,
    bindings: Any,
    async_eval_watcher_path: Path,
) -> None:
    ctx, state = build_actor_runtime_session(
        cfg,
        run_dir=run_dir,
        logger=logger,
        bindings=bindings,
        async_eval_watcher_path=async_eval_watcher_path,
    )
    _validate_controller_runtime(ctx)

    env = ctx.env
    step_logger = ctx.step_logger
    episode_logger = ctx.episode_logger
    tb_writer = ctx.tb_writer
    profiler = ctx.profiler
    profiling_logger = ctx.profiling_logger
    policy_prefetcher = ctx.policy_prefetcher
    sync_replay_prefetcher = ctx.sync_replay_prefetcher
    checkpoint_writer = ctx.checkpoint_writer
    async_learner = ctx.async_learner
    phase_progress = None

    def _update_train_progress(*, force_postfix: bool = False, train_env_step_value=None) -> None:
        if train_env_step_value is not None:
            state.train_env_step = int(train_env_step_value)
        update_train_progress(ctx, state, force_postfix=force_postfix)

    def _save_checkpoint(step_value: int) -> Path:
        return save_checkpoint_at_step(ctx, state, int(step_value))

    def _maybe_send_agentlace_timer_stats(
        *,
        train_env_step_value: int,
        decision_step_value: int,
        train_episode_id_value: int,
        force: bool = False,
    ) -> None:
        state.train_env_step = int(train_env_step_value)
        state.decision_step = int(decision_step_value)
        send_agentlace_timer_stats(
            ctx,
            state,
            train_episode_id_value=int(train_episode_id_value),
            force=bool(force),
        )

    def _maybe_wait_for_async_learner_budget(
        *,
        train_env_step_value: int,
        decision_step_value: int,
    ) -> None:
        state.train_env_step = int(train_env_step_value)
        state.decision_step = int(decision_step_value)
        wait_for_async_learner_budget(ctx, state)

    def _advance_async_update_calls(
        *,
        phase_train_flag: bool,
        train_step_before: int,
        train_step_after: int,
        replay_size_before: int,
        replay_size_after: int,
    ) -> None:
        state.train_env_step = int(train_step_after)
        advance_async_update_calls_impl(
            ctx,
            phase_train_flag=bool(phase_train_flag),
            train_step_before=int(train_step_before),
            train_step_after=int(train_step_after),
            replay_size_before=int(replay_size_before),
            replay_size_after=int(replay_size_after),
        )

    try:
        sync_async_bounded_lag_baseline(ctx)
        for phase in cfg.training.phases:
            if int(ctx.max_train_env_steps) > 0 and int(state.train_env_step) >= int(
                ctx.max_train_env_steps
            ):
                break
            phase_name = str(phase.name)
            phase_episodes = int(phase.episodes)
            logger.info(
                "Start controller phase=%s episodes=%s",
                phase_name,
                phase_episodes,
            )
            phase_progress = new_progress(
                ctx,
                desc=f"{phase_name}:episode",
                total=int(phase_episodes),
                position=1,
                leave=False,
            )
            phase_episode_count = 0
            try:
                while phase_episode_count < phase_episodes:
                    if int(ctx.max_train_env_steps) > 0 and int(state.train_env_step) >= int(
                        ctx.max_train_env_steps
                    ):
                        break

                    seed = int(state.seed_cursor)
                    state.seed_cursor += 1
                    current_phase_episode_idx = int(phase_episode_count + 1)
                    current_train_episode_id = int(state.train_episode_id + 1)
                    current_init_episode_idx = int(state.init_episode_idx)

                    if bool(cfg.training.get("expert_check", False)):
                        passed, _ = env.expert_precheck(
                            seed=seed, init_episode_idx=current_init_episode_idx
                        )
                        if not passed:
                            state.skipped_seeds += 1
                            logger.warning(
                                "skip seed=%s in controller phase=%s: expert precheck failed",
                                seed,
                                phase_name,
                            )
                            continue

                    state.init_episode_idx += 1
                    clear_obs_cache(ctx)
                    obs_raw = env.reset(seed=seed, init_episode_idx=current_init_episode_idx)
                    episode_result = _run_controller_episode(
                        ctx,
                        EpisodeSpec(
                            phase_name=str(phase_name),
                            phase_train=True,
                            phase_episode_idx=int(current_phase_episode_idx),
                            train_episode_id=int(current_train_episode_id),
                            seed=int(seed),
                            init_episode_idx=int(current_init_episode_idx),
                            max_episode_steps=int(env.step_limit),
                        ),
                        obs_raw,
                        agent=ctx.agent,
                        learner_agent=ctx.learner_agent,
                        async_learner=ctx.async_learner,
                        replay_buffer=ctx.replay_buffer,
                        sync_replay_lock=ctx.sync_replay_lock,
                        sync_replay_prefetcher=ctx.sync_replay_prefetcher,
                        train_env_step=int(state.train_env_step),
                        decision_step=int(state.decision_step),
                        last_update_info=dict(state.last_update_info),
                        profiling_last_flush_step=int(state.profiling_last_flush_step),
                        update_train_progress_fn=_update_train_progress,
                        advance_async_update_calls_fn=_advance_async_update_calls,
                        maybe_send_agentlace_timer_stats_fn=_maybe_send_agentlace_timer_stats,
                        maybe_wait_for_async_learner_budget_fn=_maybe_wait_for_async_learner_budget,
                        save_checkpoint_at_step_fn=_save_checkpoint,
                        timer_train_episode_id=int(state.train_episode_id),
                    )

                    state.episode_success = bool(episode_result.episode_success)
                    state.train_env_step = int(episode_result.train_env_step)
                    state.decision_step = int(episode_result.decision_step)
                    state.last_update_info = dict(episode_result.last_update_info)
                    ctx.agent = episode_result.agent
                    ctx.learner_agent = episode_result.learner_agent
                    ctx.async_learner = episode_result.async_learner
                    ctx.replay_buffer = episode_result.replay_buffer
                    ctx.sync_replay_lock = episode_result.sync_replay_lock
                    ctx.sync_replay_prefetcher = episode_result.sync_replay_prefetcher
                    state.profiling_last_flush_step = int(
                        episode_result.profiling_last_flush_step
                    )
                    state.train_total_success += int(episode_result.episode_success)
                    state.train_recent_successes.append(int(episode_result.episode_success))

                    running_success_rate = float(state.train_total_success) / float(
                        current_train_episode_id
                    )
                    recent_success_rate = float(sum(state.train_recent_successes)) / float(
                        len(state.train_recent_successes)
                    )
                    episode_logger.write(
                        {
                            "phase": phase_name,
                            "warmup_episode_id": None,
                            "train_episode_id": int(current_train_episode_id),
                            "phase_episode_idx": int(current_phase_episode_idx),
                            "seed": int(env.last_seed if env.last_seed is not None else seed),
                            "init_state_idx": (
                                int(env.current_init_state_idx)
                                if env.current_init_state_idx is not None
                                else None
                            ),
                            "success": bool(episode_result.episode_success),
                            "episode_steps": int(episode_result.episode_steps),
                            "episode_return": float(episode_result.episode_return),
                            "train_env_step": int(state.train_env_step),
                            "decision_step": int(state.decision_step),
                            "running_success_rate": running_success_rate,
                            "recent_success_rate": recent_success_rate,
                        }
                    )
                    tb_writer.add_scalar(
                        "train_episode/success",
                        int(episode_result.episode_success),
                        int(current_train_episode_id),
                    )
                    tb_writer.add_scalar(
                        "train_episode/return",
                        float(episode_result.episode_return),
                        int(current_train_episode_id),
                    )
                    tb_writer.add_scalar(
                        "train_episode/length",
                        int(episode_result.episode_steps),
                        int(current_train_episode_id),
                    )
                    tb_writer.add_scalar(
                        "train_episode/running_success_rate",
                        running_success_rate,
                        int(current_train_episode_id),
                    )
                    tb_writer.add_scalar(
                        "train_episode/recent_success_rate_20",
                        recent_success_rate,
                        int(current_train_episode_id),
                    )

                    logger.info(
                        "controller phase=%s train_episode=%s success=%s steps=%s return=%.2f "
                        "train_env_step=%s success_rate=%.3f recent=%.3f",
                        phase_name,
                        current_train_episode_id,
                        bool(episode_result.episode_success),
                        int(episode_result.episode_steps),
                        float(episode_result.episode_return),
                        int(state.train_env_step),
                        running_success_rate,
                        recent_success_rate,
                    )
                    if bool(ctx.external_agentlace_actor_mode) and ctx.async_learner is not None:
                        _maybe_send_agentlace_timer_stats(
                            train_env_step_value=int(state.train_env_step),
                            decision_step_value=int(state.decision_step),
                            train_episode_id_value=int(current_train_episode_id),
                            force=True,
                        )
                    state.train_episode_id = int(current_train_episode_id)
                    _sync_async_eval_results_to_tb(
                        tb_writer,
                        summary_jsonl_path=ctx.async_eval_summary_path,
                        sync_state=ctx.async_eval_tb_sync_state,
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

        if ctx.async_learner is not None:
            ctx.async_learner.stop()
            state.last_update_info = ctx.async_learner.get_last_update_info()
        if checkpoint_writer is not None:
            checkpoint_writer.close(wait=True)
            checkpoint_writer = None

        final_profiling_payload = _emit_profiling_snapshot(
            profiler,
            profile_logger=profiling_logger,
            tb_writer=tb_writer,
            logger=logger,
            train_env_step=int(state.train_env_step),
            decision_step=int(state.decision_step),
            train_episode_id=int(state.train_episode_id),
            learner_update_steps=int(ctx.async_learner.get_update_steps())
            if ctx.async_learner is not None
            else 0,
            replay_prefetch_queue_size=(
                int(ctx.async_learner.get_prefetch_queue_size())
                if ctx.async_learner is not None
                else int(sync_replay_prefetcher.get_queue_size())
                if sync_replay_prefetcher is not None
                else 0
            ),
        )
        summary = {
            "runtime_mode": "controller",
            "train_env_step": int(state.train_env_step),
            "decision_step": int(state.decision_step),
            "train_episode_id": int(state.train_episode_id),
            "train_total_success": int(state.train_total_success),
            "train_success_rate": float(
                state.train_total_success / max(1, int(state.train_episode_id))
            ),
            "skipped_seeds": int(state.skipped_seeds),
            "seed_start": int(cfg.task.seed_base),
            "seed_next": int(state.seed_cursor),
            "replay_size": int(len(ctx.replay_buffer) if ctx.replay_buffer is not None else 0),
            "last_update_info": _to_jsonable(state.last_update_info),
            "offline_stats": _to_jsonable(ctx.offline_stats),
            "online_prefill_stats": _to_jsonable(ctx.online_prefill_stats),
            "profiling": {
                "enabled": bool(ctx.profiling_enabled),
                "snapshot": (
                    _to_jsonable(final_profiling_payload.get("metrics", {}))
                    if final_profiling_payload is not None
                    else {}
                ),
            },
        }
        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("controller training done: %s", summary)
    finally:
        if ctx.async_learner is not None:
            ctx.async_learner.stop()
        if checkpoint_writer is not None:
            checkpoint_writer.close(wait=True)
        if sync_replay_prefetcher is not None:
            sync_replay_prefetcher.stop()
        if policy_prefetcher is not None:
            policy_prefetcher.close()
        if phase_progress is not None:
            phase_progress.close()
        try:
            flush_external_agentlace_actor(ctx)
        except Exception:  # noqa: BLE001
            pass
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        step_logger.close()
        episode_logger.close()
        if profiling_logger is not None:
            profiling_logger.close()
        tb_writer.close()
