from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from concurrent.futures import Future

from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.action import as_numpy_action_chunk
from serl_launcher.residual.action import compose_residual_action
from serl_launcher.residual.action import compose_residual_action_chunk
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.runtime.actor_support import build_chunk_step_record
from serl_launcher.residual.runtime.actor_support import build_policy_input
from serl_launcher.residual.runtime.actor_support import build_step_obs_profiled
from serl_launcher.residual.runtime.actor_support import replay_progress_size
from serl_launcher.residual.runtime.actor_support import resolve_train_gate
from serl_launcher.residual.runtime.config_utils import sample_probing_steps
from serl_launcher.residual.runtime.obs_utils import _clone_obs_dict
from serl_launcher.residual.runtime.obs_utils import _zero_obs_like
from serl_launcher.residual.runtime.profiling import _emit_profiling_snapshot
from serl_launcher.residual.runtime.profiling import _profile_call
from serl_launcher.residual.runtime.replay_batch import _consume_prepared_replay_batch
from serl_launcher.residual.runtime.replay_batch import _prepare_replay_batch
from serl_launcher.residual.runtime.schedules import _scheduled_alpha
from serl_launcher.residual.runtime.tb_metrics import _append_tb_step_window
from serl_launcher.residual.runtime.tb_metrics import _flush_tb_step_window
from serl_launcher.residual.runtime.tb_metrics import _log_update_metrics
from serl_launcher.residual.runtime.train_loop_utils import _count_env_step_update_triggers
from serl_launcher.residual.runtime.train_loop_utils import _insert_online_transition
from serl_launcher.residual.runtime.train_loop_utils import _iter_period_hits
from serl_launcher.residual.runtime.train_loop_utils import _remaining_train_budget_steps


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
    env = ctx.env
    policy_client = ctx.policy_client
    policy_prefetcher = ctx.policy_prefetcher
    algorithm = ctx.algorithm
    profiler = ctx.profiler
    logger = ctx.logger
    tb_writer = ctx.tb_writer
    step_logger = ctx.step_logger
    step_metric_window = ctx.step_metric_window

    chunk_horizon = int(ctx.chunk_horizon)
    env_action_dim = int(ctx.env_action_dim)
    control_indices = ctx.control_indices
    step_action_dim = int(ctx.step_action_dim)
    residual_limits = ctx.residual_limits
    residual_alpha = float(ctx.residual_alpha)
    chunk_step_enabled = bool(ctx.chunk_step_enabled)
    chunk_step_sample_stride = int(ctx.chunk_step_sample_stride)
    chunk_step_require_full_horizon = bool(ctx.chunk_step_require_full_horizon)
    chunk_step_pad_action = bool(ctx.chunk_step_pad_action)
    chunk_step_scheduler_clock = str(ctx.chunk_step_scheduler_clock)
    agent_action_dim = int(ctx.agent_action_dim)
    checkpoint_every_steps = int(ctx.checkpoint_every_steps)
    offline_enabled = bool(ctx.offline_enabled)
    offline_ratio = float(ctx.offline_ratio)
    symmetric_replay = bool(ctx.symmetric_replay)
    async_idle_sleep_sec = float(ctx.async_idle_sleep_sec)
    replay_prefetch_pin_memory = bool(ctx.replay_prefetch_pin_memory)
    replay_prefetch_to_device = bool(ctx.replay_prefetch_to_device)
    tb_step_period = int(ctx.tb_step_period)
    tb_histogram_period = int(ctx.tb_histogram_period)
    profiling_enabled = bool(ctx.profiling_enabled)
    profiling_log_period_steps = int(ctx.profiling_log_period_steps)
    max_train_env_steps = int(ctx.max_train_env_steps)
    async_bounded_lag_enabled = bool(ctx.async_bounded_lag_enabled)
    async_bounded_lag_env_steps_per_update_call = (
        ctx.async_bounded_lag_env_steps_per_update_call
    )
    agentlace_bridge_state = ctx.agentlace_bridge_state
    cfg = ctx.cfg
    episode_success = False
    episode_return = 0.0
    episode_steps = 0
    episode_done = False
    cached_base_chunk = None
    cached_infer_info = None

    probing_steps_target = sample_probing_steps(
        cfg.training,
        episode_horizon=int(spec.max_episode_steps),
    )
    if probing_steps_target > 0:
        probing_remaining = int(min(probing_steps_target, spec.max_episode_steps))
        probe_future: Optional[Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]] = (
            None
        )
        while probing_remaining > 0 and episode_steps < spec.max_episode_steps:
            if probe_future is not None:
                probe_chunk, probe_info = probe_future.result()
                probe_future = None
            else:
                probe_chunk, probe_info = policy_client.infer_chunk(
                    build_policy_input(ctx, obs_raw, env.current_instruction)
                )
            probe_base_chunk = select_action_chunk_window(
                probe_chunk,
                horizon=chunk_horizon,
                action_dim=env_action_dim,
            )
            for probe_step in range(chunk_horizon):
                if probing_remaining <= 0 or episode_steps >= spec.max_episode_steps:
                    break
                base_action = probe_base_chunk[probe_step]
                next_obs_raw, reward, env_done, _, info = _profile_call(
                    profiler,
                    "env_step",
                    env.step,
                    base_action,
                )
                episode_steps += 1
                if spec.phase_train:
                    train_env_step += 1
                    update_train_progress()
                probing_remaining -= 1
                episode_return += float(reward)
                episode_success = bool(info["success"])
                timeout = bool(episode_steps >= spec.max_episode_steps)
                budget_exhausted = bool(
                    spec.phase_train
                    and max_train_env_steps > 0
                    and train_env_step >= max_train_env_steps
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
                        "train_env_step": int(train_env_step) if spec.phase_train else None,
                        "decision_step": int(decision_step) if spec.phase_train else None,
                        "warmup_episode_id": None,
                        "train_episode_id": spec.train_episode_id,
                        "phase_episode_idx": int(spec.phase_episode_idx),
                        "phase": str(spec.phase_name),
                        "episode_step": episode_steps,
                        "seed": int(env.last_seed if env.last_seed is not None else spec.seed),
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
                        "a_res_policy": [0.0] * step_action_dim,
                        "a_res_policy_applied": [0.0] * step_action_dim,
                        "a_res": np.zeros_like(base_action, dtype=np.float32).tolist(),
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

    while episode_steps < spec.max_episode_steps and not episode_done:
        if cached_base_chunk is None:
            policy_chunk, infer_info = policy_client.infer_chunk(
                build_policy_input(ctx, obs_raw, env.current_instruction)
            )
            base_chunk = select_action_chunk_window(
                policy_chunk,
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
            obs_input = build_step_obs_profiled(
                ctx,
                obs_raw,
                base_chunk[0],
                base_action_chunk=base_chunk,
                alpha=float(alpha_step),
            )
            gate_prob, gate_on = resolve_train_gate(
                ctx,
                phase_train_flag=bool(spec.phase_train),
                alpha_value=float(alpha_step),
                env_step_value=int(train_env_step_before_chunk),
                decision_step_value=int(decision_step),
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
                if async_learner is not None:
                    sampled_chunk = async_learner.sample_actor_action(
                        obs_input,
                        agent_action_dim,
                    )
                else:
                    sample_actions_start = time.perf_counter()
                    sampled = algorithm.sample_actions(
                        agent,
                        obs_input,
                        deterministic=False,
                    )
                    profiler.record_duration(
                        "agent_sample_actions",
                        (time.perf_counter() - sample_actions_start) * 1000.0,
                    )
                    sampled_chunk = as_numpy_action(sampled, agent_action_dim)
                residual_chunk = as_numpy_action_chunk(
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
                    train_env_step=train_env_step,
                )
                if spec.phase_train
                else None
            )
            execute_horizon = int(min(chunk_horizon, spec.max_episode_steps - episode_steps))
            if remaining_budget_steps is not None:
                execute_horizon = int(min(execute_horizon, remaining_budget_steps))
            if spec.phase_train and train_env_step_before_chunk < int(cfg.training.random_steps):
                execute_horizon = int(
                    min(
                        execute_horizon,
                        int(cfg.training.random_steps) - train_env_step_before_chunk,
                    )
                )
            if execute_horizon <= 0:
                episode_done = True
                break

            current_decision_id = int(decision_step + 1) if spec.phase_train else None
            replay_size_before = int(replay_progress_size(replay_buffer))
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
            current_step_obs_raw = obs_raw
            chunk_step_payloads = []
            for chunk_step in range(actual_chunk_steps):
                current_episode_step = int(episode_steps)
                reward = float(chunk_rewards[chunk_step])
                info = chunk_infos[chunk_step]
                episode_steps += 1
                if spec.phase_train:
                    train_env_step += 1
                    update_train_progress()
                episode_return += reward
                episode_success = bool(info.get("success", episode_success))

                timeout = bool(episode_steps >= spec.max_episode_steps)
                remaining_budget_steps = (
                    _remaining_train_budget_steps(
                        max_train_env_steps=max_train_env_steps,
                        train_env_step=train_env_step,
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
                        "train_env_step": int(train_env_step) if spec.phase_train else None,
                        "decision_step": current_decision_id,
                        "warmup_episode_id": None,
                        "train_episode_id": spec.train_episode_id,
                        "phase_episode_idx": int(spec.phase_episode_idx),
                        "phase": str(spec.phase_name),
                        "episode_step": episode_steps,
                        "seed": int(env.last_seed if env.last_seed is not None else spec.seed),
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
                        "infer_policy_ms": infer_info.get("policy_ms") if chunk_step == 0 else None,
                        "infer_server_ms": infer_info.get("server_ms") if chunk_step == 0 else None,
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
                        "success": bool(episode_success),
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
                if spec.phase_train and train_env_step % tb_step_period == 0:
                    _flush_tb_step_window(
                        tb_writer,
                        step_window=step_metric_window,
                        global_env_step=train_env_step,
                        control_indices=control_indices,
                        histogram=bool(train_env_step % tb_histogram_period == 0),
                    )

                if chunk_step < (actual_chunk_steps - 1):
                    current_step_obs_raw = chunk_observations[chunk_step]
                if done:
                    episode_done = True
                    break

            train_env_step_after_chunk = int(train_env_step)
            if not done:
                next_policy_chunk, next_infer_info = policy_client.infer_chunk(
                    build_policy_input(ctx, next_obs_raw, env.current_instruction)
                )
                next_base_chunk = select_action_chunk_window(
                    next_policy_chunk,
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

            replay_size_after = int(replay_progress_size(replay_buffer))
            if async_learner is None:
                if spec.phase_train:
                    trigger_count = _count_env_step_update_triggers(
                        train_step_before=train_env_step_before_chunk,
                        train_step_after=train_env_step_after_chunk,
                        replay_size_before=replay_size_before,
                        replay_size_after=replay_size_after,
                        training_starts=int(cfg.training.training_starts),
                        update_every=int(cfg.training.update_every),
                    )
                    for _ in range(
                        int(trigger_count * int(cfg.training.updates_per_step))
                    ):
                        if sync_replay_prefetcher is not None:
                            sampled_batch = sync_replay_prefetcher.get(
                                timeout=ctx.async_idle_sleep_sec
                            )
                            if sampled_batch is None:
                                continue
                            batch, online_bs, offline_bs = sampled_batch
                        else:
                            replay_sample_start = time.perf_counter()
                            sampled = _sample_mixed_batch(
                                replay_buffer,
                                ctx.offline_buffer if offline_enabled else None,
                                batch_size=int(cfg.replay.batch_size),
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
                                device=learner_agent.device,
                                pin_memory=bool(ctx.replay_prefetch_pin_memory),
                                to_device=bool(ctx.replay_prefetch_to_device),
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
                        learner_agent, last_update_info = algorithm.update_high_utd(
                            learner_agent,
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
                advance_async_update_calls(
                    phase_train_flag=bool(spec.phase_train),
                    train_step_before=int(train_env_step_before_chunk),
                    train_step_after=int(train_env_step_after_chunk),
                    replay_size_before=int(replay_size_before),
                    replay_size_after=int(replay_size_after),
                )
                maybe_wait_for_async_learner_budget(
                    train_env_step_value=int(train_env_step_after_chunk),
                    decision_step_value=int(
                        current_decision_id
                        if current_decision_id is not None
                        else decision_step
                    ),
                )
                last_update_info = async_learner.get_last_update_info()

            if spec.phase_train and train_env_step % tb_step_period == 0 and last_update_info:
                _log_update_metrics(tb_writer, last_update_info, train_env_step)
                tb_writer.add_scalar(
                    "system/online_buffer_size",
                    int(len(replay_buffer)),
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
                            float(agentlace_bridge_state.target_update_calls),
                            train_env_step,
                        )
                        tb_writer.add_scalar(
                            "system/async_bounded_lag_tracked_env_steps",
                            float(agentlace_bridge_state.tracked_env_steps),
                            train_env_step,
                        )
                        if async_bounded_lag_env_steps_per_update_call is not None:
                            tb_writer.add_scalar(
                                "system/async_env_steps_per_update_call",
                                float(async_bounded_lag_env_steps_per_update_call),
                                train_env_step,
                            )
                        tb_writer.add_scalar(
                            "system/async_required_update_steps",
                            float(agentlace_bridge_state.last_required_update_steps),
                            train_env_step,
                        )
                        tb_writer.add_scalar(
                            "system/async_update_lag_before_wait",
                            float(agentlace_bridge_state.last_lag_before_wait),
                            train_env_step,
                        )
                        tb_writer.add_scalar(
                            "system/async_update_lag_after_wait",
                            float(agentlace_bridge_state.last_lag_after_wait),
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

            if spec.phase_train:
                decision_step = int(current_decision_id)

            if (
                profiling_enabled
                and profiling_log_period_steps > 0
                and train_env_step > 0
                and (train_env_step - profiling_last_flush_step) >= profiling_log_period_steps
            ):
                _emit_profiling_snapshot(
                    profiler,
                    profile_logger=ctx.profiling_logger,
                    tb_writer=tb_writer,
                    logger=logger,
                    train_env_step=train_env_step,
                    decision_step=decision_step,
                    train_episode_id=int(timer_train_episode_id),
                    learner_update_steps=(
                        int(async_learner.get_update_steps())
                        if async_learner is not None
                        else 0
                    ),
                    replay_prefetch_queue_size=(
                        int(async_learner.get_prefetch_queue_size())
                        if async_learner is not None
                        else int(sync_replay_prefetcher.get_queue_size())
                        if sync_replay_prefetcher is not None
                        else 0
                    ),
                )
                profiling_last_flush_step = int(train_env_step)
            maybe_send_agentlace_timer_stats(
                train_env_step_value=int(train_env_step),
                decision_step_value=int(decision_step),
                train_episode_id_value=int(timer_train_episode_id),
            )

            checkpoint_hits = _iter_period_hits(
                step_before=train_env_step_before_chunk,
                step_after=train_env_step_after_chunk,
                period=ctx.checkpoint_every_steps,
            )
            if spec.phase_train and checkpoint_hits:
                for checkpoint_step in checkpoint_hits:
                    save_checkpoint_at_step(int(checkpoint_step))

            obs_raw = next_obs_raw
            if episode_done:
                break
        else:
            next_obs_raw = obs_raw
            for chunk_step in range(chunk_horizon):
                if episode_steps >= spec.max_episode_steps:
                    episode_done = True
                    break

                train_env_step_before_step = int(train_env_step)
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
                    decision_step_value=int(decision_step),
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
                    if async_learner is not None:
                        residual_step_action = async_learner.sample_actor_action(
                            obs_input,
                            step_action_dim,
                        )
                    else:
                        sample_actions_start = time.perf_counter()
                        sampled = algorithm.sample_actions(
                            agent,
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

                current_decision_id = int(decision_step + 1) if spec.phase_train else None
                next_obs_raw, reward, env_done, _, info = _profile_call(
                    profiler,
                    "env_step",
                    env.step,
                    final_action,
                )
                episode_steps += 1
                if spec.phase_train:
                    train_env_step += 1
                    update_train_progress()
                train_env_step_after_step = int(train_env_step)
                next_alpha_step = _scheduled_alpha(
                    cfg,
                    base_alpha=residual_alpha,
                    schedule_step=train_env_step_after_step,
                )
                episode_return += float(reward)
                episode_success = bool(info["success"])
                timeout = bool(episode_steps >= spec.max_episode_steps)
                budget_exhausted = bool(
                    spec.phase_train
                    and max_train_env_steps > 0
                    and train_env_step >= max_train_env_steps
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
                        "train_env_step": int(train_env_step) if spec.phase_train else None,
                        "decision_step": current_decision_id,
                        "warmup_episode_id": None,
                        "train_episode_id": spec.train_episode_id,
                        "phase_episode_idx": int(spec.phase_episode_idx),
                        "phase": str(spec.phase_name),
                        "episode_step": episode_steps,
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
                if spec.phase_train and train_env_step % tb_step_period == 0:
                    _flush_tb_step_window(
                        tb_writer,
                        step_window=step_metric_window,
                        global_env_step=train_env_step,
                        control_indices=control_indices,
                        histogram=bool(train_env_step % tb_histogram_period == 0),
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
                    if next_chunk_future is not None:
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
                    "episode_id": int(spec.init_episode_idx),
                    "episode_step": int(episode_steps - 1),
                }
                replay_size_before = int(replay_progress_size(replay_buffer))
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
                replay_size_after = int(replay_progress_size(replay_buffer))

                if async_learner is None:
                    if (
                        spec.phase_train
                        and replay_progress_size(replay_buffer) >= int(cfg.training.training_starts)
                        and train_env_step_before_step % int(cfg.training.update_every) == 0
                    ):
                        for _ in range(int(cfg.training.updates_per_step)):
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
                                    ctx.offline_buffer if offline_enabled else None,
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
                            learner_agent, last_update_info = algorithm.update_high_utd(
                                learner_agent,
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
                    advance_async_update_calls(
                        phase_train_flag=bool(spec.phase_train),
                        train_step_before=int(train_env_step_before_step),
                        train_step_after=int(train_env_step_after_step),
                        replay_size_before=int(replay_size_before),
                        replay_size_after=int(replay_size_after),
                    )
                    maybe_wait_for_async_learner_budget(
                        train_env_step_value=int(train_env_step_after_step),
                        decision_step_value=int(
                            current_decision_id
                            if current_decision_id is not None
                            else decision_step
                        ),
                    )
                    last_update_info = async_learner.get_last_update_info()

                if spec.phase_train and train_env_step % tb_step_period == 0 and last_update_info:
                    _log_update_metrics(tb_writer, last_update_info, train_env_step)
                    tb_writer.add_scalar(
                        "system/online_buffer_size",
                        int(len(replay_buffer)),
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
                                float(agentlace_bridge_state.target_update_calls),
                                train_env_step,
                            )
                            tb_writer.add_scalar(
                                "system/async_bounded_lag_tracked_env_steps",
                                float(agentlace_bridge_state.tracked_env_steps),
                                train_env_step,
                            )
                            if async_bounded_lag_env_steps_per_update_call is not None:
                                tb_writer.add_scalar(
                                    "system/async_env_steps_per_update_call",
                                    float(async_bounded_lag_env_steps_per_update_call),
                                    train_env_step,
                                )
                            tb_writer.add_scalar(
                                "system/async_required_update_steps",
                                float(agentlace_bridge_state.last_required_update_steps),
                                train_env_step,
                            )
                            tb_writer.add_scalar(
                                "system/async_update_lag_before_wait",
                                float(agentlace_bridge_state.last_lag_before_wait),
                                train_env_step,
                            )
                            tb_writer.add_scalar(
                                "system/async_update_lag_after_wait",
                                float(agentlace_bridge_state.last_lag_after_wait),
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

                if spec.phase_train:
                    decision_step = int(current_decision_id)

                if (
                    profiling_enabled
                    and profiling_log_period_steps > 0
                    and train_env_step > 0
                    and (train_env_step - profiling_last_flush_step) >= profiling_log_period_steps
                ):
                    _emit_profiling_snapshot(
                        profiler,
                        profile_logger=ctx.profiling_logger,
                        tb_writer=tb_writer,
                        logger=logger,
                        train_env_step=train_env_step,
                        decision_step=decision_step,
                        train_episode_id=int(timer_train_episode_id),
                        learner_update_steps=(
                            int(async_learner.get_update_steps())
                            if async_learner is not None
                            else 0
                        ),
                        replay_prefetch_queue_size=(
                            int(async_learner.get_prefetch_queue_size())
                            if async_learner is not None
                            else int(sync_replay_prefetcher.get_queue_size())
                            if sync_replay_prefetcher is not None
                            else 0
                        ),
                    )
                    profiling_last_flush_step = int(train_env_step)
                maybe_send_agentlace_timer_stats(
                    train_env_step_value=int(train_env_step),
                    decision_step_value=int(decision_step),
                    train_episode_id_value=int(timer_train_episode_id),
                )

                if (
                    spec.phase_train
                    and checkpoint_every_steps > 0
                    and train_env_step_after_step % checkpoint_every_steps == 0
                ):
                    save_checkpoint_at_step(int(train_env_step_after_step))

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
            train_env_step > 0 and train_env_step % tb_histogram_period == 0
        ),
    )

    return EpisodeResult(
        episode_success=bool(episode_success),
        episode_return=float(episode_return),
        episode_steps=int(episode_steps),
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
