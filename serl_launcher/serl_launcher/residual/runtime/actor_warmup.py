from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Dict, Optional, Tuple

import numpy as np

from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.runtime.actor_support import ActorLoopState
from serl_launcher.residual.runtime.actor_support import ActorRuntimeContext
from serl_launcher.residual.runtime.actor_support import build_chunk_step_record
from serl_launcher.residual.runtime.actor_support import build_policy_input
from serl_launcher.residual.runtime.actor_support import ensure_training_runtime_started
from serl_launcher.residual.runtime.actor_support import flush_external_agentlace_actor
from serl_launcher.residual.runtime.actor_support import new_progress
from serl_launcher.residual.runtime.obs_utils import _clone_obs_dict
from serl_launcher.residual.runtime.obs_utils import _zero_obs_like
from serl_launcher.residual.runtime.profiling import _profile_call
from serl_launcher.residual.runtime.train_loop_utils import _insert_online_transition


def run_base_only_warmup(ctx: ActorRuntimeContext, state: ActorLoopState) -> None:
    if not ctx.need_warmup_first:
        ensure_training_runtime_started(ctx)
        return

    if ctx.online_prefill_loaded_episodes > 0:
        ctx.logger.info(
            "Warmup phase: collecting remaining %s/%s base-only episodes after loading "
            "online prefill, no actor/critic updates",
            ctx.warmup_episodes_cfg,
            ctx.configured_warmup_episodes,
        )
    else:
        ctx.logger.info(
            "Warmup phase: collecting %s base-only episodes, no actor/critic updates",
            ctx.warmup_episodes_cfg,
        )

    state.warmup_progress = new_progress(
        ctx,
        desc="warmup_episode",
        total=int(ctx.warmup_episodes_cfg),
        position=1,
        leave=False,
    )

    for _ in range(int(ctx.warmup_episodes_cfg)):
        current_warmup_episode_id = int(state.warmup_episode_id + 1)
        current_init_episode_idx = int(state.init_episode_idx)
        state.init_episode_idx += 1

        seed = int(state.seed_cursor)
        state.seed_cursor += 1
        ctx.obs_cache.clear()
        obs_raw = _profile_call(
            ctx.profiler,
            "env_reset",
            ctx.env.reset,
            seed=seed,
            init_episode_idx=current_init_episode_idx,
        )
        max_episode_steps = int(ctx.env.step_limit)
        if ctx.cfg.training.max_env_steps_per_episode is not None:
            max_episode_steps = min(
                max_episode_steps,
                int(ctx.cfg.training.max_env_steps_per_episode),
            )
        episode_success = False
        episode_return = 0.0
        episode_steps = 0
        episode_done = False
        cached_base_chunk = None
        cached_infer_info = None

        while (episode_steps < max_episode_steps) and (not episode_done):
            if cached_base_chunk is None:
                policy_chunk, infer_info = ctx.policy_client.infer_chunk(
                    build_policy_input(ctx, obs_raw, ctx.env.current_instruction)
                )
                base_chunk = select_action_chunk_window(
                    policy_chunk,
                    horizon=ctx.chunk_horizon,
                    action_dim=ctx.env_action_dim,
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

            if ctx.chunk_step_enabled:
                alpha_step = 0.0
                execute_horizon = int(
                    min(ctx.chunk_horizon, max_episode_steps - episode_steps)
                )
                executed_base_chunk = np.asarray(
                    base_chunk[:execute_horizon], dtype=np.float32
                )
                chunk_result = _profile_call(
                    ctx.profiler,
                    "env_step_chunk",
                    ctx.env.step_chunk,
                    executed_base_chunk,
                )
                chunk_observations = list(chunk_result["observations"])
                next_obs_raw = chunk_result["obs"]
                chunk_rewards = [float(v) for v in chunk_result["rewards"]]
                chunk_infos = [dict(v) for v in chunk_result["infos"]]
                chunk_dones = [bool(v) for v in chunk_result["dones"]]
                actual_chunk_steps = int(len(chunk_rewards))
                if actual_chunk_steps <= 0:
                    raise RuntimeError("Warmup chunk execution returned zero steps")
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
                    ctx.step_logger.write(
                        {
                            "train_env_step": None,
                            "decision_step": None,
                            "warmup_episode_id": current_warmup_episode_id,
                            "train_episode_id": None,
                            "phase_episode_idx": current_warmup_episode_id,
                            "phase": "warmup_base_only",
                            "episode_step": episode_steps,
                            "seed": int(
                                ctx.env.last_seed
                                if ctx.env.last_seed is not None
                                else seed
                            ),
                            "init_state_idx": (
                                int(ctx.env.current_init_state_idx)
                                if ctx.env.current_init_state_idx is not None
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
                                (ctx.step_action_dim,), dtype=np.float32
                            ).tolist(),
                            "a_res_policy_applied": np.zeros(
                                (ctx.step_action_dim,), dtype=np.float32
                            ).tolist(),
                            "a_res": np.zeros(
                                (ctx.env_action_dim,), dtype=np.float32
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
                    step_payload = build_chunk_step_record(
                        ctx,
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
                        ctx.replay_buffer,
                        step_payload,
                        chunk_step_enabled=ctx.chunk_step_enabled,
                    )
                    if chunk_step < (actual_chunk_steps - 1):
                        current_step_obs_raw = chunk_observations[chunk_step]
                    if done:
                        episode_done = True
                        break

                if not done:
                    next_policy_chunk, next_infer_info = ctx.policy_client.infer_chunk(
                        build_policy_input(ctx, next_obs_raw, ctx.env.current_instruction)
                    )
                    next_base_chunk = select_action_chunk_window(
                        next_policy_chunk,
                        horizon=ctx.chunk_horizon,
                        action_dim=ctx.env_action_dim,
                    )
                    cached_base_chunk = next_base_chunk
                    cached_infer_info = next_infer_info
                obs_raw = next_obs_raw
                continue

            next_obs_raw = obs_raw
            for chunk_step in range(ctx.chunk_horizon):
                if episode_steps >= max_episode_steps:
                    episode_done = True
                    break
                alpha_step = 0.0
                obs_input = ctx.build_residual_step_obs_profiled(
                    ctx.profiler,
                    next_obs_raw,
                    base_chunk[chunk_step],
                    image_keys=ctx.image_keys,
                    stack_horizon=ctx.stack_horizon,
                    normalizer=ctx.normalizer,
                    obs_cache=ctx.obs_cache,
                    alpha=float(alpha_step),
                    state_mode=ctx.obs_state_mode,
                )
                residual_step_action = np.zeros(
                    (ctx.step_action_dim,), dtype=np.float32
                )
                final_action = base_chunk[chunk_step].copy()
                next_obs_raw, reward, env_done, _, info = _profile_call(
                    ctx.profiler,
                    "env_step",
                    ctx.env.step,
                    final_action,
                )
                episode_steps += 1
                episode_return += float(reward)
                episode_success = bool(info["success"])
                timeout = bool(episode_steps >= max_episode_steps)
                done = bool(env_done or timeout)
                next_alpha_step = 0.0
                next_chunk_future: Optional[
                    Future[Tuple[np.ndarray, Dict[str, Optional[float]]]]
                ] = None
                if (
                    (not done)
                    and chunk_step == (ctx.chunk_horizon - 1)
                    and ctx.policy_prefetcher is not None
                ):
                    next_chunk_future = ctx.policy_prefetcher.submit(
                        build_policy_input(ctx, next_obs_raw, ctx.env.current_instruction)
                    )
                if done:
                    next_obs_input = _zero_obs_like(obs_input)
                    mask = 0.0
                elif chunk_step < (ctx.chunk_horizon - 1):
                    next_obs_input = ctx.build_residual_step_obs_profiled(
                        ctx.profiler,
                        next_obs_raw,
                        base_chunk[chunk_step + 1],
                        image_keys=ctx.image_keys,
                        stack_horizon=ctx.stack_horizon,
                        normalizer=ctx.normalizer,
                        obs_cache=ctx.obs_cache,
                        alpha=float(next_alpha_step),
                        state_mode=ctx.obs_state_mode,
                    )
                    mask = 1.0
                else:
                    if next_chunk_future is not None:
                        next_policy_chunk, next_infer_info = next_chunk_future.result()
                    else:
                        next_policy_chunk, next_infer_info = ctx.policy_client.infer_chunk(
                            build_policy_input(
                                ctx, next_obs_raw, ctx.env.current_instruction
                            )
                        )
                    next_base_chunk = select_action_chunk_window(
                        next_policy_chunk,
                        horizon=ctx.chunk_horizon,
                        action_dim=ctx.env_action_dim,
                    )
                    next_obs_input = ctx.build_residual_step_obs_profiled(
                        ctx.profiler,
                        next_obs_raw,
                        next_base_chunk[0],
                        image_keys=ctx.image_keys,
                        stack_horizon=ctx.stack_horizon,
                        normalizer=ctx.normalizer,
                        obs_cache=ctx.obs_cache,
                        alpha=float(next_alpha_step),
                        state_mode=ctx.obs_state_mode,
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
                    ctx.replay_buffer,
                    transition_payload,
                    chunk_step_enabled=ctx.chunk_step_enabled,
                )
                ctx.step_logger.write(
                    {
                        "train_env_step": None,
                        "decision_step": None,
                        "warmup_episode_id": current_warmup_episode_id,
                        "train_episode_id": None,
                        "phase_episode_idx": current_warmup_episode_id,
                        "phase": "warmup_base_only",
                        "episode_step": episode_steps,
                        "seed": int(
                            ctx.env.last_seed if ctx.env.last_seed is not None else seed
                        ),
                        "init_state_idx": (
                            int(ctx.env.current_init_state_idx)
                            if ctx.env.current_init_state_idx is not None
                            else None
                        ),
                        "is_warmup": True,
                        "is_probing": False,
                        "replan_point": bool(chunk_step == 0),
                        "chunk_step": int(chunk_step),
                        "chunk_horizon": int(ctx.chunk_horizon),
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
                            (ctx.step_action_dim,), dtype=np.float32
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

        state.warmup_total_success += int(episode_success)
        state.warmup_recent_successes.append(int(episode_success))
        warmup_running_success_rate = float(state.warmup_total_success) / float(
            current_warmup_episode_id
        )
        warmup_recent_success_rate = float(sum(state.warmup_recent_successes)) / float(
            len(state.warmup_recent_successes)
        )
        ctx.episode_logger.write(
            {
                "phase": "warmup_base_only",
                "warmup_episode_id": current_warmup_episode_id,
                "train_episode_id": None,
                "phase_episode_idx": current_warmup_episode_id,
                "seed": int(ctx.env.last_seed if ctx.env.last_seed is not None else seed),
                "init_state_idx": (
                    int(ctx.env.current_init_state_idx)
                    if ctx.env.current_init_state_idx is not None
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
        ctx.tb_writer.add_scalar(
            "warmup/episode/success",
            int(episode_success),
            current_warmup_episode_id,
        )
        ctx.tb_writer.add_scalar(
            "warmup/episode/return",
            float(episode_return),
            current_warmup_episode_id,
        )
        ctx.tb_writer.add_scalar(
            "warmup/episode/length",
            int(episode_steps),
            current_warmup_episode_id,
        )
        ctx.tb_writer.add_scalar(
            "warmup/episode/running_success_rate",
            warmup_running_success_rate,
            current_warmup_episode_id,
        )
        ctx.tb_writer.add_scalar(
            "warmup/episode/recent_success_rate_20",
            warmup_recent_success_rate,
            current_warmup_episode_id,
        )
        ctx.tb_writer.add_scalar(
            "warmup/system/online_buffer_size",
            int(len(ctx.replay_buffer)),
            current_warmup_episode_id,
        )
        ctx.logger.info(
            "warmup episode %s/%s success=%s steps=%s return=%.2f",
            current_warmup_episode_id,
            ctx.configured_warmup_episodes,
            episode_success,
            episode_steps,
            episode_return,
        )
        flush_external_agentlace_actor(ctx)
        state.warmup_episode_id = current_warmup_episode_id
        if state.warmup_progress is not None:
            state.warmup_progress.update(1)
            state.warmup_progress.set_postfix(
                {"success": int(episode_success)},
                refresh=False,
            )

    ctx.logger.info(
        "Warmup complete. Warmup episodes=%s total_success=%s buffer_size=%s. "
        "Starting residual training phase.",
        state.warmup_episode_id,
        state.warmup_total_success,
        len(ctx.replay_buffer),
    )
    if state.warmup_progress is not None:
        state.warmup_progress.close()
        state.warmup_progress = None

    ensure_training_runtime_started(ctx)
