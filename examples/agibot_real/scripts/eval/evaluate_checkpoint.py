from __future__ import annotations

"""Evaluate an AgiBot residual checkpoint."""

import json
import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from serl_launcher.agents.continuous.drq_config import create_drq_agent_from_cfg
from serl_launcher.policy.factory import build_policy_backend_info
from serl_launcher.policy.factory import build_policy_client
from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.action import reshape_flat_action_to_chunk
from serl_launcher.residual.action import compose_residual_action
from serl_launcher.residual.action import compose_residual_action_chunk
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.train.config import build_residual_action_transform
from serl_launcher.residual.train.config import resolve_control_indices_from_cfg
from serl_launcher.residual.train.config import resolve_residual_observation_state_mode
from serl_launcher.residual.train.schedules import _epsilon_gating_enabled
from serl_launcher.residual.train.schedules import _epsilon_gating_eval_force_on
from serl_launcher.residual.train.schedules import _scheduled_epsilon_gating_probability
from serl_launcher.residual.utils.alpha_utils import require_residual_alpha
from serl_launcher.utils.checkpoint_utils import load_agent_checkpoint
from serl_launcher.utils.logger import JsonlLogger

REPO_PARENT = Path(__file__).resolve().parents[5]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.config import resolve_agibot_cfg_image_keys
from serl_torch.examples.agibot_real.env.factory import _create_env
from serl_torch.examples.agibot_real.runtime.controller_rollout import (
    ControllerExecutedStep,
)
from serl_torch.examples.agibot_real.runtime.controller_rollout import (
    ControllerPlannedStep,
)
from serl_torch.examples.agibot_real.runtime.controller_rollout import (
    require_controller_rollout_capability,
)
from serl_torch.examples.agibot_real.runtime.controller_rollout import (
    run_controller_episode,
)
from serl_torch.examples.agibot_real.runtime.obs_adapter import AgiBotObservationCache
from serl_torch.examples.agibot_real.runtime.obs_adapter import build_residual_step_obs
from serl_torch.examples.agibot_real.runtime.policy_adapter import (
    build_agibot_policy_input,
)


@hydra.main(
    version_base=None, config_path="../../conf", config_name="eval_residual_fast"
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("agibot_eval_residual_fast")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    env = _create_env(cfg, logger)
    logger.info(
        "AgiBot task: name=%s prompt=%s", cfg.task.name, env.current_instruction
    )

    obs_state_mode = resolve_residual_observation_state_mode(cfg)
    logger.info("Residual observation state_mode=%s", obs_state_mode)

    policy_backend_info = build_policy_backend_info(cfg)
    policy_client = build_policy_client(cfg, logger=logger)
    logger.info(
        "Chunk policy backend: type=%s id=%s",
        policy_backend_info["type"],
        policy_backend_info["id"],
    )

    image_keys = resolve_agibot_cfg_image_keys(cfg)
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    env_action_dim = int(cfg.env.action_dim)
    control_indices = resolve_control_indices_from_cfg(
        cfg, full_action_dim=env_action_dim
    )
    step_action_dim = int(len(control_indices))
    chunk_horizon = int(cfg.residual.chunk_horizon)
    residual_alpha = require_residual_alpha(cfg.get("residual", None))
    epsilon_gating_enabled = _epsilon_gating_enabled(cfg)
    epsilon_gating_eval_force_on = _epsilon_gating_eval_force_on(cfg)
    chunk_step_enabled = bool(cfg.get("chunk_step", {}).get("enabled", False))
    controller_enabled = bool(getattr(env, "controller_enabled", False))
    if controller_enabled:
        require_controller_rollout_capability(
            env=env,
            chunk_step_enabled=chunk_step_enabled,
            script_name="evaluate_checkpoint",
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
    residual_limits = build_residual_limits(
        control_indices,
        action_limits=cfg.residual.get("action_limits", None),
        full_action_dim=env_action_dim,
    )
    action_transform = build_residual_action_transform(
        control_indices=control_indices,
        residual_limits=residual_limits,
        full_action_dim=env_action_dim,
        chunk_horizon=chunk_horizon,
        chunk_step_enabled=chunk_step_enabled,
        clip_gripper=bool(cfg.residual.clip_gripper),
    )

    def _resolve_eval_gate(alpha_value: float) -> tuple[float, bool]:
        if alpha_value <= 0.0:
            return 1.0, False
        if not epsilon_gating_enabled or epsilon_gating_eval_force_on:
            return 1.0, True
        gate_prob = _scheduled_epsilon_gating_probability(
            cfg, schedule_step=int(10**12)
        )
        gate_on = bool(np.random.random() < float(gate_prob))
        return float(gate_prob), gate_on

    checkpoint_path = str(cfg.eval.checkpoint_path) if cfg.eval.checkpoint_path else ""
    if checkpoint_path and checkpoint_path.lower() != "null":
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())

    checkpoint_loaded = False
    agent = None
    step_logger = JsonlLogger(run_dir / str(cfg.logging.step_log_file))
    episode_logger = JsonlLogger(run_dir / str(cfg.logging.episode_log_file))
    tb_writer: Optional[SummaryWriter] = None
    if bool(cfg.logging.get("tensorboard", True)):
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    total_success = 0
    total_env_steps = 0
    total_policy_steps = 0
    precheck_failed_episodes = 0
    obs_cache = AgiBotObservationCache()

    try:
        episode_id = 0
        while episode_id < int(cfg.eval.episodes):
            if bool(cfg.eval.get("expert_check", False)):
                passed, _ = env.expert_precheck(init_episode_idx=episode_id)
                if not passed:
                    precheck_failed_episodes += 1
                    logger.warning(
                        "expert precheck failed for eval episode=%s; counting as failed episode",
                        episode_id,
                    )
                    episode_id += 1
                    continue

            obs_raw = env.reset(init_episode_idx=episode_id)
            obs_cache.clear()
            success = False
            episode_steps = 0
            episode_return = 0.0
            max_episode_steps = int(env.step_limit)
            if cfg.eval.max_env_steps_per_episode is not None:
                max_episode_steps = min(
                    max_episode_steps, int(cfg.eval.max_env_steps_per_episode)
                )

            if controller_enabled:
                controller_success = False

                def _plan_eval_chunk(
                    controller_obs: Dict[str, Any],
                    remaining_steps: int,
                ) -> list[ControllerPlannedStep]:
                    nonlocal agent, checkpoint_loaded, total_policy_steps
                    action_chunk, infer_info = policy_client.infer_chunk(
                        build_agibot_policy_input(
                            controller_obs,
                            env.current_instruction,
                            obs_cache=obs_cache,
                        )
                    )
                    base_chunk = select_action_chunk_window(
                        action_chunk, horizon=chunk_horizon
                    )
                    if (
                        base_chunk.ndim != 2
                        or int(base_chunk.shape[1]) != env_action_dim
                    ):
                        raise ValueError(
                            "Unexpected base action chunk shape: "
                            f"{base_chunk.shape}, expected [H,{env_action_dim}]"
                        )

                    if checkpoint_path and agent is None:
                        sample_obs = build_residual_step_obs(
                            controller_obs,
                            base_chunk[0],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            obs_cache=obs_cache,
                            base_action_chunk=base_chunk,
                            alpha=float(residual_alpha),
                            state_mode=obs_state_mode,
                            action_dim=env_action_dim,
                        )
                        agent = create_drq_agent_from_cfg(
                            cfg,
                            sample_obs=sample_obs,
                            action_dim=agent_action_dim,
                            image_keys=image_keys,
                            critic_action_dim=critic_action_dim,
                            action_transform=action_transform,
                        )
                        agent = load_agent_checkpoint(checkpoint_path, agent)
                        checkpoint_loaded = True
                        logger.info(
                            "Loaded residual checkpoint from: %s", checkpoint_path
                        )

                    gate_prob, gate_on = _resolve_eval_gate(float(residual_alpha))
                    if checkpoint_loaded and agent is not None:
                        obs_input = build_residual_step_obs(
                            controller_obs,
                            base_chunk[0],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            obs_cache=obs_cache,
                            base_action_chunk=base_chunk,
                            alpha=float(residual_alpha),
                            state_mode=obs_state_mode,
                            action_dim=env_action_dim,
                        )
                        sampled = agent.sample_actions(
                            obs_input,
                            deterministic=bool(cfg.eval.deterministic),
                        )
                        residual_chunk = reshape_flat_action_to_chunk(
                            sampled,
                            action_dim=step_action_dim,
                            chunk_horizon=chunk_horizon,
                        )
                    else:
                        residual_chunk = np.zeros(
                            (chunk_horizon, step_action_dim),
                            dtype=np.float32,
                        )
                    if not gate_on:
                        residual_chunk = np.zeros_like(residual_chunk)

                    execute_horizon = int(min(chunk_horizon, int(remaining_steps)))
                    executed_base_chunk = np.asarray(
                        base_chunk[:execute_horizon], dtype=np.float32
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
                        alpha=residual_alpha,
                        clip_gripper=bool(cfg.residual.clip_gripper),
                    )
                    sequence_ids = env.enqueue_action_chunk(final_chunk)
                    accepted_horizon = int(len(sequence_ids))
                    if accepted_horizon <= 0:
                        logger.warning(
                            "Eval controller enqueue accepted no actions; "
                            "the operator may have changed controller state during planning."
                        )
                        return []
                    if accepted_horizon != execute_horizon:
                        logger.warning(
                            "Eval controller enqueue accepted %s/%s actions; truncating the plan.",
                            accepted_horizon,
                            execute_horizon,
                        )
                        execute_horizon = int(accepted_horizon)
                        executed_base_chunk = executed_base_chunk[:execute_horizon]
                        executed_residual_chunk = executed_residual_chunk[
                            :execute_horizon
                        ]
                        delta_chunk = delta_chunk[:execute_horizon]
                        final_chunk = final_chunk[:execute_horizon]
                    total_policy_steps += 1
                    return [
                        ControllerPlannedStep(
                            sequence_id=int(sequence_id),
                            obs_before=(controller_obs if chunk_step == 0 else None),
                            final_action=np.asarray(
                                final_chunk[chunk_step], dtype=np.float32
                            ),
                            chunk_step=int(chunk_step),
                            executed_horizon=int(execute_horizon),
                            metadata={
                                "base_action": np.asarray(
                                    executed_base_chunk[chunk_step],
                                    dtype=np.float32,
                                ),
                                "delta_action": np.asarray(
                                    delta_chunk[chunk_step],
                                    dtype=np.float32,
                                ),
                                "gate_prob": float(gate_prob),
                                "gate_on": bool(gate_on),
                                "infer_info": dict(infer_info),
                            },
                        )
                        for chunk_step, sequence_id in enumerate(sequence_ids)
                    ]

                def _on_eval_step(
                    executed: ControllerExecutedStep, current_step: int
                ) -> None:
                    nonlocal total_env_steps, controller_success
                    total_env_steps += 1
                    controller_success = bool(
                        executed.info.get("success", controller_success)
                    )
                    metadata = executed.planned.metadata
                    infer_info = dict(metadata.get("infer_info", {}))
                    step_logger.write(
                        {
                            "global_env_step": int(total_env_steps),
                            "global_policy_step": int(total_policy_steps),
                            "episode_id": episode_id,
                            "episode_step": int(current_step),
                            "replan_point": bool(executed.planned.chunk_step == 0),
                            "chunk_step": int(executed.planned.chunk_step),
                            "chunk_horizon": int(executed.planned.executed_horizon),
                            "infer_e2e_ms": infer_info.get("e2e_ms")
                            if executed.planned.chunk_step == 0
                            else None,
                            "infer_policy_ms": infer_info.get("policy_ms")
                            if executed.planned.chunk_step == 0
                            else None,
                            "infer_server_ms": infer_info.get("server_ms")
                            if executed.planned.chunk_step == 0
                            else None,
                            "a_base": np.asarray(
                                metadata["base_action"],
                                dtype=np.float32,
                            ).tolist(),
                            "a_res": np.asarray(
                                metadata["delta_action"],
                                dtype=np.float32,
                            ).tolist(),
                            "a_final": np.asarray(
                                executed.planned.final_action,
                                dtype=np.float32,
                            ).tolist(),
                            "epsilon_gate_prob": float(metadata["gate_prob"]),
                            "epsilon_gate_on": bool(metadata["gate_on"]),
                            "reward": float(executed.reward),
                            "done": bool(executed.done or executed.truncated),
                            "success": bool(controller_success),
                        }
                    )

                controller_summary = run_controller_episode(
                    env=env,
                    initial_obs=obs_raw,
                    max_episode_steps=max_episode_steps,
                    chunk_horizon=chunk_horizon,
                    cfg=cfg,
                    logger=logger,
                    plan_chunk_fn=_plan_eval_chunk,
                    on_step_fn=_on_eval_step,
                )
                success = bool(controller_summary.success)
                episode_steps = int(controller_summary.episode_steps)
                episode_return = float(controller_summary.episode_return)
            else:
                decision_done = False
                while (not decision_done) and episode_steps < max_episode_steps:
                    action_chunk, infer_info = policy_client.infer_chunk(
                        build_agibot_policy_input(
                            obs_raw, env.current_instruction, obs_cache=obs_cache
                        )
                    )
                    base_chunk = select_action_chunk_window(
                        action_chunk, horizon=chunk_horizon
                    )
                    if (
                        base_chunk.ndim != 2
                        or int(base_chunk.shape[1]) != env_action_dim
                    ):
                        raise ValueError(
                            f"Unexpected base action chunk shape: {base_chunk.shape}, expected [H,{env_action_dim}]"
                        )

                    if checkpoint_path and agent is None:
                        sample_obs = build_residual_step_obs(
                            obs_raw,
                            base_chunk[0],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            obs_cache=obs_cache,
                            base_action_chunk=(
                                base_chunk if chunk_step_enabled else None
                            ),
                            alpha=float(residual_alpha),
                            state_mode=obs_state_mode,
                            action_dim=env_action_dim,
                        )
                        agent = create_drq_agent_from_cfg(
                            cfg,
                            sample_obs=sample_obs,
                            action_dim=agent_action_dim,
                            image_keys=image_keys,
                            critic_action_dim=critic_action_dim,
                            action_transform=action_transform,
                        )
                        agent = load_agent_checkpoint(checkpoint_path, agent)
                        checkpoint_loaded = True
                        logger.info(
                            "Loaded residual checkpoint from: %s", checkpoint_path
                        )

                    if chunk_step_enabled:
                        gate_prob, gate_on = _resolve_eval_gate(float(residual_alpha))
                        if checkpoint_loaded and agent is not None:
                            obs_input = build_residual_step_obs(
                                obs_raw,
                                base_chunk[0],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                obs_cache=obs_cache,
                                base_action_chunk=base_chunk,
                                alpha=float(residual_alpha),
                                state_mode=obs_state_mode,
                                action_dim=env_action_dim,
                            )
                            sampled = agent.sample_actions(
                                obs_input, deterministic=bool(cfg.eval.deterministic)
                            )
                            residual_chunk = reshape_flat_action_to_chunk(
                                sampled,
                                action_dim=step_action_dim,
                                chunk_horizon=chunk_horizon,
                            )
                        else:
                            residual_chunk = np.zeros(
                                (chunk_horizon, step_action_dim), dtype=np.float32
                            )
                        if not gate_on:
                            residual_chunk = np.zeros_like(residual_chunk)

                        execute_horizon = int(
                            min(chunk_horizon, max_episode_steps - episode_steps)
                        )
                        executed_base_chunk = base_chunk[:execute_horizon]
                        executed_residual_chunk = residual_chunk[:execute_horizon]
                        delta_chunk, final_chunk = compose_residual_action_chunk(
                            base_chunk=executed_base_chunk,
                            residual_chunk=executed_residual_chunk,
                            indices=control_indices,
                            limits=residual_limits,
                            alpha=residual_alpha,
                            clip_gripper=bool(cfg.residual.clip_gripper),
                        )
                        total_policy_steps += 1
                        chunk_result = env.step_chunk(final_chunk)
                        obs_raw = chunk_result["obs"]
                        chunk_rewards = [float(v) for v in chunk_result["rewards"]]
                        chunk_infos = [dict(v) for v in chunk_result["infos"]]
                        chunk_dones = [bool(v) for v in chunk_result["dones"]]
                        for executed_step in range(len(chunk_rewards)):
                            reward = float(chunk_rewards[executed_step])
                            info = chunk_infos[executed_step]
                            episode_steps += 1
                            total_env_steps += 1
                            episode_return += reward
                            success = bool(info.get("success", success))
                            timeout = bool(episode_steps >= max_episode_steps)
                            done = bool(chunk_dones[executed_step] or timeout)
                            step_logger.write(
                                {
                                    "global_env_step": int(total_env_steps),
                                    "global_policy_step": int(total_policy_steps),
                                    "episode_id": episode_id,
                                    "episode_step": episode_steps,
                                    "replan_point": bool(executed_step == 0),
                                    "chunk_step": int(executed_step),
                                    "chunk_horizon": int(execute_horizon),
                                    "infer_e2e_ms": infer_info.get("e2e_ms")
                                    if executed_step == 0
                                    else None,
                                    "infer_policy_ms": infer_info.get("policy_ms")
                                    if executed_step == 0
                                    else None,
                                    "infer_server_ms": infer_info.get("server_ms")
                                    if executed_step == 0
                                    else None,
                                    "a_base": executed_base_chunk[
                                        executed_step
                                    ].tolist(),
                                    "a_res": delta_chunk[executed_step].tolist(),
                                    "a_final": final_chunk[executed_step].tolist(),
                                    "epsilon_gate_prob": float(gate_prob),
                                    "epsilon_gate_on": bool(gate_on),
                                    "reward": float(reward),
                                    "done": bool(done),
                                    "success": bool(success),
                                }
                            )
                            if done:
                                decision_done = True
                                break
                    else:
                        for chunk_step in range(chunk_horizon):
                            if episode_steps >= max_episode_steps:
                                decision_done = True
                                break
                            obs_input = build_residual_step_obs(
                                obs_raw,
                                base_chunk[chunk_step],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                obs_cache=obs_cache,
                                base_action_chunk=None,
                                alpha=float(residual_alpha),
                                state_mode=obs_state_mode,
                                action_dim=env_action_dim,
                            )
                            gate_prob, gate_on = _resolve_eval_gate(
                                float(residual_alpha)
                            )
                            if checkpoint_loaded and agent is not None:
                                sampled = agent.sample_actions(
                                    obs_input,
                                    deterministic=bool(cfg.eval.deterministic),
                                )
                                residual_step_action = as_numpy_action(
                                    sampled, step_action_dim
                                )
                            else:
                                residual_step_action = np.zeros(
                                    (step_action_dim,), dtype=np.float32
                                )
                            if not gate_on:
                                residual_step_action = np.zeros_like(
                                    residual_step_action
                                )
                            delta_action, final_action = compose_residual_action(
                                base_action=base_chunk[chunk_step],
                                residual_action=residual_step_action,
                                indices=control_indices,
                                limits=residual_limits,
                                alpha=residual_alpha,
                                clip_gripper=bool(cfg.residual.clip_gripper),
                            )
                            total_policy_steps += 1
                            obs_raw, reward, env_done, truncated, info = env.step(
                                final_action
                            )
                            episode_steps += 1
                            total_env_steps += 1
                            episode_return += float(reward)
                            success = bool(info.get("success", success))
                            timeout = bool(episode_steps >= max_episode_steps)
                            done = bool(env_done or truncated or timeout)
                            step_logger.write(
                                {
                                    "global_env_step": int(total_env_steps),
                                    "global_policy_step": int(total_policy_steps),
                                    "episode_id": episode_id,
                                    "episode_step": episode_steps,
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
                                    "a_res": delta_action.tolist(),
                                    "a_final": final_action.tolist(),
                                    "epsilon_gate_prob": float(gate_prob),
                                    "epsilon_gate_on": bool(gate_on),
                                    "reward": float(reward),
                                    "done": bool(done),
                                    "success": bool(success),
                                }
                            )
                            if done:
                                decision_done = True
                                break

            total_success += int(bool(success))
            episode_payload = {
                "episode_id": episode_id,
                "episode_steps": int(episode_steps),
                "episode_return": float(episode_return),
                "success": bool(success),
                "checkpoint_path": checkpoint_path or None,
                "residual_alpha": float(residual_alpha),
            }
            episode_logger.write(episode_payload)
            if tb_writer is not None:
                tb_writer.add_scalar(
                    "eval/episode_return", float(episode_return), episode_id
                )
                tb_writer.add_scalar("eval/success", float(bool(success)), episode_id)
                tb_writer.add_scalar(
                    "eval/episode_steps", float(episode_steps), episode_id
                )
            logger.info(
                "Eval episode=%s success=%s steps=%s return=%.4f",
                episode_id,
                bool(success),
                episode_steps,
                float(episode_return),
            )
            episode_id += 1
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        env.close(clear_cache=False)

    summary = {
        "episodes": int(cfg.eval.episodes),
        "successes": int(total_success),
        "success_rate": float(total_success / max(1, int(cfg.eval.episodes))),
        "total_env_steps": int(total_env_steps),
        "total_policy_steps": int(total_policy_steps),
        "precheck_failed_episodes": int(precheck_failed_episodes),
        "checkpoint_path": checkpoint_path or None,
        "residual_alpha": float(residual_alpha),
        "task_name": str(cfg.task.name),
        "task_key": resolve_agibot_cfg_task_key(cfg),
    }
    summary_path = run_dir / str(cfg.logging.summary_file)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
