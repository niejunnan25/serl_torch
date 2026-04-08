from __future__ import annotations

"""Fast evaluation for LIBERO residual policies."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from serl_launcher.data.normalizer import StateActionNormalizer, load_normalizer
from serl_launcher.policy.factory import build_policy_backend_info
from serl_launcher.policy.factory import build_policy_client
from serl_launcher.agents.continuous.builders import build_drq_agent
from serl_launcher.residual.action import as_numpy_action
from serl_launcher.residual.action import as_numpy_action_chunk
from serl_launcher.residual.action import compose_residual_action
from serl_launcher.residual.action import compose_residual_action_chunk
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.train.config import build_residual_action_transform
from serl_launcher.residual.train.config import resolve_control_indices_from_cfg
from serl_launcher.residual.train.config import (
    resolve_residual_observation_state_mode,
)
from serl_launcher.residual.train.config import sample_probing_steps
from serl_launcher.residual.train.schedules import _epsilon_gating_enabled
from serl_launcher.residual.train.schedules import _epsilon_gating_eval_force_on
from serl_launcher.residual.train.schedules import _scheduled_epsilon_gating_probability
from serl_launcher.training.seeding import set_global_seeds
from serl_launcher.utils.alpha_utils import require_residual_alpha
from serl_launcher.utils.logger import JsonlLogger

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.env_wrappers import LiberoTaskEnv
from serl_torch.examples.libero.env_wrappers import RemoteLiberoTaskEnv
from serl_torch.examples.libero.config import resolve_libero_cfg_image_keys
from serl_torch.examples.libero.runtime.obs_adapter import LiberoObservationCache
from serl_torch.examples.libero.runtime.obs_adapter import build_residual_step_obs
from serl_torch.examples.libero.runtime.policy_adapter import build_libero_policy_input

from torch.utils.tensorboard import SummaryWriter

from serl_launcher.utils.checkpoint_utils import load_agent_checkpoint


def _create_env(cfg: DictConfig, logger: logging.Logger):
    env_backend = str(cfg.get("env", {}).get("backend", "remote")).lower()
    common_kwargs = dict(
        suite_name=str(cfg.task.suite_name),
        task_id=int(cfg.task.task_id),
        action_dim=cfg.get("env", {}).get("action_dim", None),
        resolution=int(cfg.task.resolution),
        num_steps_wait=int(cfg.task.num_steps_wait),
        max_episode_steps=(
            int(cfg.task.max_episode_steps)
            if cfg.task.max_episode_steps is not None
            else None
        ),
        libero_root=cfg.get("libero_root", None),
        libero_config_dir=cfg.get("libero_config_dir", None),
        libero_datasets_root=cfg.get("libero_datasets_root", None),
        env_seed_mode=str(cfg.task.get("env_seed_mode", "per_episode")),
        fixed_env_seed=cfg.task.get("fixed_env_seed", None),
        init_state_index_mode=str(cfg.task.get("init_state_index_mode", "seed")),
        logger=logger,
    )
    if env_backend == "local":
        return LiberoTaskEnv(**common_kwargs)
    if env_backend == "remote":
        remote_cfg = cfg.get("env", {}).get("remote", {})
        return RemoteLiberoTaskEnv(
            host=str(remote_cfg.get("host", "127.0.0.1")),
            port=int(remote_cfg.get("port", 30000)),
            timeout_sec=float(remote_cfg.get("timeout_sec", 120.0)),
            **common_kwargs,
        )
    raise ValueError(f"env.backend must be 'local' or 'remote', got {env_backend}")


@hydra.main(version_base=None, config_path="../conf", config_name="eval_residual_fast")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("libero_eval_residual_fast")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_global_seeds(int(cfg.seed))

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
        stats_dir = norm_cfg.get(
            "stats_dir",
            str(Path(__file__).resolve().parents[1] / "data" / "stats"),
        )
        normalizer = load_normalizer(
            task_key, stats_dir=stats_dir
        )
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)
    obs_state_mode = resolve_residual_observation_state_mode(cfg)
    logger.info(
        "Residual observation state_mode=%s normalization.enabled=%s",
        obs_state_mode,
        bool(norm_cfg.get("enabled", False)) if norm_cfg is not None else False,
    )

    policy_backend_info = build_policy_backend_info(cfg)
    policy_client = build_policy_client(cfg, logger=logger)
    logger.info(
        "Chunk policy backend: type=%s id=%s",
        policy_backend_info["type"],
        policy_backend_info["id"],
    )

    image_keys = resolve_libero_cfg_image_keys(cfg)
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    env_action_dim_cfg = cfg.get("env", {}).get("action_dim", None)
    if env_action_dim_cfg is None:
        raise ValueError("env.action_dim must be set in yaml (e.g. env.action_dim: 7)")
    env_action_dim = int(env_action_dim_cfg)
    if env_action_dim <= 0:
        raise ValueError(f"env.action_dim must be positive, got {env_action_dim}")

    control_indices = resolve_control_indices_from_cfg(
        cfg, full_action_dim=env_action_dim
    )
    step_action_dim = int(len(control_indices))
    chunk_horizon = int(cfg.residual.chunk_horizon)
    residual_alpha = require_residual_alpha(cfg.get("residual", None))
    epsilon_gating_enabled = _epsilon_gating_enabled(cfg)
    epsilon_gating_eval_force_on = _epsilon_gating_eval_force_on(cfg)
    chunk_step_cfg = cfg.get("chunk_step", None)
    chunk_step_enabled = (
        bool(chunk_step_cfg.get("enabled", False))
        if chunk_step_cfg is not None
        else False
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
    residual_action_limits_cfg = cfg.residual.get("action_limits", None)
    residual_limits = build_residual_limits(
        control_indices,
        action_limits=residual_action_limits_cfg,
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
        if not epsilon_gating_enabled:
            return 1.0, True
        if epsilon_gating_eval_force_on:
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
    enable_tensorboard = bool(cfg.logging.get("tensorboard", True))
    tb_writer: Optional[SummaryWriter] = None
    if enable_tensorboard:
        tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    else:
        logger.info("TensorBoard logging disabled for this eval run")

    total_success = 0
    total_env_steps = 0
    total_policy_steps = 0
    precheck_failed_episodes = 0
    total_probing_steps = 0
    obs_cache = LiberoObservationCache()

    def _policy_input(obs_raw: Dict[str, Any], prompt: str):
        return build_libero_policy_input(
            obs_raw,
            prompt,
            obs_cache=obs_cache,
        )

    eval_seed = int(cfg.eval.get("seed", 7))
    logger.info("Evaluation uses fixed seed=%s for all episodes", eval_seed)

    try:
        episode_id = 0
        while episode_id < int(cfg.eval.episodes):
            seed = int(eval_seed)

            if bool(cfg.eval.get("expert_check", False)):
                passed, _ = env.expert_precheck(seed=seed, init_episode_idx=episode_id)
                if not passed:
                    precheck_failed_episodes += 1
                    logger.warning(
                        "expert precheck failed for eval episode=%s (seed=%s); counting as failed episode",
                        episode_id,
                        seed,
                    )
                    episode_id += 1
                    continue

            obs_raw = env.reset(seed=seed, init_episode_idx=episode_id)
            obs_cache.clear()
            success = False
            episode_steps = 0
            episode_return = 0.0

            max_episode_steps = int(env.step_limit)
            if cfg.eval.max_env_steps_per_episode is not None:
                max_episode_steps = min(
                    max_episode_steps, int(cfg.eval.max_env_steps_per_episode)
                )

            probing_steps_target = sample_probing_steps(
                cfg.eval, episode_horizon=max_episode_steps
            )
            if probing_steps_target > 0:
                probing_remaining = int(
                    min(probing_steps_target, max_episode_steps - episode_steps)
                )
                while probing_remaining > 0 and episode_steps < max_episode_steps:
                    probe_chunk, probe_info = policy_client.infer_chunk(
                        _policy_input(obs_raw, env.current_instruction)
                    )
                    probe_base_chunk = select_action_chunk_window(
                        probe_chunk,
                        horizon=chunk_horizon,
                        action_dim=env_action_dim,
                    )
                    probe_done = False
                    for probe_step in range(chunk_horizon):
                        if probing_remaining <= 0 or episode_steps >= max_episode_steps:
                            break
                        base_action = probe_base_chunk[probe_step]
                        next_obs_raw, reward, env_done, _, info = env.step(base_action)
                        episode_steps += 1
                        total_env_steps += 1
                        total_probing_steps += 1
                        probing_remaining -= 1
                        episode_return += float(reward)
                        success = bool(info["success"])
                        timeout = bool(episode_steps >= max_episode_steps)
                        done = bool(env_done or timeout)
                        step_logger.write(
                            {
                                "global_env_step": int(total_env_steps),
                                "global_policy_step": int(total_policy_steps),
                                "episode_id": episode_id,
                                "episode_step": episode_steps,
                                "seed": int(
                                    env.last_seed if env.last_seed is not None else seed
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
                                "a_res": [0.0] * step_action_dim,
                                "a_final": base_action.tolist(),
                                "reward": float(reward),
                                "done": bool(done),
                                "success": bool(success),
                            }
                        )
                        obs_raw = next_obs_raw
                        if done:
                            probe_done = True
                            break
                    if probe_done:
                        break

            decision_done = bool(episode_steps >= max_episode_steps or success)
            while episode_steps < max_episode_steps:
                if decision_done:
                    break
                openpi_chunk, infer_info = policy_client.infer_chunk(
                    _policy_input(obs_raw, env.current_instruction)
                )
                base_chunk = select_action_chunk_window(
                    openpi_chunk,
                    horizon=chunk_horizon,
                    action_dim=env_action_dim,
                )
                next_obs_raw = obs_raw
                for chunk_step in range(chunk_horizon):
                    if episode_steps >= max_episode_steps:
                        decision_done = True
                        break

                    obs_input = build_residual_step_obs(
                        next_obs_raw,
                        base_chunk[chunk_step],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                        base_action_chunk=(base_chunk if chunk_step_enabled else None),
                        alpha=float(residual_alpha),
                        state_mode=obs_state_mode,
                    )

                    if checkpoint_path and agent is None:
                        agent = build_drq_agent(
                            cfg,
                            sample_obs=obs_input,
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
                            sampled = agent.sample_actions(
                                obs_input,
                                deterministic=bool(cfg.eval.deterministic),
                            )
                            residual_chunk = as_numpy_action_chunk(
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
                        next_obs_raw = chunk_result["obs"]
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
                                    "seed": int(
                                        env.last_seed
                                        if env.last_seed is not None
                                        else seed
                                    ),
                                    "is_probing": False,
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
                        break
                    else:
                        gate_prob, gate_on = _resolve_eval_gate(float(residual_alpha))
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
                            residual_step_action = np.zeros_like(residual_step_action)

                        delta_action, final_action = compose_residual_action(
                            base_action=base_chunk[chunk_step],
                            residual_action=residual_step_action,
                            indices=control_indices,
                            limits=residual_limits,
                            alpha=residual_alpha,
                            clip_gripper=bool(cfg.residual.clip_gripper),
                        )

                        total_policy_steps += 1
                        next_obs_raw, reward, env_done, _, info = env.step(final_action)
                        episode_steps += 1
                        total_env_steps += 1
                        episode_return += float(reward)
                        success = bool(info["success"])

                        timeout = bool(episode_steps >= max_episode_steps)
                        done = bool(env_done or timeout)
                        step_logger.write(
                            {
                                "global_env_step": int(total_env_steps),
                                "global_policy_step": int(total_policy_steps),
                                "episode_id": episode_id,
                                "episode_step": episode_steps,
                                "seed": int(
                                    env.last_seed if env.last_seed is not None else seed
                                ),
                                "is_probing": False,
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

                obs_raw = next_obs_raw
                if decision_done:
                    break

            total_success += int(success)
            running_success_rate = float(total_success) / float(episode_id + 1)
            episode_logger.write(
                {
                    "episode_id": episode_id,
                    "seed": int(env.last_seed if env.last_seed is not None else seed),
                    "success": bool(success),
                    "episode_steps": int(episode_steps),
                    "episode_return": float(episode_return),
                    "global_env_step": int(total_env_steps),
                    "global_policy_step": int(total_policy_steps),
                    "running_success_rate": running_success_rate,
                }
            )

            if tb_writer is not None:
                tb_writer.add_scalar("eval/success", int(success), episode_id)
                tb_writer.add_scalar("eval/return", float(episode_return), episode_id)
                tb_writer.add_scalar("eval/length", int(episode_steps), episode_id)
                tb_writer.add_scalar(
                    "eval/running_success_rate", running_success_rate, episode_id
                )

            logger.info(
                "episode=%s success=%s steps=%s return=%.2f success_rate=%.3f",
                episode_id,
                success,
                episode_steps,
                episode_return,
                running_success_rate,
            )
            episode_id += 1

        summary = {
            "episodes": int(episode_id),
            "total_env_steps": int(total_env_steps),
            "total_policy_steps": int(total_policy_steps),
            "total_success": int(total_success),
            "success_rate": float(total_success / max(1, int(episode_id))),
            "base_policy_type": str(policy_backend_info["type"]),
            "base_policy_id": str(policy_backend_info["id"]),
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_path": checkpoint_path,
            "chunk_horizon": int(chunk_horizon),
            "residual_action_dim": int(agent_action_dim),
            "chunk_step_enabled": bool(chunk_step_enabled),
            "residual_alpha": float(residual_alpha),
            "expert_check": bool(cfg.eval.expert_check),
            "eval_seed": int(eval_seed),
            "expert_precheck_failed_episodes": int(precheck_failed_episodes),
            "enable_base_probing": bool(cfg.eval.get("enable_base_probing", False)),
            "probing_alpha": (
                float(cfg.eval.get("probing_alpha"))
                if cfg.eval.get("probing_alpha", None) is not None
                else None
            ),
            "probing_steps_range": [
                int(cfg.eval.get("probing_min_steps", 0)),
                int(cfg.eval.get("probing_max_steps", 0)),
            ],
            "total_probing_steps": int(total_probing_steps),
        }
        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("evaluation done: %s", summary)

    finally:
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        step_logger.close()
        episode_logger.close()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
