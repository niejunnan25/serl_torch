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

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.data import StateActionNormalizer, load_normalizer
from serl_torch.examples.libero.env_wrappers import (
    LiberoTaskEnv,
    RemoteLiberoTaskEnv,
    resolve_openpi_root,
    setup_openpi_client_pythonpath,
)
from serl_torch.examples.libero.policy import (
    LiberoObservationCache,
    OpenPIChunkClient,
    as_numpy_action,
    build_residual_limits,
    build_residual_step_obs,
    compose_residual_action,
    select_action_chunk_window,
)
from serl_torch.examples.libero.utils import JsonlLogger, ensure_serl_launcher_importable
from serl_torch.examples.libero.utils.config_utils import (
    build_drq_agent,
    resolve_control_indices_from_cfg,
    resolve_image_keys,
    sample_probing_steps,
    set_global_seeds,
)

ensure_serl_launcher_importable()

from torch.utils.tensorboard import SummaryWriter

from serl_launcher.utils.checkpoint_utils import load_agent_checkpoint


def _create_env(cfg: DictConfig, logger: logging.Logger):
    env_backend = str(cfg.get("env", {}).get("backend", "remote")).lower()
    common_kwargs = dict(
        suite_name=str(cfg.task.suite_name),
        task_id=int(cfg.task.task_id),
        resolution=int(cfg.task.resolution),
        num_steps_wait=int(cfg.task.num_steps_wait),
        max_episode_steps=(
            int(cfg.task.max_episode_steps) if cfg.task.max_episode_steps is not None else None
        ),
        libero_root=cfg.get("libero_root", None),
        openpi_root=cfg.get("openpi_root", None),
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

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("libero_eval_residual_fast")
    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_global_seeds(int(cfg.seed))

    openpi_root = resolve_openpi_root(cfg.get("openpi_root", None))
    setup_openpi_client_pythonpath(openpi_root)
    logger.info("openpi root: %s", openpi_root)

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
        normalizer = load_normalizer(task_key, stats_dir=norm_cfg.get("stats_dir", None))
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)

    openpi_client = OpenPIChunkClient(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )

    image_keys = resolve_image_keys(cfg)
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    control_indices = resolve_control_indices_from_cfg(cfg)
    action_dim = int(len(control_indices))
    chunk_horizon = int(cfg.residual.chunk_horizon)
    residual_xi = float(cfg.residual.get("xi", 1.0))
    residual_limits = build_residual_limits(
        control_indices,
        arm_limit=float(cfg.residual.arm_delta_limit),
        gripper_limit=float(cfg.residual.gripper_delta_limit),
    )

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
    skipped_seeds = 0
    seed_attempts = 0
    total_probing_steps = 0
    obs_cache = LiberoObservationCache()

    max_seed_attempts_cfg = cfg.eval.get("max_seed_attempts", None)
    if max_seed_attempts_cfg is None:
        max_seed_attempts = max(1000, int(cfg.eval.episodes) * 100)
    else:
        max_seed_attempts = int(max_seed_attempts_cfg)
    fixed_seed_cfg = cfg.eval.get("fixed_seed", None)
    fixed_seed = None if fixed_seed_cfg is None else int(fixed_seed_cfg)
    if fixed_seed is not None:
        logger.info("Evaluation fixed_seed enabled: all episodes use seed=%s", fixed_seed)

    try:
        episode_id = 0
        seed_cursor = int(cfg.task.seed_base)
        while episode_id < int(cfg.eval.episodes):
            seed_attempts += 1
            if seed_attempts > max_seed_attempts:
                raise RuntimeError(
                    "Exceeded max seed attempts during evaluation. "
                    f"attempts={seed_attempts}, completed_episodes={episode_id}, skipped_seeds={skipped_seeds}"
                )
            if fixed_seed is None:
                seed = int(seed_cursor)
                seed_cursor += 1
            else:
                seed = int(fixed_seed)

            if bool(cfg.eval.get("expert_check", False)):
                passed, _ = env.expert_precheck(seed=seed, episode_id=episode_id)
                if not passed:
                    skipped_seeds += 1
                    logger.warning("skip seed=%s: expert precheck failed", seed)
                    continue

            obs_raw = env.reset(seed=seed, episode_id=episode_id)
            obs_cache.clear()
            success = False
            episode_steps = 0
            episode_return = 0.0

            max_episode_steps = int(env.step_limit)
            if cfg.eval.max_env_steps_per_episode is not None:
                max_episode_steps = min(max_episode_steps, int(cfg.eval.max_env_steps_per_episode))

            probing_steps_target = sample_probing_steps(cfg.eval, episode_horizon=max_episode_steps)
            if probing_steps_target > 0:
                probing_remaining = int(min(probing_steps_target, max_episode_steps - episode_steps))
                while probing_remaining > 0 and episode_steps < max_episode_steps:
                    probe_chunk, probe_info = openpi_client.infer_chunk(
                        obs_raw,
                        env.current_instruction,
                        obs_cache=obs_cache,
                    )
                    probe_base_chunk = select_action_chunk_window(probe_chunk, horizon=chunk_horizon)
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
                                "seed": int(env.last_seed if env.last_seed is not None else seed),
                                "is_probing": True,
                                "replan_point": bool(probe_step == 0),
                                "chunk_step": int(probe_step),
                                "chunk_horizon": int(chunk_horizon),
                                "infer_e2e_ms": probe_info.get("e2e_ms") if probe_step == 0 else None,
                                "infer_policy_ms": probe_info.get("policy_ms") if probe_step == 0 else None,
                                "infer_server_ms": probe_info.get("server_ms") if probe_step == 0 else None,
                                "a_base": base_action.tolist(),
                                "a_res": [0.0] * action_dim,
                                "a_final": base_action.tolist(),
                                "residual_scale": 0.0,
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
                openpi_chunk, infer_info = openpi_client.infer_chunk(
                    obs_raw,
                    env.current_instruction,
                    obs_cache=obs_cache,
                )
                base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
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
                    )

                    if checkpoint_path and agent is None:
                        agent = build_drq_agent(
                            cfg,
                            sample_obs=obs_input,
                            action_dim=action_dim,
                            image_keys=image_keys,
                        )
                        agent = load_agent_checkpoint(checkpoint_path, agent)
                        checkpoint_loaded = True
                        logger.info("Loaded residual checkpoint from: %s", checkpoint_path)

                    if checkpoint_loaded and agent is not None:
                        sampled = agent.sample_actions(
                            obs_input,
                            deterministic=bool(cfg.eval.deterministic),
                        )
                        residual_step_action = as_numpy_action(sampled, action_dim)
                    else:
                        residual_step_action = np.zeros((action_dim,), dtype=np.float32)

                    delta_action, final_action = compose_residual_action(
                        base_action=base_chunk[chunk_step],
                        residual_action=residual_step_action,
                        indices=control_indices,
                        limits=residual_limits,
                        residual_scale=float(cfg.eval.residual_scale),
                        xi=residual_xi,
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
                            "seed": int(env.last_seed if env.last_seed is not None else seed),
                            "is_probing": False,
                            "replan_point": bool(chunk_step == 0),
                            "chunk_step": int(chunk_step),
                            "chunk_horizon": int(chunk_horizon),
                            "infer_e2e_ms": infer_info.get("e2e_ms") if chunk_step == 0 else None,
                            "infer_policy_ms": infer_info.get("policy_ms") if chunk_step == 0 else None,
                            "infer_server_ms": infer_info.get("server_ms") if chunk_step == 0 else None,
                            "a_base": base_chunk[chunk_step].tolist(),
                            "a_res": delta_action.tolist(),
                            "a_final": final_action.tolist(),
                            "residual_scale": float(cfg.eval.residual_scale),
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
                tb_writer.add_scalar("eval/running_success_rate", running_success_rate, episode_id)

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
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_path": checkpoint_path,
            "chunk_horizon": int(chunk_horizon),
            "residual_action_dim": int(action_dim),
            "residual_xi": float(residual_xi),
            "expert_check": bool(cfg.eval.expert_check),
            "skipped_seeds": int(skipped_seeds),
            "seed_attempts": int(seed_attempts),
            "max_seed_attempts": int(max_seed_attempts),
            "seed_start": int(cfg.task.seed_base),
            "seed_next": int(seed_cursor),
            "fixed_seed": int(fixed_seed) if fixed_seed is not None else None,
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
