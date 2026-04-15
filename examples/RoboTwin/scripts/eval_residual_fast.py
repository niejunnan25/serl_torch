"""
RoboTwin 残差策略快速评估脚本。

在仿真环境中评估「OpenPI 基策略 + SAC 残差策略」的联合表现：
- 每步先用 OpenPI 服务得到基动作块（base chunk）
- 在同一个 base chunk 内，每个环境步都用 DrQ/SAC 残差策略输出一步残差
- 将基动作与残差组合后，在环境中执行一个 chunk 内的多步
- 支持无 checkpoint 时仅用基策略（残差为零）运行

评估数据流：
obs_raw -> OpenPI infer_chunk -> base_chunk(H,14)
      -> for t in chunk:
           build_residual_step_obs(obs_t, base_action_t)
           residual = policy(obs_input) 或 0
           final_action = base_chunk[t] + residual_delta
           env.step(final_action) -> obs_{t+1}, reward, done
      -> 记录 step/episode 级日志并汇总成功率
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import ALOHA_ACTION_DIM, JsonlLogger, ensure_serl_launcher_importable
from utils.config_utils import (
    set_global_seeds,
    resolve_image_keys,
    resolve_control_indices_from_cfg,
    create_drq_agent,
    sample_probing_steps,
)
from env_wrappers import (
    RoboTwinTaskEnv,
    RemoteRoboTwinTaskEnv,
    load_task_args,
    resolve_robo_root,
    setup_robotwin_pythonpath,
)
from policy import (
    OpenPIChunkClient,
    as_numpy_action,
    build_residual_limits,
    build_residual_step_obs,
    compose_residual_action,
    select_action_chunk_window,
)

ensure_serl_launcher_importable()

from torch.utils.tensorboard import SummaryWriter

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.checkpoint_utils import load_agent_checkpoint


@hydra.main(version_base=None, config_path="../conf", config_name="eval_residual_fast")
def main(cfg: DictConfig) -> None:
    """
    主入口：加载配置、创建环境与 OpenPI 客户端，按 episode 循环评估基策略+残差策略，
    记录每步与每 episode 的日志，并写入汇总 JSON。
    """
    # ---------- 输出目录与日志 ----------
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("eval_residual_fast")

    set_global_seeds(int(cfg.seed))

    # ---------- 环境后端：local 或 remote ----------
    env_backend = str(cfg.get("env", {}).get("backend", "local")).lower()
    if env_backend not in {"local", "remote"}:
        raise ValueError(f"env.backend must be 'local' or 'remote', got {env_backend}")

    robo_root = resolve_robo_root(cfg.robo_root) if env_backend == "local" else None
    if env_backend == "local":
        setup_robotwin_pythonpath(robo_root)
        os.chdir(robo_root)
        logger.info("RoboTwin root: %s", robo_root)
    else:
        logger.info("Use remote env backend; skip local RoboTwin import/chdir")

    logger.info("Hydra run dir: %s", run_dir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # ---------- 任务参数与环境 ----------
    if env_backend == "local":
        assert robo_root is not None
        task_args = load_task_args(
            robo_root, str(cfg.task.name), str(cfg.task.task_config)
        )
    else:
        task_args = {"task_config": str(cfg.task.task_config)}

    instruction_type = str(cfg.task.get("instruction_type", "seen"))
    if env_backend == "local":
        env = RoboTwinTaskEnv(
            task_name=str(cfg.task.name),
            task_args=task_args,
            prompt=str(cfg.task.prompt),
            max_setup_retries=int(cfg.task.setup_retries),
            instruction_type=instruction_type,
            logger=logger,
        )
    else:
        remote_cfg = cfg.get("env", {}).get("remote", {})
        env = RemoteRoboTwinTaskEnv(
            host=str(remote_cfg.get("host", "127.0.0.1")),
            port=int(remote_cfg.get("port", 9100)),
            timeout_sec=float(remote_cfg.get("timeout_sec", 120.0)),
            robo_root=remote_cfg.get("robo_root", cfg.robo_root),
            task_name=str(cfg.task.name),
            task_args=task_args,
            prompt=str(cfg.task.prompt),
            max_setup_retries=int(cfg.task.setup_retries),
            instruction_type=instruction_type,
            logger=logger,
        )

    # ---------- OpenPI 基策略客户端（用于获取 base action chunk）----------
    openpi_client = OpenPIChunkClient(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )

    # ---------- 观测与动作维度校验 ----------
    # 与训练保持一致：obs_stack_horizon=1，残差动作维度由配置解析。
    image_keys = resolve_image_keys(cfg)
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    control_indices = resolve_control_indices_from_cfg(cfg)

    # 残差限幅：臂与夹爪的 delta 上下界，用于 compose 时裁剪
    residual_limits = build_residual_limits(
        control_indices,
        arm_limit=float(cfg.residual.arm_delta_limit),
        gripper_limit=float(cfg.residual.gripper_delta_limit),
    )
    residual_xi = float(cfg.residual.get("xi", 1.0))
    if residual_xi <= 0.0:
        raise ValueError(f"residual.xi must be positive, got {residual_xi}")

    # chunk 长度由配置决定；残差按步输出，每步 action_dim 维。
    chunk_horizon = int(cfg.residual.chunk_horizon)
    if chunk_horizon <= 0:
        raise ValueError(
            f"residual.chunk_horizon must be positive, got {chunk_horizon}"
        )
    per_step_action_dim = int(len(control_indices))
    action_dim = int(per_step_action_dim)
    logger.info(
        "Residual config: image_keys=%s action_dim=%s action_indices=%s chunk_horizon=%s xi=%.4f",
        list(image_keys),
        action_dim,
        control_indices.tolist(),
        chunk_horizon,
        residual_xi,
    )

    # ---------- 残差策略 checkpoint 与日志器 ----------
    checkpoint_path = str(cfg.eval.checkpoint_path) if cfg.eval.checkpoint_path else ""
    if checkpoint_path and checkpoint_path.lower() != "null":
        # 相对路径按 examples/RoboTwin 根目录解析（训练输出在此目录下）
        p = Path(checkpoint_path)
        if not p.is_absolute() and not p.exists():
            candidate = PROJECT_ROOT / checkpoint_path
            if candidate.exists():
                checkpoint_path = str(candidate.resolve())
    checkpoint_loaded = False
    agent: DrQAgent | None = None

    step_logger = JsonlLogger(run_dir / str(cfg.logging.step_log_file))
    episode_logger = JsonlLogger(run_dir / str(cfg.logging.episode_log_file))

    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    logger.info("TensorBoard log dir: %s", run_dir / "tb")

    total_success = 0
    total_env_steps = 0
    total_policy_steps = 0
    skipped_seeds = 0
    seed_attempts = 0
    total_probing_steps = 0

    collect_path_cfg = cfg.eval.get("collect_dataset_path", None)
    collect_only_success = bool(cfg.eval.get("collect_only_success", True))
    collect_include_probing = bool(cfg.eval.get("collect_include_probing", False))
    collect_enabled = (
        collect_path_cfg is not None and str(collect_path_cfg).lower() != "null"
    )
    collected_episodes = 0
    collected_steps = 0
    collected_payload: List[Dict[str, Any]] = []

    max_seed_attempts_cfg = cfg.eval.get("max_seed_attempts", None)
    if max_seed_attempts_cfg is None:
        max_seed_attempts = max(1000, int(cfg.eval.episodes) * 100)
    else:
        max_seed_attempts = int(max_seed_attempts_cfg)

    try:
        # ---------- 按 episode 循环评估 ----------
        # 外层是 episode；内层是 chunk（OpenPI 一次）与 step（residual 每步一次）。
        episode_id = 0
        seed_cursor = int(cfg.task.seed_base)

        # 循环评估 episode_id 次，每次评估一个 episode
        while episode_id < int(cfg.eval.episodes):
            seed_attempts += 1
            if seed_attempts > max_seed_attempts:
                raise RuntimeError(
                    "Exceeded max seed attempts during evaluation. "
                    f"attempts={seed_attempts}, completed_episodes={episode_id}, skipped_seeds={skipped_seeds}"
                )
            seed = int(seed_cursor)
            seed_cursor += 1

            episode_info = None
            if bool(cfg.eval.expert_check):
                passed, episode_info = env.expert_precheck(
                    seed=seed, episode_id=episode_id
                )
                if not passed:
                    skipped_seeds += 1
                    logger.warning("skip seed=%s: expert precheck failed", seed)
                    continue

            obs_raw = env.reset(
                seed=seed, episode_id=episode_id, episode_info=episode_info
            )

            success = False
            episode_steps = 0
            episode_return = 0.0
            episode_records: List[Dict[str, Any]] = []

            # 本 episode 允许的最大环境步数（可被 eval 配置截断）
            max_episode_steps = int(env.step_limit)
            if cfg.eval.max_env_steps_per_episode is not None:
                max_episode_steps = min(
                    max_episode_steps, int(cfg.eval.max_env_steps_per_episode)
                )

            # Stage-2 probing: base rollout 做初始状态分布对齐，随后 residual takeover。
            probing_steps_target = sample_probing_steps(
                cfg.eval, episode_horizon=max_episode_steps
            )
            if probing_steps_target > 0:
                probing_remaining = int(
                    min(probing_steps_target, max_episode_steps - episode_steps)
                )
                while probing_remaining > 0 and episode_steps < max_episode_steps:
                    probe_chunk, probe_info = openpi_client.infer_chunk(
                        obs_raw, env.current_instruction
                    )
                    probe_base_chunk = select_action_chunk_window(
                        probe_chunk, horizon=chunk_horizon
                    )
                    probe_done = False
                    for probe_step in range(chunk_horizon):
                        if probing_remaining <= 0 or episode_steps >= max_episode_steps:
                            break
                        base_action = probe_base_chunk[probe_step]
                        obs_before = obs_raw
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
                                "a_res": [0.0] * ALOHA_ACTION_DIM,
                                "a_final": base_action.tolist(),
                                "residual_scale": 0.0,
                                "reward": float(reward),
                                "done": bool(done),
                                "success": bool(success),
                            }
                        )

                        if collect_enabled and collect_include_probing:
                            episode_records.append(
                                {
                                    "is_probing": True,
                                    "obs": copy.deepcopy(obs_before),
                                    "next_obs": copy.deepcopy(next_obs_raw),
                                    "a_base": np.asarray(
                                        base_action, dtype=np.float32
                                    ).copy(),
                                    "a_res": np.zeros(
                                        (ALOHA_ACTION_DIM,), dtype=np.float32
                                    ),
                                    "a_final": np.asarray(
                                        base_action, dtype=np.float32
                                    ).copy(),
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

            if episode_steps >= max_episode_steps or success:
                decision_done = True
            else:
                decision_done = False

            while episode_steps < max_episode_steps:
                if decision_done:
                    break
                # 1) OpenPI 推理得到基策略动作块及耗时信息
                openpi_chunk, infer_info = openpi_client.infer_chunk(
                    obs_raw, env.current_instruction
                )
                base_chunk = select_action_chunk_window(
                    openpi_chunk, horizon=chunk_horizon
                )
                next_obs_raw = obs_raw
                decision_done = False

                # 2) 在 chunk 内逐步执行：每步都推理一次残差动作
                for chunk_step in range(chunk_horizon):
                    if episode_steps >= max_episode_steps:
                        decision_done = True
                        break

                    # 组装残差策略输入：图像 + 融合状态(state + base_action_t)。
                    obs_input = build_residual_step_obs(
                        next_obs_raw,
                        base_chunk[chunk_step],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                    )

                    # 3) 若配置了 checkpoint 且尚未加载，则构建 agent 并加载权重（懒加载）
                    if checkpoint_path and agent is None:
                        agent = create_drq_agent(
                            cfg,
                            sample_obs=obs_input,
                            action_dim=action_dim,
                            image_keys=image_keys,
                        )
                        agent = load_agent_checkpoint(checkpoint_path, agent)
                        checkpoint_loaded = True
                        logger.info(
                            "Loaded residual checkpoint from: %s", checkpoint_path
                        )

                    # 4) 有残差策略则采样残差动作，否则残差全零（仅基策略）
                    if checkpoint_loaded and agent is not None:
                        sampled = agent.sample_actions(
                            obs_input,
                            deterministic=bool(cfg.eval.deterministic),
                        )
                        residual_step_action = as_numpy_action(sampled, action_dim)
                    else:
                        residual_step_action = np.zeros((action_dim,), dtype=np.float32)

                    # 5) 将当前步 base 与 residual 合成 final action。
                    #    residual 会经过 clip/limit/scale 再注入到受控维度。
                    delta_action, final_action = compose_residual_action(
                        base_action=base_chunk[chunk_step],
                        residual_action=residual_step_action,
                        indices=control_indices,
                        limits=residual_limits,
                        residual_scale=float(cfg.eval.residual_scale),
                        xi=residual_xi,
                        clip_gripper=bool(cfg.residual.clip_gripper),
                    )

                    # 这里 total_policy_steps 计“残差策略决策步数”，而不是 OpenPI chunk 次数。
                    total_policy_steps += 1
                    obs_before = next_obs_raw
                    next_obs_raw, reward, env_done, _, info = env.step(final_action)
                    episode_steps += 1
                    total_env_steps += 1
                    episode_return += float(reward)
                    success = bool(info["success"])

                    timeout = bool(episode_steps >= max_episode_steps)
                    done = bool(env_done or timeout)

                    # 每步写一条 jsonl：含全局/本 episode 步数、是否重规划点、推理耗时、基/残差/最终动作、奖励与成功标志
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
                            "residual_scale": float(cfg.eval.residual_scale),
                            "reward": float(reward),
                            "done": bool(done),
                            "success": bool(success),
                        }
                    )

                    if collect_enabled:
                        episode_records.append(
                            {
                                "is_probing": False,
                                "obs": copy.deepcopy(obs_before),
                                "next_obs": copy.deepcopy(next_obs_raw),
                                "a_base": np.asarray(
                                    base_chunk[chunk_step], dtype=np.float32
                                ).copy(),
                                "a_res": np.asarray(
                                    delta_action, dtype=np.float32
                                ).copy(),
                                "a_final": np.asarray(
                                    final_action, dtype=np.float32
                                ).copy(),
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

            # 本 episode 结束：累计成功次数并写 episode 级日志
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

            if collect_enabled and ((not collect_only_success) or success):
                collected_payload.append(
                    {
                        "episode_id": int(episode_id),
                        "seed": int(
                            env.last_seed if env.last_seed is not None else seed
                        ),
                        "success": bool(success),
                        "episode_steps": int(episode_steps),
                        "episode_return": float(episode_return),
                        "transitions": episode_records,
                    }
                )
                collected_episodes += 1
                collected_steps += len(episode_records)

            episode_id += 1

        # ---------- 评估结束：写汇总 JSON ----------
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
            "collect_enabled": bool(collect_enabled),
            "collect_only_success": bool(collect_only_success),
            "collected_episodes": int(collected_episodes),
            "collected_steps": int(collected_steps),
        }  # 汇总：episode 数、总步数、成功数、成功率、是否加载残差、chunk 与动作维度

        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if collect_enabled:
            output_path = Path(str(collect_path_cfg))
            if not output_path.is_absolute():
                output_path = (run_dir / output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "summary": summary,
                "episodes": collected_payload,
            }
            with open(output_path, "wb") as f:
                pickle.dump(payload, f)
            logger.info("collected dataset saved to: %s", output_path)

        logger.info("evaluation done: %s", summary)

    finally:
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        step_logger.close()
        episode_logger.close()
        tb_writer.close()


if __name__ == "__main__":
    main()
