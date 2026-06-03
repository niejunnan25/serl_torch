"""RLT evaluation runner.

Runs N episodes with a loaded RLT agent checkpoint, collecting
success rate and episode return metrics.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SERL_LAUNCHER_ROOT = REPO_ROOT / "serl_launcher"
for _path in (SERL_LAUNCHER_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from serl_launcher.common.checkpoint_codec import apply_checkpoint_payload_to_agent
from serl_launcher.policy.vla_features.client import VLAFeatureClient
from examples.libero_rlt.config import LiberoRLTTrainConfig
from serl_launcher.agents.rlt.agent import create_rlt_agent_from_cfg
from serl_launcher.agents.rlt.observation import build_rlt_obs
from examples.libero.env.factory import create_env
from openpi_client import image_tools

logger = logging.getLogger(__name__)


def prepare_vla_obs(obs: Dict[str, Any], task_description: str, resize_size: int = 224) -> Dict[str, Any]:
    """Convert a raw LIBERO observation into the OpenPI policy input format."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )

    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": task_description,
    }


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_eval(
    cfg: LiberoRLTTrainConfig,
    *,
    checkpoint_payload: dict[str, Any],
    episodes: int,
    deterministic: bool = True,
    start_episode_idx: int = 0,
    max_env_steps_per_episode: int | None = None,
    eval_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run evaluation episodes and return aggregated metrics.

    Returns dict with:
        success_rate, mean_return, mean_steps, episodes_run, episode_results
    """
    if eval_logger is None:
        eval_logger = logger

    rlt = cfg.rlt
    chunk_size = rlt.chunk_size
    execute_horizon = rlt.execute_horizon
    action_dim = rlt.action_dim

    vla_client = VLAFeatureClient(
        host=rlt.eval_vla_server_host,
        port=rlt.eval_vla_server_port,
        logger=eval_logger,
    )
    eval_logger.info(
        "Connected to VLA feature server at %s:%s",
        rlt.eval_vla_server_host,
        rlt.eval_vla_server_port,
    )

    agent = create_rlt_agent_from_cfg(cfg)
    apply_checkpoint_payload_to_agent(agent, checkpoint_payload, load_optimizers=False)

    async_eval_env = cfg.training.async_eval.env
    eval_env_cfg = replace(
        cfg.env,
        backend=async_eval_env.backend,
        remote=async_eval_env.remote,
    )
    eval_cfg = replace(cfg, env=eval_env_cfg)
    env = create_env(eval_cfg, eval_logger)
    task_description = env.current_instruction
    eval_logger.info("Loaded environment with task description: %s", task_description)

    episode_results = []
    successes = 0
    total_return = 0.0
    total_steps = 0

    try:
        for ep_idx in range(episodes):
            obs = env.reset(seed=cfg.env.seed, init_episode_idx=start_episode_idx + ep_idx)
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            done = False

            while not done:
                processed_obs = prepare_vla_obs(obs, task_description)
                features = vla_client.infer(processed_obs)

                rlt_obs = build_rlt_obs(
                    z_rl=features["z_rl"],
                    proprio=features["proprio"],
                    reference_action=features["reference_action"],
                )

                action_chunk_flat = agent.sample_action(rlt_obs, deterministic=deterministic)
                action_chunk = action_chunk_flat.reshape(chunk_size, action_dim)

                for step_idx in range(execute_horizon):
                    next_obs, reward, terminated, truncated, info = env.step(action_chunk[step_idx])
                    episode_return += float(reward)
                    episode_steps += 1

                    env_done = bool(info.get("env_done", False))
                    episode_success = episode_success or env_done

                    if terminated or truncated or env_done:
                        done = True
                        break

                    if max_env_steps_per_episode and episode_steps >= max_env_steps_per_episode:
                        done = True
                        break

                obs = next_obs
                if done:
                    break

            successes += int(episode_success)
            total_return += episode_return
            total_steps += episode_steps
            episode_results.append({
                "episode_idx": ep_idx,
                "success": int(episode_success),
                "return": episode_return,
                "steps": episode_steps,
            })
            eval_logger.info(
                "eval ep %d/%d: success=%s return=%.2f steps=%d",
                ep_idx + 1,
                episodes,
                episode_success,
                episode_return,
                episode_steps,
            )

    finally:
        try:
            env.close(clear_cache=False)
        except Exception:
            pass
        try:
            vla_client.close()
        except Exception:
            pass

    success_rate = successes / max(1, episodes)
    mean_return = total_return / max(1, episodes)
    mean_steps = total_steps / max(1, episodes)

    eval_logger.info(
        "eval complete: %d episodes, success_rate=%.3f, mean_return=%.2f, mean_steps=%.1f",
        episodes, success_rate, mean_return, mean_steps,
    )

    return {
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_steps": mean_steps,
        "episodes_run": episodes,
        "successes": successes,
        "episode_results": episode_results,
    }
