#!/usr/bin/env python3
"""Materialize unified residual-training episode PKLs from online LIBERO rollouts."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, TypeVar

import hydra
import numpy as np
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.policy.factory import build_policy_backend_info
from serl_launcher.policy.factory import build_policy_client
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.data.materialize import build_residual_training_manifest
from serl_launcher.residual.data.materialize import materialize_with_config
from serl_launcher.training.seeding import set_global_seeds
from serl_torch.examples.libero.training_config import LIBERO_ONLINE_TRAINING_CONFIG
from serl_torch.examples.libero.env_wrappers.factory import _create_env
from serl_torch.examples.libero.runtime.policy_adapter import build_libero_policy_input

_T = TypeVar("_T")
DEFAULT_CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _progress(
    iterable: Iterable[_T],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    leave: bool = True,
) -> Iterable[_T]:
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
    )


def _resolve_config_path(config_arg: str) -> Path:
    candidate = Path(str(config_arg)).expanduser()
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_absolute() or "/" in str(config_arg):
        return candidate.resolve()
    return (DEFAULT_CONF_DIR / candidate).resolve()


def _resolve_collection_mode(*, chunk_step_enabled: bool) -> str:
    return "stepchunk" if bool(chunk_step_enabled) else "step"


def _compose_config(config_path: Path, overrides: List[str]):
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(config_path.parent),
    ):
        cfg = hydra.compose(
            config_name=config_path.stem,
            overrides=list(overrides),
        )
    return cfg


def _append_frame(buffers: Dict[str, List[np.ndarray]], obs_raw: Dict[str, Any]) -> None:
    frame = _canonicalize_obs_frame(obs_raw)
    buffers["agentview_rgb"].append(frame["agentview_rgb"])
    buffers["eye_in_hand_rgb"].append(frame["eye_in_hand_rgb"])
    buffers["ee_pos"].append(frame["ee_pos"])
    buffers["ee_ori"].append(frame["ee_ori"])
    buffers["gripper_states"].append(frame["gripper_states"])


def _find_first_key(obs: Dict[str, Any], candidates: Tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(
        f"Missing keys {candidates} in observation. Available keys: {list(obs.keys())}"
    )


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float32).copy()
    quat_arr[3] = np.clip(quat_arr[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat_arr[3] * quat_arr[3])
    if np.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (
        quat_arr[:3] * 2.0 * np.arccos(float(quat_arr[3])) / float(den)
    ).astype(np.float32)


def _canonicalize_obs_frame(obs_raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    if "robot0_eef_quat" in obs_raw:
        ee_ori = _quat2axisangle(obs_raw["robot0_eef_quat"])
    else:
        ee_ori = np.asarray(
            _find_first_key(
                obs_raw,
                ("robot0_eef_axis_angle", "ee_ori", "eef_axis_angle"),
            ),
            dtype=np.float32,
        )

    return {
        "agentview_rgb": np.asarray(
            _find_first_key(
                obs_raw,
                ("agentview_rgb", "agentview_image", "image", "front_rgb"),
            ),
            dtype=np.uint8,
        ).copy(),
        "eye_in_hand_rgb": np.asarray(
            _find_first_key(
                obs_raw,
                (
                    "eye_in_hand_rgb",
                    "robot0_eye_in_hand_image",
                    "wrist_image",
                    "hand_rgb",
                ),
            ),
            dtype=np.uint8,
        ).copy(),
        "ee_pos": np.asarray(
            _find_first_key(obs_raw, ("robot0_eef_pos", "ee_pos", "eef_pos")),
            dtype=np.float32,
        ).copy(),
        "ee_ori": np.asarray(ee_ori, dtype=np.float32).copy(),
        "gripper_states": np.asarray(
            _find_first_key(
                obs_raw,
                ("robot0_gripper_qpos", "gripper_states", "gripper_qpos"),
            ),
            dtype=np.float32,
        ).copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect unified LIBERO residual-training episodes",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Training config yaml name or absolute path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of base-only warmup episodes to collect (defaults to training.warmup.episodes)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "residual_training"
            / "online"
        ),
        help="Root output directory for unified residual-training manifests and episode PKLs",
    )
    args, unknown = parser.parse_known_args()
    invalid_flags = [token for token in unknown if token.startswith("-")]
    if invalid_flags:
        parser.error("unrecognized arguments: " + " ".join(invalid_flags))
    overrides = list(unknown)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("libero_collect_residual_training")

    config_path = _resolve_config_path(args.config)
    cfg = _compose_config(config_path, overrides)
    set_global_seeds(int(cfg.seed))

    warmup_cfg = cfg.training.get("warmup", None)
    default_episodes = (
        int(warmup_cfg.get("episodes", 0)) if warmup_cfg is not None else 0
    )
    num_episodes = (
        int(args.episodes) if args.episodes is not None else int(default_episodes)
    )
    if num_episodes <= 0:
        raise ValueError(
            "online residual-training collection requires a positive episode count; "
            f"got {num_episodes}"
        )

    task_key = f"{cfg.task.suite_name}_task_{int(cfg.task.task_id)}"
    chunk_step_enabled = bool(cfg.get("chunk_step", {}).get("enabled", False))
    mode = _resolve_collection_mode(chunk_step_enabled=chunk_step_enabled)
    chunk_horizon = int(cfg.residual.chunk_horizon)
    action_dim = int(cfg.env.action_dim)

    output_root = Path(args.output_dir).expanduser().resolve()
    task_output_dir = output_root / task_key / mode
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Config path: %s", config_path)
    logger.info(
        "Collecting residual training episodes: task=%s mode=%s episodes=%s output=%s",
        task_key,
        mode,
        num_episodes,
        task_output_dir,
    )

    env = _create_env(cfg, logger)
    policy_backend_info = build_policy_backend_info(cfg)
    policy_client = build_policy_client(cfg, logger=logger)
    logger.info(
        "Chunk policy backend: type=%s id=%s",
        policy_backend_info["type"],
        policy_backend_info["id"],
    )

    seed_cursor = int(cfg.task.seed_base)
    init_episode_idx = 0
    manifest_files: List[str] = []
    total_frames = 0
    success_episodes = 0
    episode_return_sum = 0.0
    episode_step_sum = 0
    t0 = time.time()

    try:
        episode_iter = _progress(
            range(num_episodes),
            total=num_episodes,
            desc=f"{task_key}:{mode}",
            unit="ep",
            leave=True,
        )
        for episode_index in episode_iter:
            current_init_episode_idx = int(init_episode_idx)
            init_episode_idx += 1
            seed = int(seed_cursor)
            seed_cursor += 1

            obs_raw = env.reset(seed=seed, init_episode_idx=current_init_episode_idx)
            max_episode_steps = int(env.step_limit)
            if cfg.training.max_env_steps_per_episode is not None:
                max_episode_steps = min(
                    max_episode_steps,
                    int(cfg.training.max_env_steps_per_episode),
                )

            frame_buffers: Dict[str, List[np.ndarray]] = {
                "agentview_rgb": [],
                "eye_in_hand_rgb": [],
                "ee_pos": [],
                "ee_ori": [],
                "gripper_states": [],
            }
            actions: List[np.ndarray] = []
            rewards: List[float] = []
            dones: List[bool] = []
            base_chunks: List[np.ndarray] = []

            episode_success = False
            episode_return = 0.0
            episode_steps = 0
            episode_done = False

            while (episode_steps < max_episode_steps) and (not episode_done):
                openpi_chunk, _ = policy_client.infer_chunk(
                    build_libero_policy_input(
                        obs_raw,
                        env.current_instruction,
                    )
                )
                base_chunk = select_action_chunk_window(
                    openpi_chunk,
                    horizon=chunk_horizon,
                    action_dim=action_dim,
                )
                base_chunks.append(np.asarray(base_chunk, dtype=np.float32))

                if chunk_step_enabled:
                    execute_horizon = int(
                        min(chunk_horizon, max_episode_steps - episode_steps)
                    )
                    executed_base_chunk = np.asarray(
                        base_chunk[:execute_horizon],
                        dtype=np.float32,
                    )
                    chunk_result = env.step_chunk(executed_base_chunk)
                    chunk_observations = list(chunk_result["observations"])
                    next_obs_raw = chunk_result["obs"]
                    chunk_rewards = [float(v) for v in chunk_result["rewards"]]
                    chunk_infos = [dict(v) for v in chunk_result["infos"]]
                    chunk_dones = [bool(v) for v in chunk_result["dones"]]
                    actual_chunk_steps = int(len(chunk_rewards))
                    if actual_chunk_steps <= 0:
                        raise RuntimeError(
                            "env.step_chunk returned zero executed steps during online residual-training collection"
                        )
                    current_step_obs_raw = obs_raw
                    for chunk_step in range(actual_chunk_steps):
                        _append_frame(frame_buffers, current_step_obs_raw)
                        reward = float(chunk_rewards[chunk_step])
                        info = chunk_infos[chunk_step]
                        episode_steps += 1
                        episode_return += reward
                        episode_success = bool(info.get("success", episode_success))
                        timeout = bool(episode_steps >= max_episode_steps)
                        done = bool(chunk_dones[chunk_step] or timeout)
                        actions.append(
                            np.asarray(
                                executed_base_chunk[chunk_step], dtype=np.float32
                            )
                        )
                        rewards.append(reward)
                        dones.append(done)
                        if chunk_step < (actual_chunk_steps - 1):
                            current_step_obs_raw = chunk_observations[chunk_step]
                        if done:
                            episode_done = True
                            break
                    obs_raw = next_obs_raw
                    continue

                for chunk_step in range(chunk_horizon):
                    if episode_steps >= max_episode_steps:
                        episode_done = True
                        break
                    _append_frame(frame_buffers, obs_raw)
                    base_action = np.asarray(base_chunk[chunk_step], dtype=np.float32)
                    next_obs_raw, reward, env_done, _, info = env.step(base_action)
                    reward = float(reward)
                    episode_steps += 1
                    episode_return += reward
                    episode_success = bool(info["success"])
                    timeout = bool(episode_steps >= max_episode_steps)
                    done = bool(env_done or timeout)
                    actions.append(base_action.copy())
                    rewards.append(reward)
                    dones.append(done)
                    obs_raw = next_obs_raw
                    if done:
                        episode_done = True
                        break

            if dones:
                dones[-1] = True

            payload = materialize_with_config(
                {
                    "source": "online",
                    "suite_name": str(cfg.task.suite_name),
                    "task_id": int(cfg.task.task_id),
                    "task_key": task_key,
                    "task_description": str(env.current_instruction),
                    "prompt": str(env.current_instruction),
                    "alpha": 0.0,
                    "agentview_rgb": np.asarray(
                        frame_buffers["agentview_rgb"], dtype=np.uint8
                    ),
                    "eye_in_hand_rgb": np.asarray(
                        frame_buffers["eye_in_hand_rgb"], dtype=np.uint8
                    ),
                    "ee_pos": np.asarray(frame_buffers["ee_pos"], dtype=np.float32),
                    "ee_ori": np.asarray(frame_buffers["ee_ori"], dtype=np.float32),
                    "gripper_states": np.asarray(
                        frame_buffers["gripper_states"], dtype=np.float32
                    ),
                    "base_chunks": np.asarray(base_chunks, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "rewards": np.asarray(rewards, dtype=np.float32),
                    "dones": np.asarray(dones, dtype=bool),
                    "episode_index": int(episode_index),
                    "episode_steps": int(episode_steps),
                    "episode_return": float(episode_return),
                    "episode_success": bool(episode_success),
                    "metadata": {
                        "collection_mode": mode,
                        "base_policy_type": str(policy_backend_info["type"]),
                        "base_policy_id": str(policy_backend_info["id"]),
                        "seed": int(seed),
                        "applied_seed": (
                            int(env.last_seed) if env.last_seed is not None else int(seed)
                        ),
                        "init_episode_idx": int(current_init_episode_idx),
                        "init_state_idx": (
                            int(env.current_init_state_idx)
                            if env.current_init_state_idx is not None
                            else None
                        ),
                    },
                },
                data_config=LIBERO_ONLINE_TRAINING_CONFIG,
            )
            episode_path = task_output_dir / f"episode_{episode_index:06d}.pkl"
            with open(episode_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            manifest_files.append(str(episode_path))
            total_frames += int(payload["action"]["final"].shape[0])
            success_episodes += int(episode_success)
            episode_return_sum += float(episode_return)
            episode_step_sum += int(episode_steps)

    finally:
        env.close()

    manifest = build_residual_training_manifest(
        schema=LIBERO_ONLINE_TRAINING_CONFIG.schema,
        source="online",
        task_key=task_key,
        suite_name=str(cfg.task.suite_name),
        task_id=int(cfg.task.task_id),
        task_description=str(env.task_description),
        chunk_horizon=int(chunk_horizon),
        action_dim=int(action_dim),
        num_episodes=len(manifest_files),
        total_frames=int(total_frames),
        episode_files=manifest_files,
        metadata={
            "collection_mode": mode,
            "base_policy_type": str(policy_backend_info["type"]),
            "base_policy_id": str(policy_backend_info["id"]),
            "config_path": str(config_path),
            "overrides": list(overrides),
            "env_backend": str(cfg.env.backend),
            "env_host": str(cfg.env.get("remote", {}).get("host", "127.0.0.1"))
            if str(cfg.env.backend).lower() == "remote"
            else None,
            "env_port": int(cfg.env.get("remote", {}).get("port", 30000))
            if str(cfg.env.backend).lower() == "remote"
            else None,
            "success_episodes": int(success_episodes),
            "success_rate": float(success_episodes / max(1, len(manifest_files))),
            "mean_episode_return": float(
                episode_return_sum / max(1, len(manifest_files))
            ),
            "mean_episode_steps": float(
                episode_step_sum / max(1, len(manifest_files))
            ),
            "elapsed_sec": float(time.time() - t0),
        },
    )
    if str(policy_backend_info["type"]) == "openpi":
        manifest["metadata"]["openpi_host"] = str(cfg.openpi.host)
        manifest["metadata"]["openpi_port"] = int(cfg.openpi.port)
    manifest_path = task_output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "Collected residual training episodes: task=%s mode=%s episodes=%s frames=%s manifest=%s",
        task_key,
        mode,
        manifest["num_episodes"],
        manifest["total_frames"],
        manifest_path,
    )


if __name__ == "__main__":
    main()
