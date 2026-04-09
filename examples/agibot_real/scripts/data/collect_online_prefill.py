#!/usr/bin/env python3
"""Collect AgiBot online warmup/prefill rollouts into unified residual-training PKLs."""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import TypeVar

import hydra
import numpy as np
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[5]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_launcher.policy.factory import build_policy_backend_info
from serl_launcher.policy.factory import build_policy_client
from serl_launcher.residual.action import select_action_chunk_window
from serl_launcher.residual.data.materialize import build_residual_training_manifest
from serl_launcher.residual.data.materialize import materialize_with_config
from serl_launcher.training.seeding import set_global_seeds
from serl_torch.examples.agibot_real.config import resolve_agibot_cfg_task_key
from serl_torch.examples.agibot_real.env_wrappers.factory import _create_env
from serl_torch.examples.agibot_real.runtime.controller_rollout import (
    ControllerExecutedStep,
)
from serl_torch.examples.agibot_real.runtime.controller_rollout import (
    ControllerPlannedStep,
)
from serl_torch.examples.agibot_real.runtime.controller_rollout import (
    run_controller_episode,
)
from serl_torch.examples.agibot_real.runtime.policy_adapter import build_agibot_policy_input
from serl_torch.examples.agibot_real.training_config import AGIBOT_ONLINE_TRAINING_CONFIG

_T = TypeVar("_T")
DEFAULT_CONF_DIR = Path(__file__).resolve().parents[2] / "conf"


def _progress(
    iterable: Iterable[_T],
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    leave: bool = True,
) -> Iterable[_T]:
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=leave)


def _resolve_config_path(config_arg: str) -> Path:
    candidate = Path(str(config_arg)).expanduser()
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_absolute() or "/" in str(config_arg):
        return candidate.resolve()
    return (DEFAULT_CONF_DIR / candidate).resolve()


def _compose_config(config_path: Path, overrides: List[str]):
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        cfg = hydra.compose(config_name=config_path.stem, overrides=list(overrides))
    return cfg


def _append_frame(buffers: Dict[str, List[np.ndarray]], obs_raw: Dict[str, Any]) -> None:
    buffers["head_image"].append(np.asarray(obs_raw["image/head"], dtype=np.uint8).copy())
    buffers["left_wrist_image"].append(np.asarray(obs_raw["image/left_wrist"], dtype=np.uint8).copy())
    buffers["right_wrist_image"].append(np.asarray(obs_raw["image/right_wrist"], dtype=np.uint8).copy())
    buffers["pose"].append(np.asarray(obs_raw["state/pose"], dtype=np.float32).copy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect AgiBot warmup/prefill episodes into residual-training PKLs",
    )
    parser.add_argument("config", type=str, help="Training config yaml name or absolute path")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "residual_training" / "online"),
    )
    args, unknown = parser.parse_known_args()
    invalid_flags = [token for token in unknown if token.startswith("-")]
    if invalid_flags:
        parser.error("unrecognized arguments: " + " ".join(invalid_flags))
    overrides = list(unknown)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("agibot_collect_online_prefill")

    config_path = _resolve_config_path(args.config)
    cfg = _compose_config(config_path, overrides)
    set_global_seeds(int(cfg.seed))

    warmup_cfg = cfg.training.get("warmup", None)
    default_episodes = int(warmup_cfg.get("episodes", 0)) if warmup_cfg is not None else 0
    num_episodes = int(args.episodes) if args.episodes is not None else int(default_episodes)
    if num_episodes <= 0:
        raise ValueError("online residual-training collection requires a positive episode count")

    task_key = resolve_agibot_cfg_task_key(cfg)
    chunk_horizon = int(cfg.residual.chunk_horizon)
    mode = "stepchunk" if int(chunk_horizon) > 1 else "step"
    output_root = Path(args.output_dir).expanduser().resolve()
    task_output_dir = output_root / task_key / mode
    task_output_dir.mkdir(parents=True, exist_ok=True)

    env = _create_env(cfg, logger)
    controller_enabled = bool(getattr(env, "controller_enabled", False))
    policy_backend_info = build_policy_backend_info(cfg)
    policy_client = build_policy_client(cfg, logger=logger)
    logger.info(
        "Collecting online prefill: task=%s episodes=%s output=%s",
        task_key,
        num_episodes,
        task_output_dir,
    )

    manifest_files: List[str] = []
    total_frames = 0
    success_episodes = 0
    t0 = time.time()

    try:
        episode_iter = _progress(range(num_episodes), total=num_episodes, desc=f"{task_key}:{mode}", unit="ep")
        for episode_index in episode_iter:
            seed = int(cfg.task.seed_base) + int(episode_index)
            obs_raw = env.reset(seed=seed, init_episode_idx=episode_index)
            prompt = str(env.current_instruction)
            buffers: Dict[str, List[np.ndarray]] = {
                "head_image": [],
                "left_wrist_image": [],
                "right_wrist_image": [],
                "pose": [],
                "actions": [],
                "rewards": [],
                "dones": [],
            }
            base_chunks: List[np.ndarray] = []
            episode_steps = 0
            episode_return = 0.0
            success = False
            max_episode_steps = int(env.step_limit)
            if cfg.training.max_env_steps_per_episode is not None:
                max_episode_steps = min(max_episode_steps, int(cfg.training.max_env_steps_per_episode))

            if controller_enabled:
                def _plan_prefill_chunk(
                    controller_obs: Dict[str, Any],
                    remaining_steps: int,
                ) -> list[ControllerPlannedStep]:
                    action_chunk, _ = policy_client.infer_chunk(
                        build_agibot_policy_input(controller_obs, prompt)
                    )
                    base_chunk = select_action_chunk_window(action_chunk, horizon=chunk_horizon)
                    execute_horizon = int(min(int(base_chunk.shape[0]), int(remaining_steps)))
                    final_chunk = np.asarray(base_chunk[:execute_horizon], dtype=np.float32)
                    sequence_ids = env.enqueue_action_chunk(final_chunk)
                    accepted_horizon = int(len(sequence_ids))
                    if accepted_horizon <= 0:
                        logger.warning(
                            "Prefill controller enqueue accepted no actions; "
                            "the operator may have changed controller state during planning."
                        )
                        return []
                    if accepted_horizon != execute_horizon:
                        logger.warning(
                            "Prefill controller enqueue accepted %s/%s actions; truncating the plan.",
                            accepted_horizon,
                            execute_horizon,
                        )
                        execute_horizon = int(accepted_horizon)
                        final_chunk = final_chunk[:execute_horizon]
                    base_chunks.append(np.asarray(final_chunk, dtype=np.float32))
                    return [
                        ControllerPlannedStep(
                            sequence_id=int(sequence_id),
                            obs_before=(controller_obs if chunk_step == 0 else None),
                            final_action=np.asarray(final_chunk[chunk_step], dtype=np.float32),
                            chunk_step=int(chunk_step),
                            executed_horizon=int(execute_horizon),
                        )
                        for chunk_step, sequence_id in enumerate(sequence_ids)
                    ]

                def _on_prefill_step(executed: ControllerExecutedStep, _current_step: int) -> None:
                    obs_before = executed.planned.obs_before
                    if obs_before is None:
                        raise RuntimeError(
                            "controller prefill step is missing the pre-action observation"
                        )
                    _append_frame(buffers, obs_before)
                    buffers["actions"].append(
                        np.asarray(executed.planned.final_action, dtype=np.float32).copy()
                    )
                    buffers["rewards"].append(float(executed.reward))
                    buffers["dones"].append(bool(executed.done or executed.truncated))

                controller_summary = run_controller_episode(
                    env=env,
                    initial_obs=obs_raw,
                    max_episode_steps=max_episode_steps,
                    chunk_horizon=chunk_horizon,
                    cfg=cfg,
                    logger=logger,
                    plan_chunk_fn=_plan_prefill_chunk,
                    on_step_fn=_on_prefill_step,
                )
                episode_steps = int(controller_summary.episode_steps)
                episode_return = float(controller_summary.episode_return)
                success = bool(controller_summary.success)
            else:
                while episode_steps < max_episode_steps:
                    action_chunk, _ = policy_client.infer_chunk(
                        build_agibot_policy_input(obs_raw, prompt)
                    )
                    base_chunk = select_action_chunk_window(action_chunk, horizon=chunk_horizon)
                    base_chunks.append(np.asarray(base_chunk, dtype=np.float32))

                    decision_done = False
                    for chunk_step in range(int(base_chunk.shape[0])):
                        if episode_steps >= max_episode_steps:
                            decision_done = True
                            break
                        _append_frame(buffers, obs_raw)
                        final_action = np.asarray(base_chunk[chunk_step], dtype=np.float32)
                        next_obs_raw, reward, done, truncated, info = env.step(final_action)
                        buffers["actions"].append(final_action)
                        buffers["rewards"].append(float(reward))
                        buffers["dones"].append(bool(done or truncated))
                        episode_steps += 1
                        episode_return += float(reward)
                        success = bool(info.get("success", success))
                        obs_raw = next_obs_raw
                        if done or truncated:
                            decision_done = True
                            break
                    if decision_done:
                        break

            payload = materialize_with_config(
                {
                    "source": "online_prefill",
                    "suite_name": "agibot_real",
                    "task_id": 0,
                    "task_key": task_key,
                    "task_description": prompt,
                    "prompt": prompt,
                    "alpha": 0.0,
                    "head_image": np.asarray(buffers["head_image"], dtype=np.uint8),
                    "left_wrist_image": np.asarray(buffers["left_wrist_image"], dtype=np.uint8),
                    "right_wrist_image": np.asarray(buffers["right_wrist_image"], dtype=np.uint8),
                    "pose": np.asarray(buffers["pose"], dtype=np.float32),
                    "base_chunks": np.asarray(base_chunks, dtype=np.float32),
                    "actions": np.asarray(buffers["actions"], dtype=np.float32),
                    "rewards": np.asarray(buffers["rewards"], dtype=np.float32),
                    "dones": np.asarray(buffers["dones"], dtype=bool),
                    "episode_index": int(episode_index),
                    "episode_steps": int(episode_steps),
                    "episode_return": float(episode_return),
                    "episode_success": bool(success),
                    "metadata": {
                        "source_episode_format": "agibot_online_prefill",
                        "base_policy_type": str(policy_backend_info["type"]),
                        "base_policy_id": str(policy_backend_info["id"]),
                        "task_name": str(cfg.task.name),
                        "controller_enabled": bool(controller_enabled),
                    },
                },
                data_config=AGIBOT_ONLINE_TRAINING_CONFIG,
            )
            file_name = f"episode_{episode_index:06d}.pkl"
            episode_path = task_output_dir / file_name
            with episode_path.open("wb") as f:
                pickle.dump(payload, f)
            manifest_files.append(file_name)
            total_frames += int(payload["episode"]["steps"])
            success_episodes += int(bool(payload["episode"]["success"]))
    finally:
        env.close(clear_cache=False)

    manifest = build_residual_training_manifest(
        schema=AGIBOT_ONLINE_TRAINING_CONFIG.schema,
        source="online_prefill",
        task_key=task_key,
        suite_name="agibot_real",
        task_id=0,
        task_description=str(cfg.task.prompt),
        chunk_horizon=int(chunk_horizon),
        action_dim=int(cfg.env.action_dim),
        num_episodes=int(len(manifest_files)),
        total_frames=int(total_frames),
        episode_files=manifest_files,
        metadata={
            "task_name": str(cfg.task.name),
            "base_policy_type": str(policy_backend_info["type"]),
            "base_policy_id": str(policy_backend_info["id"]),
            "success_episodes": int(success_episodes),
            "controller_enabled": bool(controller_enabled),
            "elapsed_sec": float(time.time() - t0),
        },
    )
    manifest_path = task_output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(
        "Collected online prefill: task_key=%s episodes=%s frames=%s manifest=%s",
        task_key,
        len(manifest_files),
        total_frames,
        manifest_path,
    )


if __name__ == "__main__":
    main()
