"""Online replay prefill helpers for LIBERO warmup reuse."""
from __future__ import annotations

import json
import logging
import pickle
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from ..policy import (
    LiberoObservationCache,
    build_residual_step_core,
    select_action_chunk_window,
)
from ..utils.obs_utils import _clone_obs_dict, _zero_obs_like
from ..utils.profiling import _RuntimeProfiler, _build_residual_step_obs_profiled

if TYPE_CHECKING:
    from serl_launcher.data.replay_buffer import ReplayBuffer


ONLINE_PREFILL_EPISODE_FORMAT = "libero_online_prefill_episode_v1"
ONLINE_PREFILL_MANIFEST_FORMAT = "libero_online_prefill_manifest_v1"


def _resolve_online_prefill_paths(dataset_paths: Any, base_dir: Path) -> List[Path]:
    resolved: List[Path] = []
    if dataset_paths is None:
        return resolved

    if isinstance(dataset_paths, (str, Path)):
        items = [dataset_paths]
    else:
        items = list(dataset_paths)

    for item in items:
        candidate = Path(str(item)).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if candidate.is_file():
            if candidate.name == "manifest.json":
                with open(candidate, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for episode_file in manifest.get("episode_files", []):
                    episode_path = Path(str(episode_file)).expanduser()
                    if not episode_path.is_absolute():
                        episode_path = (candidate.parent / episode_path).resolve()
                    else:
                        episode_path = episode_path.resolve()
                    resolved.append(episode_path)
            elif candidate.suffix == ".pkl":
                resolved.append(candidate)
        elif candidate.is_dir():
            manifest_path = candidate / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for episode_file in manifest.get("episode_files", []):
                    episode_path = Path(str(episode_file)).expanduser()
                    if not episode_path.is_absolute():
                        episode_path = (manifest_path.parent / episode_path).resolve()
                    else:
                        episode_path = episode_path.resolve()
                    resolved.append(episode_path)
            else:
                resolved.extend(sorted(candidate.glob("episode_*.pkl")))

    deduped: List[Path] = []
    seen = set()
    for path in resolved:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _build_prefill_frame_obs(
    payload: Dict[str, Any], frame_idx: int
) -> Dict[str, np.ndarray]:
    return {
        "agentview_rgb": np.asarray(
            payload["agentview_rgb"][frame_idx],
            dtype=np.uint8,
        ),
        "eye_in_hand_rgb": np.asarray(
            payload["eye_in_hand_rgb"][frame_idx],
            dtype=np.uint8,
        ),
        "ee_pos": np.asarray(payload["ee_pos"][frame_idx], dtype=np.float32),
        "ee_ori": np.asarray(payload["ee_ori"][frame_idx], dtype=np.float32),
        "gripper_states": np.asarray(
            payload["gripper_states"][frame_idx],
            dtype=np.float32,
        ),
    }


def _resolve_online_prefill_mode(*, chunk_step_enabled: bool) -> str:
    return "stepchunk" if bool(chunk_step_enabled) else "step"


def _get_prefill_base_chunk_for_start(
    payload: Dict[str, Any],
    *,
    chunk_start: int,
    chunk_horizon: int,
    full_action_dim: int,
) -> np.ndarray:
    stored_base_chunks = payload.get("base_chunks", None)
    if stored_base_chunks is None:
        raise KeyError("online prefill episode is missing required key 'base_chunks'")

    stored_horizon = int(payload.get("chunk_horizon", chunk_horizon))
    if stored_horizon != int(chunk_horizon):
        raise ValueError(
            "online prefill chunk_horizon does not match training config: "
            f"payload={stored_horizon} config={int(chunk_horizon)}"
        )

    base_chunks = np.asarray(stored_base_chunks, dtype=np.float32)
    if base_chunks.ndim != 3:
        raise ValueError(
            "online prefill base_chunks must be rank-3, got "
            f"shape={base_chunks.shape}"
        )

    chunk_index = int(chunk_start // chunk_horizon)
    if chunk_index >= base_chunks.shape[0]:
        raise IndexError(
            "online prefill base_chunks are shorter than action sequence: "
            f"chunk_index={chunk_index} available={base_chunks.shape[0]}"
        )

    return select_action_chunk_window(
        base_chunks[chunk_index],
        horizon=chunk_horizon,
        action_dim=full_action_dim,
    )


def _load_online_prefill_buffer(
    cfg: DictConfig,
    *,
    replay_buffer: "ReplayBuffer",
    sample_obs_template: Dict[str, np.ndarray],
    action_dim: int,
    chunk_horizon: int,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    chunk_step_enabled: bool,
    logger: logging.Logger,
    normalizer: Optional[Any] = None,
    profiler: Optional[_RuntimeProfiler] = None,
    max_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    del sample_obs_template
    stats: Dict[str, Any] = {
        "enabled": 1,
        "mode": _resolve_online_prefill_mode(chunk_step_enabled=chunk_step_enabled),
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "episodes_loaded": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
        "success_episodes": 0,
        "episode_return_sum": 0.0,
        "episode_step_sum": 0,
        "recent_episode_successes": [],
    }

    training_cfg = cfg.get("training", {})
    prefill_cfg = training_cfg.get("online_prefill", None)
    dataset_paths = None if prefill_cfg is None else prefill_cfg.get("dataset_paths", None)
    prefill_paths = _resolve_online_prefill_paths(dataset_paths, Path.cwd())
    stats["files_total"] = len(prefill_paths)
    if not prefill_paths:
        logger.warning(
            "training.online_prefill.enabled=true but training.online_prefill.dataset_paths is empty"
        )
        return stats

    task_key = f"{cfg.task.suite_name}_task_{int(cfg.task.task_id)}"
    expected_mode = _resolve_online_prefill_mode(chunk_step_enabled=chunk_step_enabled)
    obs_cache = LiberoObservationCache(max_obs_entries=256, max_step_obs_entries=512)
    recent_successes: Deque[int] = deque(maxlen=20)

    logger.info(
        "online prefill dataset_paths resolved: %d episode PKL files found",
        len(prefill_paths),
    )
    for path in prefill_paths:
        if max_episodes is not None and stats["episodes_loaded"] >= int(max_episodes):
            break
        if not path.exists():
            stats["files_missing"] += 1
            logger.warning("online prefill dataset not found: %s", path)
            continue

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["skipped"] += 1
            logger.warning("failed to load online prefill dataset %s: %s", path, exc)
            continue

        try:
            if (
                not isinstance(payload, dict)
                or payload.get("format") != ONLINE_PREFILL_EPISODE_FORMAT
            ):
                raise ValueError("unsupported online prefill payload format")

            payload_mode = str(payload.get("mode", "")).strip().lower()
            if payload_mode != expected_mode:
                raise ValueError(
                    "online prefill mode does not match training config: "
                    f"payload={payload_mode!r} expected={expected_mode!r}"
                )

            payload_task_key = str(payload.get("task_key", "")).strip()
            if payload_task_key and payload_task_key != task_key:
                raise ValueError(
                    "online prefill task key does not match training config: "
                    f"payload={payload_task_key!r} expected={task_key!r}"
                )

            payload_action_dim = int(payload.get("action_dim", action_dim))
            if payload_action_dim != int(action_dim):
                raise ValueError(
                    "online prefill action_dim does not match training config: "
                    f"payload={payload_action_dim} expected={int(action_dim)}"
                )

            actions = np.asarray(payload.get("actions", []), dtype=np.float32)
            rewards = np.asarray(
                payload.get("rewards", np.zeros((actions.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            dones = np.asarray(
                payload.get("dones", np.zeros((actions.shape[0],), dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise ValueError(
                    f"invalid action array in online prefill payload: {actions.shape}"
                )
            if actions.shape[1] != int(action_dim):
                raise ValueError(
                    "online prefill action dim does not match env.action_dim: "
                    f"path={path} dataset_dim={int(actions.shape[1])} "
                    f"env_action_dim={int(action_dim)}"
                )
            required_obs_keys = (
                "agentview_rgb",
                "eye_in_hand_rgb",
                "ee_pos",
                "ee_ori",
                "gripper_states",
            )
            for obs_key in required_obs_keys:
                obs_arr = np.asarray(payload.get(obs_key, []))
                if obs_arr.ndim == 0 or int(obs_arr.shape[0]) < int(actions.shape[0]):
                    raise ValueError(
                        "online prefill observation length is shorter than actions: "
                        f"key={obs_key!r} obs_len={int(obs_arr.shape[0]) if obs_arr.ndim > 0 else 0} "
                        f"action_len={int(actions.shape[0])}"
                    )
            if rewards.shape[0] < actions.shape[0]:
                padded_rewards = np.zeros((actions.shape[0],), dtype=np.float32)
                if rewards.shape[0] > 0:
                    padded_rewards[: rewards.shape[0]] = rewards
                rewards = padded_rewards
            if dones.shape[0] < actions.shape[0]:
                padded_dones = np.zeros((actions.shape[0],), dtype=bool)
                if dones.shape[0] > 0:
                    padded_dones[: dones.shape[0]] = dones
                padded_dones[-1] = True
                dones = padded_dones

            episode_id = int(payload.get("init_episode_idx", payload.get("episode_index", 0)))
            episode_success = int(bool(payload.get("episode_success", False)))
            episode_return = float(payload.get("episode_return", float(np.sum(rewards))))
            episode_steps = int(payload.get("episode_steps", int(actions.shape[0])))

            for step_idx in range(actions.shape[0]):
                stats["candidates"] += 1
                chunk_start = int((step_idx // chunk_horizon) * chunk_horizon)
                step_in_chunk = int(step_idx - chunk_start)
                done = bool(dones[step_idx]) or bool(step_idx >= (actions.shape[0] - 1))
                reward = float(rewards[step_idx]) if step_idx < rewards.shape[0] else 0.0
                obs_cache_key = (str(path), int(step_idx))
                obs_raw = _build_prefill_frame_obs(payload, step_idx)
                base_chunk = _get_prefill_base_chunk_for_start(
                    payload,
                    chunk_start=chunk_start,
                    chunk_horizon=chunk_horizon,
                    full_action_dim=action_dim,
                )
                base_action = np.asarray(base_chunk[step_in_chunk], dtype=np.float32)
                final_action = np.asarray(actions[step_idx], dtype=np.float32)

                if chunk_step_enabled:
                    replay_buffer.insert(
                        {
                            "obs_core": build_residual_step_core(
                                obs_raw,
                                image_keys=image_keys,
                                normalizer=normalizer,
                                obs_cache=obs_cache,
                                cache_key=obs_cache_key,
                            ),
                            "base_action": base_action,
                            "base_action_norm": (
                                base_action
                                if normalizer is None
                                else np.asarray(
                                    normalizer.normalize_action(base_action),
                                    dtype=np.float32,
                                )
                            ),
                            "actions": final_action.reshape(action_dim),
                            "rewards": np.float32(reward),
                            "dones": bool(done),
                            "alpha": np.float32(0.0),
                            "episode_id": int(episode_id),
                            "episode_step": int(step_idx),
                        }
                    )
                    stats["inserted"] += 1
                    continue

                obs_input = _build_residual_step_obs_profiled(
                    profiler,
                    obs_raw,
                    base_action,
                    image_keys=image_keys,
                    stack_horizon=stack_horizon,
                    normalizer=normalizer,
                    obs_cache=obs_cache,
                    cache_key=obs_cache_key,
                    alpha=0.0,
                )

                if done:
                    next_obs_input = _zero_obs_like(obs_input)
                    mask = 0.0
                else:
                    next_step_idx = int(step_idx + 1)
                    next_obs_raw = _build_prefill_frame_obs(payload, next_step_idx)
                    next_obs_cache_key = (str(path), next_step_idx)
                    next_chunk_start = int((next_step_idx // chunk_horizon) * chunk_horizon)
                    next_step_in_chunk = int(next_step_idx - next_chunk_start)
                    next_base_chunk = _get_prefill_base_chunk_for_start(
                        payload,
                        chunk_start=next_chunk_start,
                        chunk_horizon=chunk_horizon,
                        full_action_dim=action_dim,
                    )
                    next_obs_input = _build_residual_step_obs_profiled(
                        profiler,
                        next_obs_raw,
                        next_base_chunk[next_step_in_chunk],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                        cache_key=next_obs_cache_key,
                        alpha=0.0,
                    )
                    mask = 1.0

                replay_buffer.insert(
                    {
                        "observations": _clone_obs_dict(obs_input),
                        "actions": final_action.reshape(action_dim),
                        "next_observations": _clone_obs_dict(next_obs_input),
                        "rewards": np.float32(reward),
                        "masks": np.float32(mask),
                        "dones": bool(done),
                    }
                )
                stats["inserted"] += 1

            stats["files_loaded"] += 1
            stats["episodes_loaded"] += 1
            stats["success_episodes"] += episode_success
            stats["episode_return_sum"] += float(episode_return)
            stats["episode_step_sum"] += int(episode_steps)
            recent_successes.append(episode_success)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["skipped"] += 1
            logger.warning("online prefill conversion failed file=%s: %s", path, exc)
            continue

    stats["recent_episode_successes"] = list(recent_successes)
    return stats
