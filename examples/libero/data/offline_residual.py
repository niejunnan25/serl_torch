"""Offline residual dataset conversion helpers."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from serl_launcher.data.episode_paths import resolve_episode_files
from serl_launcher.data.normalizer import StateActionNormalizer
from serl_launcher.residual.data.action_projection import project_expert_action
from serl_launcher.residual.data.transitions import (
    build_step_transition,
    build_stepchunk_transition,
)

from ..policy import (
    LiberoObservationCache,
    OpenPIChunkClient,
    build_residual_step_core,
    select_action_chunk_window,
)
from ..utils.alpha_utils import validate_alpha
from ..utils.obs_utils import _zero_obs_like
from ..utils.profiling import _RuntimeProfiler, _build_residual_step_obs_profiled

if TYPE_CHECKING:
    from serl_launcher.data.replay_buffer import ReplayBuffer


def _build_offline_frame_obs(payload: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
    return {
        "agentview_rgb": np.asarray(
            payload["agentview_rgb"][frame_idx], dtype=np.uint8
        ),
        "eye_in_hand_rgb": np.asarray(
            payload["eye_in_hand_rgb"][frame_idx], dtype=np.uint8
        ),
        "ee_pos": np.asarray(payload["ee_pos"][frame_idx], dtype=np.float32),
        "ee_ori": np.asarray(payload["ee_ori"][frame_idx], dtype=np.float32),
        "gripper_states": np.asarray(
            payload["gripper_states"][frame_idx], dtype=np.float32
        ),
    }


def _get_episode_prompt(payload: Dict[str, Any], fallback_prompt: str) -> str:
    prompt = payload.get("task_description", payload.get("prompt", fallback_prompt))
    return str(prompt)


def _get_base_chunk_for_start(
    payload: Dict[str, Any],
    *,
    chunk_start: int,
    chunk_horizon: int,
    full_action_dim: int,
    prompt: str,
    openpi_client: OpenPIChunkClient,
    cache: Dict[int, np.ndarray],
    obs_cache: Optional[LiberoObservationCache] = None,
    obs_cache_key: Optional[Any] = None,
) -> np.ndarray:
    if chunk_start in cache:
        return cache[chunk_start]

    stored_base_chunks = payload.get("base_chunks", None)
    stored_horizon = int(payload.get("chunk_horizon", chunk_horizon))
    if stored_base_chunks is not None and stored_horizon == int(chunk_horizon):
        chunk_index = int(chunk_start // chunk_horizon)
        base_chunks = np.asarray(stored_base_chunks, dtype=np.float32)
        if base_chunks.ndim == 3 and chunk_index < base_chunks.shape[0]:
            chunk = select_action_chunk_window(
                base_chunks[chunk_index],
                horizon=chunk_horizon,
                action_dim=full_action_dim,
            )
            cache[chunk_start] = chunk
            return chunk

    obs_raw = _build_offline_frame_obs(payload, chunk_start)
    openpi_chunk, _ = openpi_client.infer_chunk(
        obs_raw,
        prompt,
        obs_cache=obs_cache,
        cache_key=obs_cache_key,
    )
    chunk = select_action_chunk_window(
        openpi_chunk,
        horizon=chunk_horizon,
        action_dim=full_action_dim,
    )
    cache[chunk_start] = chunk
    return chunk


def _load_offline_residual_buffer(
    cfg: DictConfig,
    *,
    sample_obs_template: Dict[str, np.ndarray],
    offline_buffer: "ReplayBuffer",
    action_dim: int,
    full_action_dim: int,
    chunk_horizon: int,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    residual_alpha: float,
    openpi_client: OpenPIChunkClient,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    chunk_step_enabled: bool,
    logger: logging.Logger,
    normalizer: Optional[StateActionNormalizer] = None,
    profiler: Optional[_RuntimeProfiler] = None,
    state_mode: str = "fused",
) -> Dict[str, int]:
    del sample_obs_template
    stats = {
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "clipped_values": 0,
        "errors": 0,
    }

    offline_paths = resolve_episode_files(cfg.offline.dataset_paths, base_dir=Path.cwd())
    stats["files_total"] = len(offline_paths)
    if not offline_paths:
        logger.warning("offline.enabled=true but offline.dataset_paths is empty")
        return stats

    max_transitions = (
        int(cfg.offline.max_transitions)
        if cfg.offline.max_transitions is not None
        else None
    )
    expert_reference_scale = max(
        float(cfg.offline.get("expert_reference_scale", 1.0)), 1e-6
    )
    alpha = validate_alpha(
        residual_alpha,
        name="offline residual alpha",
        allow_zero=False,
    )
    denom = residual_limits * alpha * expert_reference_scale
    clip_residual_to_unit = bool(cfg.offline.get("clip_residual_to_unit", True))
    fallback_prompt = str(getattr(cfg.task, "prompt", ""))
    obs_cache = LiberoObservationCache(max_obs_entries=256, max_step_obs_entries=512)

    logger.info(
        "offline dataset_paths resolved: %d episode PKL files found", len(offline_paths)
    )
    for path in offline_paths:
        if max_transitions is not None and stats["inserted"] >= max_transitions:
            break
        if not path.exists():
            stats["files_missing"] += 1
            logger.warning("offline dataset not found: %s", path)
            continue

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["skipped"] += 1
            logger.warning("failed to load offline dataset %s: %s", path, exc)
            continue

        if (
            not isinstance(payload, dict)
            or payload.get("format") != "libero_offline_episode_v1"
        ):
            stats["skipped"] += 1
            logger.warning("unsupported offline payload format: %s", path)
            continue

        actions = np.asarray(payload.get("actions", []), dtype=np.float32)
        rewards = np.asarray(
            payload.get("rewards", np.zeros((actions.shape[0],))), dtype=np.float32
        ).reshape(-1)
        dones = np.asarray(
            payload.get("dones", np.zeros((actions.shape[0],))), dtype=bool
        ).reshape(-1)
        if actions.ndim != 2 or actions.shape[0] == 0:
            stats["skipped"] += 1
            logger.warning(
                "invalid action array in offline payload %s: %s", path, actions.shape
            )
            continue
        if actions.shape[1] != int(full_action_dim):
            raise ValueError(
                "offline dataset action dim does not match env.action_dim: "
                f"path={path} dataset_dim={int(actions.shape[1])} env_action_dim={int(full_action_dim)}"
            )

        prompt = _get_episode_prompt(payload, fallback_prompt)
        base_chunk_cache: Dict[int, np.ndarray] = {}
        frame_cache_prefix = str(path)
        stats["files_loaded"] += 1

        for step_idx in range(actions.shape[0]):
            if max_transitions is not None and stats["inserted"] >= max_transitions:
                break

            stats["candidates"] += 1
            chunk_start = int((step_idx // chunk_horizon) * chunk_horizon)
            step_in_chunk = int(step_idx - chunk_start)

            try:
                obs_cache_key = (frame_cache_prefix, int(step_idx))
                obs_raw = _build_offline_frame_obs(payload, step_idx)
                expert_chunk = select_action_chunk_window(
                    actions[chunk_start : chunk_start + chunk_horizon],
                    horizon=chunk_horizon,
                    action_dim=full_action_dim,
                )
                base_chunk = _get_base_chunk_for_start(
                    payload,
                    chunk_start=chunk_start,
                    chunk_horizon=chunk_horizon,
                    full_action_dim=full_action_dim,
                    prompt=prompt,
                    openpi_client=openpi_client,
                    cache=base_chunk_cache,
                    obs_cache=obs_cache,
                    obs_cache_key=(frame_cache_prefix, int(chunk_start)),
                )
                base_action = base_chunk[step_in_chunk]
                expert_action = expert_chunk[step_in_chunk]
                is_last_step = bool(step_idx >= (actions.shape[0] - 1))
                done = bool(dones[step_idx]) or is_last_step
                reward = (
                    float(rewards[step_idx])
                    if step_idx < rewards.shape[0]
                    else float(done)
                )
                projected_expert_action, clipped_count = project_expert_action(
                    expert_action=expert_action,
                    base_action=base_action,
                    control_indices=control_indices,
                    denom=denom,
                    clip_residual_to_unit=clip_residual_to_unit,
                )
                stats["clipped_values"] += int(clipped_count)

                if chunk_step_enabled:
                    offline_buffer.insert(
                        build_stepchunk_transition(
                            obs_core=build_residual_step_core(
                                obs_raw,
                                image_keys=image_keys,
                                normalizer=normalizer,
                                obs_cache=obs_cache,
                                cache_key=obs_cache_key,
                            ),
                            base_action=base_action,
                            actions=projected_expert_action.reshape(full_action_dim),
                            reward=reward,
                            done=done,
                            alpha=float(residual_alpha),
                            episode_id=int(stats["files_loaded"] - 1),
                            episode_step=int(step_idx),
                            normalizer=normalizer,
                        )
                    )
                else:
                    obs_input = _build_residual_step_obs_profiled(
                        profiler,
                        obs_raw,
                        base_action,
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                        obs_cache=obs_cache,
                        cache_key=obs_cache_key,
                        alpha=float(residual_alpha),
                        state_mode=state_mode,
                    )

                    if done:
                        next_obs_input = _zero_obs_like(obs_input)
                        mask = 0.0
                    else:
                        next_obs_cache_key = (frame_cache_prefix, int(step_idx + 1))
                        next_obs_raw = _build_offline_frame_obs(payload, step_idx + 1)
                        next_chunk_start = int(
                            ((step_idx + 1) // chunk_horizon) * chunk_horizon
                        )
                        next_step_in_chunk = int((step_idx + 1) - next_chunk_start)
                        if next_chunk_start == chunk_start:
                            next_base_chunk = base_chunk
                        else:
                            next_base_chunk = _get_base_chunk_for_start(
                                payload,
                                chunk_start=next_chunk_start,
                                chunk_horizon=chunk_horizon,
                                full_action_dim=full_action_dim,
                                prompt=prompt,
                                openpi_client=openpi_client,
                                cache=base_chunk_cache,
                                obs_cache=obs_cache,
                                obs_cache_key=(
                                    frame_cache_prefix,
                                    int(next_chunk_start),
                                ),
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
                            alpha=float(residual_alpha),
                            state_mode=state_mode,
                        )
                        mask = 1.0

                    offline_buffer.insert(
                        build_step_transition(
                            observations=obs_input,
                            actions=projected_expert_action.reshape(full_action_dim),
                            next_observations=next_obs_input,
                            reward=reward,
                            done=done,
                            mask=mask,
                        )
                    )
                stats["inserted"] += 1
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                stats["skipped"] += 1
                logger.warning(
                    "offline conversion failed file=%s step=%s: %s", path, step_idx, exc
                )
                continue
    return stats
