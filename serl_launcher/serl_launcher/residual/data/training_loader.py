"""Unified replay loader for residual training episodes."""
from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from serl_launcher.data.episode_paths import resolve_episode_files
from serl_launcher.residual.data.config import ResidualDataConfig
from serl_launcher.residual.data.materialize import build_step_core_from_payload
from serl_launcher.residual.data.materialize import validate_residual_training_payload
from serl_launcher.residual.data.transitions import build_step_transition
from serl_launcher.residual.data.transitions import build_stepchunk_transition
from serl_launcher.residual.observation import build_residual_step_obs_from_core

if TYPE_CHECKING:
    from serl_launcher.data.normalizer import StateActionNormalizer
    from serl_launcher.data.replay_buffer import ReplayBuffer


def _get_payload_base_chunk_for_start(
    payload: Mapping[str, Any],
    *,
    data_config: ResidualDataConfig,
    chunk_start: int,
) -> np.ndarray:
    base_chunks = np.asarray(
        payload[data_config.schema.action.root][data_config.schema.action.base_chunks_key],
        dtype=np.float32,
    )
    chunk_horizon = int(base_chunks.shape[1])
    chunk_index = int(chunk_start // chunk_horizon)
    if chunk_index >= base_chunks.shape[0]:
        raise IndexError(
            "payload base_chunks are shorter than action sequence: "
            f"chunk_index={chunk_index} available={base_chunks.shape[0]}"
        )
    return np.asarray(base_chunks[chunk_index], dtype=np.float32)


def load_residual_training_buffer(
    dataset_paths: Any,
    *,
    replay_buffer: "ReplayBuffer",
    sample_obs_template: Dict[str, np.ndarray],
    action_dim: int,
    chunk_horizon: int,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    chunk_step_enabled: bool,
    logger: logging.Logger,
    data_config: ResidualDataConfig,
    normalizer: Optional["StateActionNormalizer"] = None,
    profiler: Optional[Any] = None,
    max_episodes: Optional[int] = None,
    max_transitions: Optional[int] = None,
    expected_task_key: Optional[str] = None,
    expected_alpha: Optional[float] = None,
    expected_projection: Optional[Mapping[str, Any]] = None,
    dataset_label: str = "residual training",
) -> Dict[str, Any]:
    del profiler
    del sample_obs_template
    stats: Dict[str, Any] = {
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
        "clipped_values": 0,
    }

    if int(stack_horizon) != 1:
        raise ValueError(
            f"Only stack_horizon=1 is currently supported, got {int(stack_horizon)}"
        )

    dataset_files = resolve_episode_files(dataset_paths, base_dir=Path.cwd())
    stats["files_total"] = len(dataset_files)
    if not dataset_files:
        logger.warning("%s dataset_paths resolved to zero episode PKL files", dataset_label)
        return stats

    recent_successes = deque(maxlen=20)
    logger.info(
        "%s dataset_paths resolved: %d episode PKL files found",
        dataset_label,
        len(dataset_files),
    )

    for path in dataset_files:
        if max_episodes is not None and stats["episodes_loaded"] >= int(max_episodes):
            break
        if max_transitions is not None and stats["inserted"] >= int(max_transitions):
            break
        if not path.exists():
            stats["files_missing"] += 1
            logger.warning("%s dataset not found: %s", dataset_label, path)
            continue

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["skipped"] += 1
            logger.warning("failed to load %s dataset %s: %s", dataset_label, path, exc)
            continue

        try:
            if not isinstance(payload, dict):
                raise ValueError("payload must be a dictionary")
            validate_residual_training_payload(
                payload,
                schema=data_config.schema,
                expected_task_key=expected_task_key,
                expected_action_dim=int(action_dim),
                expected_chunk_horizon=int(chunk_horizon),
                expected_alpha=expected_alpha,
                expected_projection=expected_projection,
            )

            actions = np.asarray(
                payload[data_config.schema.action.root].get(
                    data_config.schema.action.final_key,
                    [],
                ),
                dtype=np.float32,
            )
            rewards = np.asarray(
                payload[data_config.schema.trajectory.root].get(
                    data_config.schema.trajectory.rewards_key,
                    np.zeros((actions.shape[0],), dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1)
            dones = np.asarray(
                payload[data_config.schema.trajectory.root].get(
                    data_config.schema.trajectory.dones_key,
                    np.zeros((actions.shape[0],), dtype=bool),
                ),
                dtype=bool,
            ).reshape(-1)
            if actions.ndim != 2 or actions.shape[0] == 0:
                raise ValueError(f"invalid action array in payload: {actions.shape}")

            episode_meta = payload[data_config.schema.episode.root]
            alpha = float(
                payload[data_config.schema.action.root].get(
                    data_config.schema.action.alpha_key,
                    0.0,
                )
            )
            episode_id = int(
                episode_meta.get(data_config.schema.episode.episode_index_key, 0)
            )
            episode_success = int(
                bool(
                    episode_meta.get(
                        data_config.schema.episode.episode_success_key,
                        False,
                    )
                )
            )
            episode_return = float(
                episode_meta.get(
                    data_config.schema.episode.episode_return_key,
                    float(np.sum(rewards)),
                )
            )
            episode_steps = int(
                episode_meta.get(
                    data_config.schema.episode.episode_steps_key,
                    int(actions.shape[0]),
                )
            )
            projection_meta = dict(payload.get(data_config.schema.metadata_key, {}).get("projection", {}))
            stats["clipped_values"] += int(projection_meta.get("clipped_values", 0))

            for step_idx in range(actions.shape[0]):
                if max_transitions is not None and stats["inserted"] >= int(max_transitions):
                    break

                stats["candidates"] += 1
                chunk_start = int((step_idx // chunk_horizon) * chunk_horizon)
                step_in_chunk = int(step_idx - chunk_start)
                done = bool(dones[step_idx]) or bool(step_idx >= (actions.shape[0] - 1))
                reward = float(rewards[step_idx]) if step_idx < rewards.shape[0] else 0.0

                core = build_step_core_from_payload(
                    payload,
                    schema=data_config.schema,
                    frame_idx=step_idx,
                    image_keys=image_keys,
                    image_views=data_config.image_views,
                    normalizer=normalizer,
                )
                base_chunk = _get_payload_base_chunk_for_start(
                    payload,
                    data_config=data_config,
                    chunk_start=chunk_start,
                )
                base_action = np.asarray(base_chunk[step_in_chunk], dtype=np.float32)
                final_action = np.asarray(actions[step_idx], dtype=np.float32)

                if chunk_step_enabled:
                    replay_buffer.insert(
                        build_stepchunk_transition(
                            obs_core=core,
                            base_action=base_action,
                            actions=final_action.reshape(action_dim),
                            reward=reward,
                            done=done,
                            alpha=float(alpha),
                            episode_id=int(episode_id),
                            episode_step=int(step_idx),
                            normalizer=normalizer,
                        )
                    )
                    stats["inserted"] += 1
                    continue

                obs_input = build_residual_step_obs_from_core(
                    core,
                    base_action=base_action,
                    alpha=float(alpha),
                    normalizer=normalizer,
                    stack_horizon=stack_horizon,
                )

                if done:
                    next_obs_input = {
                        key: np.zeros_like(value) for key, value in obs_input.items()
                    }
                    mask = 0.0
                else:
                    next_step_idx = int(step_idx + 1)
                    next_chunk_start = int((next_step_idx // chunk_horizon) * chunk_horizon)
                    next_step_in_chunk = int(next_step_idx - next_chunk_start)
                    next_base_chunk = _get_payload_base_chunk_for_start(
                        payload,
                        data_config=data_config,
                        chunk_start=next_chunk_start,
                    )
                    next_core = build_step_core_from_payload(
                        payload,
                        schema=data_config.schema,
                        frame_idx=next_step_idx,
                        image_keys=image_keys,
                        image_views=data_config.image_views,
                        normalizer=normalizer,
                    )
                    next_obs_input = build_residual_step_obs_from_core(
                        next_core,
                        base_action=np.asarray(
                            next_base_chunk[next_step_in_chunk], dtype=np.float32
                        ),
                        alpha=float(alpha),
                        normalizer=normalizer,
                        stack_horizon=stack_horizon,
                    )
                    mask = 1.0

                replay_buffer.insert(
                    build_step_transition(
                        observations=obs_input,
                        actions=final_action.reshape(action_dim),
                        next_observations=next_obs_input,
                        reward=reward,
                        done=done,
                        mask=mask,
                    )
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
            logger.warning("%s conversion failed file=%s: %s", dataset_label, path, exc)
            continue

    stats["recent_episode_successes"] = list(recent_successes)
    return stats
