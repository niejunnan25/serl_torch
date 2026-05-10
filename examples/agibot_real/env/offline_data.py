from __future__ import annotations

"""Temporary AgiBot offline-data helpers aligned to the LIBERO train flow.

This module intentionally mirrors the shape of ``examples/libero/env/offline_data.py``
for the first alignment pass. The prepared replay contract is stable, while the
raw input format remains a temporary reference-only pickle layout until a real
AgiBot data source is wired in.
"""

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from serl_launcher.data.offline_prepared import EPISODE_FILE_GLOB
from serl_launcher.data.offline_prepared import MANIFEST_FILENAME
from serl_launcher.data.offline_prepared import OfflinePreparedResolution
from serl_launcher.data.offline_prepared import build_residual_prepared_fingerprint
from serl_launcher.data.offline_prepared import build_residual_training_signature
from serl_launcher.data.offline_prepared import extract_residual_manifest_signature
from serl_launcher.data.offline_prepared import (
    load_prepared_offline_replay as _load_prepared_offline_replay,
)
from serl_launcher.data.offline_prepared import read_manifest
from serl_launcher.data.offline_prepared import (
    resolve_prepared_episode_files as _resolve_prepared_episode_files,
)
from serl_launcher.data.offline_prepared import resolve_prepared_path_value
from serl_launcher.data.offline_prepared import resolve_residual_prepared_dir
from serl_launcher.data.offline_prepared import validate_prepared_paths
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.policy.typed_factory import resolve_policy_backend_id
from serl_launcher.policy.typed_factory import resolve_policy_backend_type
from serl_launcher.residual.expert_projection import project_expert_action
from serl_launcher.residual.observation import build_chunk_residual_obs
from serl_launcher.residual.observation import prepare_base_actions_chunk
from serl_launcher.residual.typed_action import ResidualActionSpec
from serl_launcher.utils.path_utils import resolve_original_cwd
from serl_launcher.utils.path_utils import resolve_path
from serl_launcher.utils.serialization import to_jsonable

from ..config import AgiBotTrainConfig
from .observation import build_agibot_state
from .observation import extract_agibot_residual_images

OFFLINE_FORMAT_VERSION = "agibot_real_offline_step_transitions_v1"
REFERENCE_SOURCE_FORMAT = "agibot_reference_episode_pickle_v1"
REFERENCE_NOTE = (
    "Temporary reference offline pipeline modeled after examples/libero. "
    "Replace this raw source format once the real AgiBot data source is defined."
)


@dataclasses.dataclass(frozen=True, slots=True)
class AgiBotTaskSpec:
    task_name: str
    task_key: str
    task_description: str
    dataset_path: Path


def resolve_task_spec(cfg: AgiBotTrainConfig) -> AgiBotTaskSpec:
    dataset_override = cfg.offline.prepare.raw_dataset_path
    if dataset_override is None:
        raise ValueError(
            "Temporary AgiBot offline prepare requires offline.prepare.raw_dataset_path"
        )
    dataset_path = resolve_path(dataset_override, base=resolve_original_cwd())
    return AgiBotTaskSpec(
        task_name=str(cfg.task.name),
        task_key=str(cfg.task.task_key),
        task_description=str(cfg.task.prompt),
        dataset_path=dataset_path,
    )


def prepare_fingerprint(
    cfg: AgiBotTrainConfig,
    *,
    task_spec: AgiBotTaskSpec,
) -> dict[str, Any]:
    return build_residual_prepared_fingerprint(
        format_version=OFFLINE_FORMAT_VERSION,
        task_key=str(task_spec.task_key),
        task_description=str(task_spec.task_description),
        policy_backend_type=resolve_policy_backend_type(cfg),
        policy_backend_id=resolve_policy_backend_id(cfg),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        action_dim=int(cfg.env.action_dim),
        alpha=float(cfg.residual.alpha),
        action_mask=cfg.residual.action_mask,
        action_limits=cfg.residual.action_limits,
        clip_gripper=bool(cfg.residual.clip_gripper),
        expert_reference_scale=float(cfg.offline.prepare.expert_reference_scale),
        clip_residual_to_unit=bool(cfg.offline.prepare.clip_residual_to_unit),
        filter_unrepresentable_steps=bool(
            cfg.offline.prepare.filter_unrepresentable_steps
        ),
        image_keys=cfg.obs.image_keys,
        vector_obs_keys=cfg.obs.vector_obs_keys,
        raw_dataset_path=task_spec.dataset_path,
        extra_fields={"raw_source_format": REFERENCE_SOURCE_FORMAT},
    )


def training_compatibility_signature(cfg: AgiBotTrainConfig) -> dict[str, Any]:
    return build_residual_training_signature(
        task_key=str(cfg.task.task_key),
        policy_backend_type=resolve_policy_backend_type(cfg),
        policy_backend_id=resolve_policy_backend_id(cfg),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        action_dim=int(cfg.env.action_dim),
        alpha=float(cfg.residual.alpha),
        action_mask=cfg.residual.action_mask,
        action_limits=cfg.residual.action_limits,
        clip_gripper=bool(cfg.residual.clip_gripper),
        expert_reference_scale=float(cfg.offline.prepare.expert_reference_scale),
        clip_residual_to_unit=bool(cfg.offline.prepare.clip_residual_to_unit),
        filter_unrepresentable_steps=bool(
            cfg.offline.prepare.filter_unrepresentable_steps
        ),
        image_keys=cfg.obs.image_keys,
        vector_obs_keys=cfg.obs.vector_obs_keys,
    )


def prepared_dir_for_cfg(
    cfg: AgiBotTrainConfig,
    *,
    task_spec: AgiBotTaskSpec,
) -> Path:
    return resolve_residual_prepared_dir(
        output_root=cfg.offline.prepare.output_root,
        task_key=str(task_spec.task_key),
        policy_backend=describe_policy_backend(cfg),
        chunk_horizon=int(cfg.residual.chunk_horizon),
        alpha=float(cfg.residual.alpha),
    )


def _manifest_signature(
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return extract_residual_manifest_signature(manifest)


def resolve_configured_prepared_paths(cfg: AgiBotTrainConfig) -> tuple[Path, ...]:
    return resolve_prepared_path_value(cfg.offline.prepared_path)


def resolve_and_validate_prepared_paths(
    cfg: AgiBotTrainConfig,
    *,
    logger: logging.Logger,
) -> OfflinePreparedResolution:
    del logger
    return validate_prepared_paths(
        resolve_configured_prepared_paths(cfg),
        expected_signature=training_compatibility_signature(cfg),
        manifest_signature_fn=_manifest_signature,
        manifest_filename=MANIFEST_FILENAME,
    )


def _coerce_obs_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _coerce_obs_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_coerce_obs_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_coerce_obs_tree(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def normalize_episode_steps(payload: Any, *, source_path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        steps = payload
    elif isinstance(payload, tuple):
        steps = list(payload)
    elif isinstance(payload, dict):
        for key in ("steps", "transitions", "episode"):
            candidate = payload.get(key, None)
            if isinstance(candidate, list):
                steps = candidate
                break
        else:
            raise ValueError(
                f"Unsupported raw offline episode payload keys for {source_path}: "
                f"{sorted(payload.keys())}"
            )
    else:
        raise ValueError(
            f"Raw offline episode must be list/tuple/dict, got {type(payload)} from {source_path}"
        )
    if not steps:
        raise ValueError(f"Raw offline episode has no steps: {source_path}")
    normalized: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(
                f"Raw offline step must be a dict, got {type(step)} in {source_path}"
            )
        normalized.append(dict(step))
    return normalized


def _raw_obs_from_step(step: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    obs = step.get("observations", None)
    if obs is None:
        obs = step.get("obs", None)
    if not isinstance(obs, dict):
        raise ValueError(f"Raw offline step missing observations in {source_path}")
    return _coerce_obs_tree(obs)


def _raw_next_obs_from_step(step: dict[str, Any]) -> dict[str, Any] | None:
    next_obs = step.get("next_observations", None)
    if next_obs is None:
        next_obs = step.get("next_obs", None)
    if next_obs is None:
        return None
    if not isinstance(next_obs, dict):
        raise ValueError("Raw offline next_observations must be a dict when provided")
    return _coerce_obs_tree(next_obs)


def _expert_action_from_step(step: dict[str, Any], *, action_dim: int) -> np.ndarray:
    for key in ("expert_action", "action", "actions"):
        value = step.get(key, None)
        if value is None:
            continue
        action = np.asarray(value, dtype=np.float32).reshape(-1)
        if int(action.shape[0]) != int(action_dim):
            raise ValueError(
                f"Raw offline action must be {int(action_dim)}D, got {action.shape}"
            )
        return action
    raise ValueError("Raw offline step missing expert action")


def _reward_from_step(step: dict[str, Any]) -> float:
    value = step.get("rewards", step.get("reward", 0.0))
    return float(value)


def _done_from_step(step: dict[str, Any], *, is_last: bool) -> bool:
    return bool(step.get("dones", step.get("done", False))) or bool(is_last)


def _truncated_from_step(step: dict[str, Any]) -> bool:
    return bool(step.get("truncated", False))


def _success_from_step(step: dict[str, Any]) -> bool:
    return bool(step.get("success", False))


def _base_chunk_from_step(
    step: dict[str, Any],
    *,
    chunk_horizon: int,
) -> np.ndarray | None:
    for key in ("base_chunk", "base_action_chunk"):
        value = step.get(key, None)
        if value is None:
            continue
        return prepare_base_actions_chunk(
            base_actions=np.asarray(value, dtype=np.float32),
            chunk_horizon=chunk_horizon,
            source=f"raw offline {key}",
        )
    return None


def _precompute_base_chunks_for_steps(
    steps: Sequence[dict[str, Any]],
    *,
    task_prompt: str,
    base_policy: Any,
    chunk_horizon: int,
    action_dim: int,
    source_path: Path,
) -> np.ndarray:
    precomputed_chunks: list[np.ndarray] = []
    missing_indices: list[int] = []

    for step_idx, step in enumerate(steps):
        maybe_base_chunk = _base_chunk_from_step(
            step,
            chunk_horizon=int(chunk_horizon),
        )
        if maybe_base_chunk is None:
            missing_indices.append(int(step_idx))
            precomputed_chunks.append(
                np.zeros((int(chunk_horizon), int(action_dim)), dtype=np.float32)
            )
        else:
            precomputed_chunks.append(np.asarray(maybe_base_chunk, dtype=np.float32))

    if not missing_indices:
        return np.asarray(precomputed_chunks, dtype=np.float32)

    step_iter: Iterable[int] = tqdm(
        missing_indices,
        total=len(missing_indices),
        desc=f"offline base chunks {source_path.name}",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )
    for step_idx in step_iter:
        obs_raw = _raw_obs_from_step(steps[step_idx], source_path=source_path)
        action_chunk, _ = base_policy.infer(obs_raw, prompt=task_prompt)
        precomputed_chunks[step_idx] = prepare_base_actions_chunk(
            base_actions=action_chunk,
            chunk_horizon=chunk_horizon,
            source="offline base policy",
        )
    return np.asarray(precomputed_chunks, dtype=np.float32)


def _zero_like_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in obs.items()}


def prepare_reference_episode_transitions(
    *,
    raw_steps: Sequence[dict[str, Any]],
    episode_id: int,
    task_prompt: str,
    action_spec: ResidualActionSpec,
    image_keys: tuple[str, ...],
    base_policy: Any,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
    filter_unrepresentable_steps: bool,
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    num_steps = int(len(raw_steps))
    if num_steps <= 0:
        raise ValueError(f"Raw offline episode has no steps: {source_path}")

    base_chunks_per_step = _precompute_base_chunks_for_steps(
        raw_steps,
        task_prompt=task_prompt,
        base_policy=base_policy,
        chunk_horizon=int(action_spec.chunk_horizon),
        action_dim=int(action_spec.action_dim),
        source_path=source_path,
    )
    if base_chunks_per_step.shape[:2] != (
        int(num_steps),
        int(action_spec.chunk_horizon),
    ):
        raise ValueError(
            "Unexpected precomputed base chunk shape: "
            f"{base_chunks_per_step.shape} vs ({num_steps}, {int(action_spec.chunk_horizon)}, *)"
        )

    transitions: list[dict[str, Any]] = []
    unrepresentable_values = 0
    steps_unrepresentable = 0
    steps_filtered = 0
    episode_return = 0.0
    episode_success = False

    for step_idx, raw_step in enumerate(raw_steps):
        obs_raw = _raw_obs_from_step(raw_step, source_path=source_path)
        base_chunk = np.asarray(base_chunks_per_step[step_idx], dtype=np.float32)
        residual_obs = build_chunk_residual_obs(
            robot_state=build_agibot_state(obs_raw),
            images=extract_agibot_residual_images(
                obs_raw,
                image_keys=image_keys,
            ),
            image_keys=image_keys,
            base_actions=base_chunk,
            residual_alpha=float(action_spec.alpha),
        )

        final_action, step_unrepresentable_values, step_unrepresentable = project_expert_action(
            expert_action=_expert_action_from_step(
                raw_step,
                action_dim=int(action_spec.action_dim),
            ),
            base_action=base_chunk[0],
            action_spec=action_spec,
            expert_reference_scale=expert_reference_scale,
            clip_residual_to_unit=clip_residual_to_unit,
        )
        unrepresentable_values += int(step_unrepresentable_values)
        if step_unrepresentable:
            steps_unrepresentable += 1
            if filter_unrepresentable_steps:
                steps_filtered += 1
                continue

        is_last = bool(step_idx >= (num_steps - 1))
        reward = _reward_from_step(raw_step)
        done = _done_from_step(raw_step, is_last=is_last)
        truncated = _truncated_from_step(raw_step)
        episode_done = bool(done or truncated)
        episode_return += float(reward)
        episode_success = bool(episode_success or _success_from_step(raw_step))

        if episode_done:
            next_residual_obs = _zero_like_obs(residual_obs)
            mask = 0.0
        else:
            next_obs_raw = _raw_next_obs_from_step(raw_step)
            if next_obs_raw is None:
                next_obs_raw = _raw_obs_from_step(
                    raw_steps[step_idx + 1],
                    source_path=source_path,
                )
            next_base_chunk = np.asarray(base_chunks_per_step[step_idx + 1], dtype=np.float32)
            next_residual_obs = build_chunk_residual_obs(
                robot_state=build_agibot_state(next_obs_raw),
                images=extract_agibot_residual_images(
                    next_obs_raw,
                    image_keys=image_keys,
                ),
                image_keys=image_keys,
                base_actions=next_base_chunk,
                residual_alpha=float(action_spec.alpha),
            )
            mask = 1.0

        transitions.append(
            {
                "episode_id": int(episode_id),
                "episode_step": int(step_idx),
                "observations": residual_obs,
                "actions": np.asarray(final_action, dtype=np.float32).reshape(-1),
                "next_observations": next_residual_obs,
                "rewards": float(reward),
                "masks": float(mask),
                "dones": bool(episode_done),
            }
        )

    episode_stats = {
        "episode_id": int(episode_id),
        "steps_total": int(num_steps),
        "steps_written": int(len(transitions)),
        "steps_unrepresentable": int(steps_unrepresentable),
        "steps_filtered": int(steps_filtered),
        "episode_return": float(episode_return),
        "success": bool(episode_success),
        "unrepresentable_values": int(unrepresentable_values),
    }
    return transitions, episode_stats


def write_manifest(
    *,
    manifest_path: Path,
    task_spec: AgiBotTaskSpec,
    cfg: AgiBotTrainConfig,
    fingerprint: dict[str, Any],
    episode_files: Sequence[Path],
    prepare_stats: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "format_version": OFFLINE_FORMAT_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "reference_only": True,
        "reference_note": REFERENCE_NOTE,
        "task": {
            "task_name": str(task_spec.task_name),
            "task_key": str(task_spec.task_key),
            "task_description": str(task_spec.task_description),
            "dataset_path": str(task_spec.dataset_path),
        },
        "policy": {
            "type": resolve_policy_backend_type(cfg),
            "id": resolve_policy_backend_id(cfg),
            "backend": describe_policy_backend(cfg),
        },
        "residual": {
            "alpha": float(cfg.residual.alpha),
            "chunk_horizon": int(cfg.residual.chunk_horizon),
            "action_mask": (
                None
                if cfg.residual.action_mask is None
                else [bool(v) for v in cfg.residual.action_mask]
            ),
            "action_limits": [float(v) for v in cfg.residual.action_limits],
            "clip_gripper": bool(cfg.residual.clip_gripper),
        },
        "offline": {
            "expert_reference_scale": float(cfg.offline.prepare.expert_reference_scale),
            "clip_residual_to_unit": bool(cfg.offline.prepare.clip_residual_to_unit),
            "filter_unrepresentable_steps": bool(
                cfg.offline.prepare.filter_unrepresentable_steps
            ),
            "raw_source_format": REFERENCE_SOURCE_FORMAT,
        },
        "fingerprint": to_jsonable(fingerprint),
        "prepare_stats": to_jsonable(prepare_stats),
        "episode_files": [str(path.name) for path in episode_files],
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    return manifest


def resolve_prepared_episode_files(paths: Sequence[Path]) -> list[Path]:
    return _resolve_prepared_episode_files(
        paths,
        manifest_filename=MANIFEST_FILENAME,
        episode_file_glob=EPISODE_FILE_GLOB,
        read_manifest_fn=read_manifest,
    )


def resolve_reference_raw_episode_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_dir():
        episode_files = sorted(dataset_path.glob(EPISODE_FILE_GLOB))
        if episode_files:
            return [path.resolve() for path in episode_files]
        raise FileNotFoundError(
            f"Temporary AgiBot raw offline directory has no {EPISODE_FILE_GLOB} files: {dataset_path}"
        )
    if dataset_path.is_file() and dataset_path.suffix == ".pkl":
        return [dataset_path.resolve()]
    raise ValueError(
        "Temporary AgiBot offline prepare expects offline.prepare.raw_dataset_path "
        f"to be a directory of {EPISODE_FILE_GLOB} files or a single .pkl file, got {dataset_path}"
    )


def load_prepared_offline_replay(
    *,
    replay_buffer: Any,
    prepared_paths: Sequence[Path],
    logger: logging.Logger,
    max_episodes: int | None = None,
    max_transitions: int | None = None,
) -> dict[str, Any]:
    return _load_prepared_offline_replay(
        replay_buffer=replay_buffer,
        prepared_paths=prepared_paths,
        logger=logger,
        max_episodes=max_episodes,
        max_transitions=max_transitions,
        manifest_filename=MANIFEST_FILENAME,
        episode_file_glob=EPISODE_FILE_GLOB,
        read_manifest_fn=read_manifest,
    )


__all__ = [
    "AgiBotTaskSpec",
    "OfflinePreparedResolution",
    "OFFLINE_FORMAT_VERSION",
    "REFERENCE_NOTE",
    "REFERENCE_SOURCE_FORMAT",
    "load_prepared_offline_replay",
    "normalize_episode_steps",
    "prepare_fingerprint",
    "prepare_reference_episode_transitions",
    "prepared_dir_for_cfg",
    "resolve_and_validate_prepared_paths",
    "resolve_configured_prepared_paths",
    "resolve_prepared_episode_files",
    "resolve_reference_raw_episode_files",
    "resolve_task_spec",
    "write_manifest",
]
