"""Canonical payload materialization and validation for residual training."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from serl_launcher.residual.data.config import ResidualDataConfig
from serl_launcher.residual.data.schema import ResidualTrainingSchema
from serl_launcher.residual.data.transforms import compose, copy_data, get_by_path


def _normalize_image_mask(
    schema: ResidualTrainingSchema,
    image_mask: Optional[Mapping[str, Any]],
) -> Dict[str, bool]:
    normalized = {
        slot.name: False for slot in schema.observation.image_slots
    }
    if image_mask is not None:
        for slot_name in normalized:
            if slot_name in image_mask:
                normalized[slot_name] = bool(image_mask[slot_name])
    for slot in schema.observation.image_slots:
        if slot.required:
            normalized[slot.name] = True
    return normalized


def _ensure_image_slot_array(
    arr: Any,
    *,
    slot_name: str,
    num_steps: int,
) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.uint8)
    if arr_np.ndim != 4:
        raise ValueError(f"{slot_name} must have rank 4 [T,H,W,C], got {arr_np.shape}")
    if int(arr_np.shape[0]) != int(num_steps):
        raise ValueError(
            f"{slot_name} length does not match action length: "
            f"{int(arr_np.shape[0])} vs {int(num_steps)}"
        )
    return arr_np


def _validate_projection_metadata(
    payload_projection: Mapping[str, Any],
    expected_projection: Mapping[str, Any],
) -> None:
    for key, expected_value in expected_projection.items():
        if key not in payload_projection:
            raise ValueError(f"payload projection metadata missing required key {key!r}")
        payload_value = payload_projection[key]
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            if not np.allclose(
                np.asarray(payload_value, dtype=np.float32),
                np.asarray(expected_value, dtype=np.float32),
            ):
                raise ValueError(
                    f"payload projection metadata mismatch for {key!r}: "
                    f"{payload_value!r} != {expected_value!r}"
                )
        elif payload_value != expected_value:
            raise ValueError(
                f"payload projection metadata mismatch for {key!r}: "
                f"{payload_value!r} != {expected_value!r}"
            )


def materialize_with_config(
    raw_data: Mapping[str, Any],
    *,
    data_config: ResidualDataConfig,
) -> Dict[str, Any]:
    data = copy_data(raw_data)
    data = compose(data_config.repack_transforms.inputs)(data)
    data = compose(data_config.data_transforms.inputs)(data)
    return finalize_residual_training_payload(data, schema=data_config.schema)


def finalize_residual_training_payload(
    data: Mapping[str, Any],
    *,
    schema: ResidualTrainingSchema,
) -> Dict[str, Any]:
    final_actions = np.asarray(
        get_by_path(data, schema.action.final_path),
        dtype=np.float32,
    )
    if final_actions.ndim != 2 or final_actions.shape[0] <= 0:
        raise ValueError(
            "action/final must have shape [T, action_dim], "
            f"got {final_actions.shape}"
        )
    num_steps = int(final_actions.shape[0])
    action_dim = int(final_actions.shape[1])

    state = np.asarray(get_by_path(data, schema.observation.state_path), dtype=np.float32)
    if state.ndim != 2 or int(state.shape[0]) != num_steps:
        raise ValueError(
            f"{schema.observation.state_path!r} must have shape [T, state_dim] "
            f"with T={num_steps}, got {state.shape}"
        )

    images = get_by_path(data, schema.observation.image_root_path)
    if not isinstance(images, Mapping):
        raise ValueError(f"{schema.observation.image_root_path!r} must be a mapping")
    image_arrays: Dict[str, Optional[np.ndarray]] = {}
    for slot in schema.observation.image_slots:
        if slot.name not in images:
            if slot.required:
                raise ValueError(f"missing required image slot {slot.name!r}")
            image_arrays[slot.name] = None
            continue
        image_arrays[slot.name] = _ensure_image_slot_array(
            images[slot.name],
            slot_name=slot.name,
            num_steps=num_steps,
        )

    reference_slot = schema.observation.image_slots[0].name
    reference_array = image_arrays[reference_slot]
    if reference_array is None:
        raise ValueError(
            f"required reference image slot {reference_slot!r} is missing"
        )
    for slot in schema.observation.image_slots:
        slot_array = image_arrays[slot.name]
        if slot_array is None:
            image_arrays[slot.name] = np.zeros_like(reference_array)
            continue
        if slot_array.shape[0] != num_steps:
            image_arrays[slot.name] = np.zeros_like(reference_array)

    rewards = np.asarray(
        get_by_path(data, schema.trajectory.rewards_path),
        dtype=np.float32,
    ).reshape(-1)
    if rewards.shape[0] != num_steps:
        raise ValueError(
            f"{schema.trajectory.rewards_path!r} length {int(rewards.shape[0])} "
            f"does not match actions {num_steps}"
        )
    dones = np.asarray(
        get_by_path(data, schema.trajectory.dones_path),
        dtype=bool,
    ).reshape(-1)
    if dones.shape[0] != num_steps:
        raise ValueError(
            f"{schema.trajectory.dones_path!r} length {int(dones.shape[0])} "
            f"does not match actions {num_steps}"
        )
    dones[-1] = True

    base_chunks = np.asarray(
        get_by_path(data, schema.action.base_chunks_path),
        dtype=np.float32,
    )
    if base_chunks.ndim != 3:
        raise ValueError(
            f"{schema.action.base_chunks_path!r} must have rank 3 [N,H,D], "
            f"got {base_chunks.shape}"
        )
    chunk_horizon = int(base_chunks.shape[1])
    if int(base_chunks.shape[2]) != action_dim:
        raise ValueError(
            f"{schema.action.base_chunks_path!r} action_dim "
            f"{int(base_chunks.shape[2])} != action_dim {action_dim}"
        )

    alpha = float(get_by_path(data, schema.action.alpha_path))
    metadata = dict(data.get(schema.metadata_key, {}))
    image_mask = _normalize_image_mask(
        schema,
        get_by_path(data, schema.observation.image_mask_path),
    )

    payload: Dict[str, Any] = {
        schema.format_key: schema.episode_format,
        schema.prompt_key: str(get_by_path(data, schema.prompt_path)),
        schema.observation.root: {
            schema.observation.image_key: {
                slot.name: image_arrays[slot.name].astype(np.uint8)
                for slot in schema.observation.image_slots
            },
            schema.observation.image_mask_key: image_mask,
            schema.observation.state_key: state.astype(np.float32),
        },
        schema.action.root: {
            schema.action.base_chunks_key: base_chunks.astype(np.float32),
            schema.action.final_key: final_actions.astype(np.float32),
            schema.action.alpha_key: float(alpha),
        },
        schema.trajectory.root: {
            schema.trajectory.rewards_key: rewards.astype(np.float32),
            schema.trajectory.dones_key: dones.astype(bool),
        },
        schema.episode.root: {
            schema.episode.source_key: str(get_by_path(data, schema.episode.source_path)),
            schema.episode.suite_name_key: str(
                get_by_path(data, schema.episode.suite_name_path)
            ),
            schema.episode.task_id_key: int(
                get_by_path(data, schema.episode.task_id_path)
            ),
            schema.episode.task_key_key: str(
                get_by_path(data, schema.episode.task_key_path)
            ),
            schema.episode.task_description_key: str(
                get_by_path(data, schema.episode.task_description_path)
            ),
            schema.episode.episode_index_key: int(
                get_by_path(data, schema.episode.episode_index_path)
            ),
            schema.episode.episode_steps_key: int(
                data.get(schema.episode.root, {}).get(
                    schema.episode.episode_steps_key,
                    num_steps,
                )
            ),
            schema.episode.episode_return_key: float(
                data.get(schema.episode.root, {}).get(
                    schema.episode.episode_return_key,
                    float(np.sum(rewards)),
                )
            ),
            schema.episode.episode_success_key: bool(
                data.get(schema.episode.root, {}).get(
                    schema.episode.episode_success_key,
                    bool(np.any(rewards > 0.0)),
                )
            ),
        },
        schema.metadata_key: metadata,
    }
    if schema.action.expert_key in data.get(schema.action.root, {}):
        expert = np.asarray(
            get_by_path(data, schema.action.expert_path),
            dtype=np.float32,
        )
        payload[schema.action.root][schema.action.expert_key] = expert
    return payload


def build_residual_training_manifest(
    *,
    schema: ResidualTrainingSchema,
    source: str,
    task_key: str,
    suite_name: str,
    task_id: int,
    task_description: str,
    chunk_horizon: int,
    action_dim: int,
    num_episodes: int,
    total_frames: int,
    episode_files: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        schema.format_key: schema.manifest_format,
        "source": str(source),
        "task_key": str(task_key),
        "suite_name": str(suite_name),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "chunk_horizon": int(chunk_horizon),
        "action_dim": int(action_dim),
        "num_episodes": int(num_episodes),
        "total_frames": int(total_frames),
        "episode_files": [str(path) for path in episode_files],
        schema.metadata_key: dict(metadata or {}),
    }


def validate_residual_training_payload(
    payload: Mapping[str, Any],
    *,
    schema: ResidualTrainingSchema,
    expected_task_key: Optional[str] = None,
    expected_action_dim: Optional[int] = None,
    expected_chunk_horizon: Optional[int] = None,
    expected_alpha: Optional[float] = None,
    expected_projection: Optional[Mapping[str, Any]] = None,
    alpha_atol: float = 1e-6,
) -> None:
    if payload.get(schema.format_key) != schema.episode_format:
        raise ValueError(
            f"unsupported residual training payload format: "
            f"{payload.get(schema.format_key)!r}"
        )

    episode = payload.get(schema.episode.root, {})
    action = payload.get(schema.action.root, {})
    observation = payload.get(schema.observation.root, {})
    trajectory = payload.get(schema.trajectory.root, {})
    if not isinstance(episode, Mapping):
        raise ValueError(f"{schema.episode.root!r} must be a mapping")
    if not isinstance(action, Mapping):
        raise ValueError(f"{schema.action.root!r} must be a mapping")
    if not isinstance(observation, Mapping):
        raise ValueError(f"{schema.observation.root!r} must be a mapping")
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"{schema.trajectory.root!r} must be a mapping")

    task_key = str(episode.get(schema.episode.task_key_key, "")).strip()
    if expected_task_key is not None and task_key and task_key != str(expected_task_key):
        raise ValueError(
            "payload task key does not match training config: "
            f"payload={task_key!r} expected={str(expected_task_key)!r}"
        )

    final_actions = np.asarray(action.get(schema.action.final_key, []), dtype=np.float32)
    if final_actions.ndim != 2 or final_actions.shape[0] <= 0:
        raise ValueError(f"invalid action/final array in payload: {final_actions.shape}")
    if expected_action_dim is not None and int(final_actions.shape[1]) != int(expected_action_dim):
        raise ValueError(
            "payload action_dim does not match training config: "
            f"payload={int(final_actions.shape[1])} expected={int(expected_action_dim)}"
        )

    base_chunks = np.asarray(action.get(schema.action.base_chunks_key, []), dtype=np.float32)
    if base_chunks.ndim != 3:
        raise ValueError(
            f"invalid {schema.action.base_chunks_key!r} array in payload: {base_chunks.shape}"
        )
    if expected_chunk_horizon is not None and int(base_chunks.shape[1]) != int(expected_chunk_horizon):
        raise ValueError(
            "payload chunk_horizon does not match training config: "
            f"payload={int(base_chunks.shape[1])} expected={int(expected_chunk_horizon)}"
        )

    payload_alpha = float(action.get(schema.action.alpha_key, 0.0))
    if expected_alpha is not None and not np.isclose(
        payload_alpha,
        float(expected_alpha),
        atol=float(alpha_atol),
    ):
        raise ValueError(
            "payload alpha does not match training config: "
            f"payload={payload_alpha} expected={float(expected_alpha)}"
        )

    state = np.asarray(observation.get(schema.observation.state_key, []), dtype=np.float32)
    if state.ndim != 2 or int(state.shape[0]) != int(final_actions.shape[0]):
        raise ValueError(
            f"{schema.observation.state_path!r} must have shape [T, state_dim] "
            f"with T={int(final_actions.shape[0])}, got {state.shape}"
        )
    images = observation.get(schema.observation.image_key, {})
    if not isinstance(images, Mapping):
        raise ValueError(f"{schema.observation.image_root_path!r} must be a mapping")
    for slot in schema.observation.image_slots:
        if slot.required and slot.name not in images:
            raise ValueError(f"missing required image slot {slot.name!r}")

    rewards = np.asarray(trajectory.get(schema.trajectory.rewards_key, []), dtype=np.float32)
    dones = np.asarray(trajectory.get(schema.trajectory.dones_key, []), dtype=bool)
    if rewards.shape[0] != final_actions.shape[0]:
        raise ValueError(
            f"{schema.trajectory.rewards_path!r} length {rewards.shape[0]} "
            f"does not match actions {final_actions.shape[0]}"
        )
    if dones.shape[0] != final_actions.shape[0]:
        raise ValueError(
            f"{schema.trajectory.dones_path!r} length {dones.shape[0]} "
            f"does not match actions {final_actions.shape[0]}"
        )

    if expected_projection:
        projection = dict(payload.get(schema.metadata_key, {}).get("projection", {}))
        _validate_projection_metadata(projection, expected_projection)


def build_step_core_from_payload(
    payload: Mapping[str, Any],
    *,
    schema: ResidualTrainingSchema,
    frame_idx: int,
    image_keys: Sequence[str],
    image_views: Optional[Mapping[str, str]] = None,
) -> Dict[str, np.ndarray]:
    state = np.asarray(
        get_by_path(payload, schema.observation.state_path)[frame_idx],
        dtype=np.float32,
    ).reshape(-1)

    images = get_by_path(payload, schema.observation.image_root_path)
    image_mask = get_by_path(payload, schema.observation.image_mask_path)
    if not isinstance(images, Mapping) or not isinstance(image_mask, Mapping):
        raise ValueError("payload observation image/image_mask entries must be mappings")

    core: Dict[str, np.ndarray] = {"state_core": state.astype(np.float32)}
    view_map = dict(image_views or {})
    for image_key in image_keys:
        slot_key = view_map.get(str(image_key), str(image_key))
        if slot_key not in images:
            raise KeyError(
                f"Unsupported image key {image_key!r}; resolved slot {slot_key!r} "
                f"is not present in payload"
            )
        if not bool(image_mask.get(slot_key, False)):
            raise KeyError(
                f"Image slot {slot_key!r} is masked out in payload image_mask"
            )
        core[str(image_key)] = np.asarray(images[slot_key][frame_idx], dtype=np.uint8).copy()
    return core
