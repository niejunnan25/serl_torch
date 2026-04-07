"""Unified residual-training payload helpers."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

LIBERO_RESIDUAL_TRAINING_FORMAT = "libero_residual_training"
LIBERO_RESIDUAL_TRAINING_MANIFEST_FORMAT = "libero_residual_training_manifest_v1"

IMAGE_SLOT_KEYS = ("image_rgb_0", "image_rgb_1", "image_rgb_2")
IMAGE_SLOT_TO_OBS_KEY = {
    "image_rgb_0": "image",
    "image_rgb_1": "wrist_image",
    "image_rgb_2": "image_rgb_2",
}
OBS_KEY_TO_IMAGE_SLOT = {
    "image": "image_rgb_0",
    "image_rgb_0": "image_rgb_0",
    "wrist_image": "image_rgb_1",
    "image_rgb_1": "image_rgb_1",
    "image_rgb_2": "image_rgb_2",
}


def _normalize_image_mask(
    image_mask: Optional[Mapping[str, Any]],
) -> Dict[str, bool]:
    mask = {key: False for key in IMAGE_SLOT_KEYS}
    if image_mask is None:
        return mask
    for key in IMAGE_SLOT_KEYS:
        if key in image_mask:
            mask[key] = bool(image_mask[key])
    return mask


def _ensure_image_slot_array(
    arr: np.ndarray,
    *,
    slot_name: str,
    num_steps: int,
) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.uint8)
    if arr_np.ndim != 4:
        raise ValueError(f"{slot_name} must have rank 4 [T,H,W,C], got {arr_np.shape}")
    if int(arr_np.shape[0]) != int(num_steps):
        raise ValueError(
            f"{slot_name} length does not match actions: "
            f"{int(arr_np.shape[0])} vs {int(num_steps)}"
        )
    return arr_np


def build_libero_residual_training_payload(
    *,
    source: str,
    suite_name: str,
    task_id: int,
    task_key: str,
    task_description: str,
    prompt: str,
    chunk_horizon: int,
    action_dim: int,
    alpha: float,
    state: np.ndarray,
    image_rgb_0: np.ndarray,
    image_rgb_1: np.ndarray,
    image_rgb_2: Optional[np.ndarray],
    image_mask: Optional[Mapping[str, Any]],
    base_chunks: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    episode_index: int,
    episode_steps: Optional[int] = None,
    episode_return: Optional[float] = None,
    episode_success: Optional[bool] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    expert_actions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    actions_arr = np.asarray(actions, dtype=np.float32)
    if actions_arr.ndim != 2 or actions_arr.shape[0] <= 0:
        raise ValueError(f"actions must have shape [T, action_dim], got {actions_arr.shape}")
    num_steps = int(actions_arr.shape[0])
    if int(actions_arr.shape[1]) != int(action_dim):
        raise ValueError(
            f"actions second dim {int(actions_arr.shape[1])} != action_dim {int(action_dim)}"
        )

    state_arr = np.asarray(state, dtype=np.float32)
    if state_arr.ndim != 2 or int(state_arr.shape[0]) != num_steps:
        raise ValueError(
            f"state must have shape [T, state_dim] with T={num_steps}, got {state_arr.shape}"
        )

    image_0_arr = _ensure_image_slot_array(
        image_rgb_0, slot_name="image_rgb_0", num_steps=num_steps
    )
    image_1_arr = _ensure_image_slot_array(
        image_rgb_1, slot_name="image_rgb_1", num_steps=num_steps
    )
    if image_rgb_2 is None:
        image_2_arr = np.zeros_like(image_0_arr)
    else:
        image_2_arr = _ensure_image_slot_array(
            image_rgb_2, slot_name="image_rgb_2", num_steps=num_steps
        )

    rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if rewards_arr.shape[0] != num_steps:
        raise ValueError(
            f"rewards length {int(rewards_arr.shape[0])} does not match actions {num_steps}"
        )
    dones_arr = np.asarray(dones, dtype=bool).reshape(-1)
    if dones_arr.shape[0] != num_steps:
        raise ValueError(
            f"dones length {int(dones_arr.shape[0])} does not match actions {num_steps}"
        )
    if num_steps > 0:
        dones_arr[-1] = True

    base_chunks_arr = np.asarray(base_chunks, dtype=np.float32)
    if base_chunks_arr.ndim != 3:
        raise ValueError(
            f"base_chunks must have rank 3 [N, H, D], got {base_chunks_arr.shape}"
        )
    if int(base_chunks_arr.shape[1]) != int(chunk_horizon):
        raise ValueError(
            f"base_chunks horizon {int(base_chunks_arr.shape[1])} != chunk_horizon {int(chunk_horizon)}"
        )
    if int(base_chunks_arr.shape[2]) != int(action_dim):
        raise ValueError(
            f"base_chunks action_dim {int(base_chunks_arr.shape[2])} != action_dim {int(action_dim)}"
        )

    normalized_mask = _normalize_image_mask(image_mask)
    normalized_mask["image_rgb_0"] = True
    normalized_mask["image_rgb_1"] = True

    payload: Dict[str, Any] = {
        "format": LIBERO_RESIDUAL_TRAINING_FORMAT,
        "source": str(source),
        "suite_name": str(suite_name),
        "task_id": int(task_id),
        "task_key": str(task_key),
        "task_description": str(task_description),
        "prompt": str(prompt),
        "chunk_horizon": int(chunk_horizon),
        "action_dim": int(action_dim),
        "alpha": float(alpha),
        "state": state_arr.astype(np.float32),
        "image_rgb_0": image_0_arr,
        "image_rgb_1": image_1_arr,
        "image_rgb_2": image_2_arr,
        "image_mask": normalized_mask,
        "base_chunks": base_chunks_arr.astype(np.float32),
        "actions": actions_arr.astype(np.float32),
        "rewards": rewards_arr.astype(np.float32),
        "dones": dones_arr.astype(bool),
        "episode_index": int(episode_index),
        "episode_steps": int(num_steps if episode_steps is None else episode_steps),
        "episode_return": float(
            float(np.sum(rewards_arr)) if episode_return is None else episode_return
        ),
        "episode_success": bool(
            bool(np.any(rewards_arr > 0.0)) if episode_success is None else episode_success
        ),
        "metadata": dict(metadata or {}),
    }
    if expert_actions is not None:
        payload["expert_actions"] = np.asarray(expert_actions, dtype=np.float32)
    return payload


def build_residual_training_manifest(
    *,
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
        "format": LIBERO_RESIDUAL_TRAINING_MANIFEST_FORMAT,
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
        "metadata": dict(metadata or {}),
    }


def _resolve_slot_for_obs_key(obs_key: str) -> str:
    slot_key = OBS_KEY_TO_IMAGE_SLOT.get(str(obs_key), None)
    if slot_key is None:
        raise KeyError(
            f"Unsupported image key {obs_key!r}. Supported keys: {sorted(OBS_KEY_TO_IMAGE_SLOT)}"
        )
    return slot_key


def build_residual_step_core_from_training_payload(
    payload: Mapping[str, Any],
    frame_idx: int,
    *,
    image_keys: Sequence[str],
    normalizer: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    state = np.asarray(payload["state"][frame_idx], dtype=np.float32).reshape(-1)
    if normalizer is not None:
        state = np.asarray(normalizer.normalize_state(state), dtype=np.float32)

    core: Dict[str, np.ndarray] = {"state_core": state.astype(np.float32)}
    for image_key in image_keys:
        slot_key = _resolve_slot_for_obs_key(str(image_key))
        slot_value = np.asarray(payload[slot_key][frame_idx], dtype=np.uint8)
        obs_key = IMAGE_SLOT_TO_OBS_KEY.get(slot_key, str(image_key))
        core[obs_key] = slot_value.copy()
    return core


def validate_residual_training_payload(
    payload: Mapping[str, Any],
    *,
    expected_task_key: Optional[str] = None,
    expected_action_dim: Optional[int] = None,
    expected_chunk_horizon: Optional[int] = None,
    expected_alpha: Optional[float] = None,
    expected_projection: Optional[Mapping[str, Any]] = None,
    alpha_atol: float = 1e-6,
) -> None:
    if payload.get("format") != LIBERO_RESIDUAL_TRAINING_FORMAT:
        raise ValueError(
            f"unsupported residual training payload format: {payload.get('format')!r}"
        )
    if expected_task_key is not None:
        task_key = str(payload.get("task_key", "")).strip()
        if task_key and task_key != str(expected_task_key):
            raise ValueError(
                "payload task key does not match training config: "
                f"payload={task_key!r} expected={str(expected_task_key)!r}"
            )
    if expected_action_dim is not None:
        payload_action_dim = int(payload.get("action_dim", -1))
        if payload_action_dim != int(expected_action_dim):
            raise ValueError(
                "payload action_dim does not match training config: "
                f"payload={payload_action_dim} expected={int(expected_action_dim)}"
            )
    if expected_chunk_horizon is not None:
        payload_chunk_horizon = int(payload.get("chunk_horizon", -1))
        if payload_chunk_horizon != int(expected_chunk_horizon):
            raise ValueError(
                "payload chunk_horizon does not match training config: "
                f"payload={payload_chunk_horizon} expected={int(expected_chunk_horizon)}"
            )
    if expected_alpha is not None:
        payload_alpha = float(payload.get("alpha", 0.0))
        if not np.isclose(payload_alpha, float(expected_alpha), atol=float(alpha_atol)):
            raise ValueError(
                "payload alpha does not match training config: "
                f"payload={payload_alpha} expected={float(expected_alpha)}"
            )
    if expected_projection:
        payload_projection = dict(payload.get("metadata", {}).get("projection", {}))
        for key, expected_value in expected_projection.items():
            if key not in payload_projection:
                raise ValueError(
                    f"payload projection metadata missing required key {key!r}"
                )
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
            else:
                if payload_value != expected_value:
                    raise ValueError(
                        f"payload projection metadata mismatch for {key!r}: "
                        f"{payload_value!r} != {expected_value!r}"
                    )
