"""Residual action helpers for 7D LIBERO actions."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..utils.constants import LIBERO_ACTION_DIM


def select_action_chunk_window(action_chunk: np.ndarray, horizon: int) -> np.ndarray:
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != LIBERO_ACTION_DIM:
        raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if chunk.shape[0] == 0:
        raise ValueError("OpenPI returned empty action chunk")
    if chunk.shape[0] >= horizon:
        return chunk[:horizon]
    pad_count = horizon - chunk.shape[0]
    tail = np.repeat(chunk[-1:, :], pad_count, axis=0)
    return np.concatenate([chunk, tail], axis=0)


def controlled_action_indices(control_gripper: bool) -> np.ndarray:
    if control_gripper:
        return np.arange(LIBERO_ACTION_DIM, dtype=np.int64)
    return np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)


def default_control_indices_for_dim(action_dim: int) -> np.ndarray:
    dim = int(action_dim)
    if dim == 7:
        return np.arange(LIBERO_ACTION_DIM, dtype=np.int64)
    if dim == 6:
        return np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    raise ValueError(
        "Unsupported residual.action_dim. "
        "Please set residual.action_indices explicitly for custom layouts."
    )


def _normalize_control_indices(indices: List[int]) -> np.ndarray:
    arr = np.asarray([int(v) for v in indices], dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("residual.action_indices cannot be empty")
    if np.any(arr < 0) or np.any(arr >= LIBERO_ACTION_DIM):
        raise ValueError(
            f"residual.action_indices must be in [0, {LIBERO_ACTION_DIM - 1}], got {arr.tolist()}"
        )
    if np.unique(arr).size != arr.size:
        raise ValueError(f"residual.action_indices has duplicates: {arr.tolist()}")
    return arr


def resolve_control_indices(
    *,
    action_dim: Optional[int] = None,
    action_indices: Optional[List[int]] = None,
    control_gripper: Optional[bool] = None,
) -> np.ndarray:
    if action_indices is not None:
        resolved = _normalize_control_indices(action_indices)
        if action_dim is not None and resolved.size != int(action_dim):
            raise ValueError(
                "residual.action_dim does not match residual.action_indices length: "
                f"{int(action_dim)} vs {int(resolved.size)}"
            )
        return resolved

    if action_dim is not None:
        return default_control_indices_for_dim(int(action_dim))

    if control_gripper is None:
        control_gripper = True
    return controlled_action_indices(bool(control_gripper))


def build_residual_limits(indices: np.ndarray, arm_limit: float, gripper_limit: float) -> np.ndarray:
    full_limits = np.asarray([arm_limit] * 6 + [gripper_limit], dtype=np.float32)
    return full_limits[indices]


def as_numpy_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != action_dim:
        raise ValueError(f"Residual action dim mismatch: got {action.shape[0]} expected {action_dim}")
    return action


def compose_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    residual_scale: float,
    xi: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    base_action = np.asarray(base_action, dtype=np.float32)
    residual_action = np.asarray(residual_action, dtype=np.float32)

    clipped = np.clip(residual_action, -1.0, 1.0)
    xi = float(max(0.0, xi))
    bounded = np.clip(clipped * xi, -xi, xi)
    applied_delta = bounded * limits * float(residual_scale)

    delta_full = np.zeros_like(base_action, dtype=np.float32)
    delta_full[indices] = applied_delta

    final_action = base_action + delta_full
    if clip_gripper:
        final_action[6] = np.clip(final_action[6], -1.0, 1.0)

    return delta_full, final_action
