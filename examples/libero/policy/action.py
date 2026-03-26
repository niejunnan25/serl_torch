"""Residual action helpers for LIBERO actions."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..utils.alpha_utils import validate_alpha


def select_action_chunk_window(
    action_chunk: np.ndarray,
    horizon: int,
    *,
    action_dim: Optional[int] = None,
) -> np.ndarray:
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2:
        raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")
    expected_action_dim = (
        int(action_dim) if action_dim is not None else int(chunk.shape[1])
    )
    if expected_action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {expected_action_dim}")
    if chunk.shape[1] != expected_action_dim:
        raise ValueError(
            f"Unexpected action chunk shape: {chunk.shape}, expected second dim {expected_action_dim}"
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if chunk.shape[0] == 0:
        raise ValueError("OpenPI returned empty action chunk")
    if chunk.shape[0] >= horizon:
        return chunk[:horizon]
    pad_count = horizon - chunk.shape[0]
    tail = np.repeat(chunk[-1:, :], pad_count, axis=0)
    return np.concatenate([chunk, tail], axis=0)


def controlled_action_indices(
    control_gripper: bool, *, full_action_dim: int
) -> np.ndarray:
    full_dim = int(full_action_dim)
    if full_dim <= 0:
        raise ValueError(f"full_action_dim must be positive, got {full_dim}")
    if control_gripper:
        return np.arange(full_dim, dtype=np.int64)
    if full_dim == 1:
        raise ValueError("Cannot disable gripper control when full_action_dim=1")
    return np.arange(full_dim - 1, dtype=np.int64)


def default_control_indices_for_dim(
    action_dim: int, *, full_action_dim: int
) -> np.ndarray:
    dim = int(action_dim)
    full_dim = int(full_action_dim)
    if full_dim <= 0:
        raise ValueError(f"full_action_dim must be positive, got {full_dim}")
    if dim == full_dim:
        return np.arange(full_dim, dtype=np.int64)
    if full_dim > 1 and dim == (full_dim - 1):
        return np.arange(full_dim - 1, dtype=np.int64)
    raise ValueError(
        "Unsupported residual.action_dim. "
        "Please set residual.action_indices explicitly for custom layouts. "
        f"(residual.action_dim={dim}, env.action_dim={full_dim})"
    )


def _normalize_control_indices(
    indices: List[int], *, full_action_dim: int
) -> np.ndarray:
    full_dim = int(full_action_dim)
    if full_dim <= 0:
        raise ValueError(f"full_action_dim must be positive, got {full_dim}")
    arr = np.asarray([int(v) for v in indices], dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("residual.action_indices cannot be empty")
    if np.any(arr < 0) or np.any(arr >= full_dim):
        raise ValueError(
            f"residual.action_indices must be in [0, {full_dim - 1}], got {arr.tolist()}"
        )
    if np.unique(arr).size != arr.size:
        raise ValueError(f"residual.action_indices has duplicates: {arr.tolist()}")
    return arr


def resolve_control_indices(
    *,
    full_action_dim: int,
    action_dim: Optional[int] = None,
    action_indices: Optional[List[int]] = None,
    control_gripper: Optional[bool] = None,
) -> np.ndarray:
    if action_indices is not None:
        resolved = _normalize_control_indices(
            action_indices, full_action_dim=full_action_dim
        )
        if action_dim is not None and resolved.size != int(action_dim):
            raise ValueError(
                "residual.action_dim does not match residual.action_indices length: "
                f"{int(action_dim)} vs {int(resolved.size)}"
            )
        return resolved

    if action_dim is not None:
        return default_control_indices_for_dim(
            int(action_dim), full_action_dim=full_action_dim
        )

    if control_gripper is None:
        control_gripper = True
    return controlled_action_indices(
        bool(control_gripper), full_action_dim=full_action_dim
    )


def build_residual_limits(
    indices: np.ndarray,
    *,
    full_action_dim: int,
    action_limits: Optional[object] = None,
) -> np.ndarray:
    full_dim = int(full_action_dim)
    if full_dim <= 0:
        raise ValueError(f"full_action_dim must be positive, got {full_dim}")
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("control indices cannot be empty")
    if np.any(idx < 0) or np.any(idx >= full_dim):
        raise ValueError(
            f"control indices exceed env.action_dim={full_dim}: {idx.tolist()}"
        )

    def _validate_limits(vec: np.ndarray, *, source: str) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{source} contains non-finite values: {arr.tolist()}")
        if np.any(arr < 0.0):
            raise ValueError(f"{source} must be >= 0 for all dims, got {arr.tolist()}")
        return arr

    if action_limits is not None:
        cfg_limits = _validate_limits(
            np.asarray(action_limits, dtype=np.float32), source="residual.action_limits"
        )
        if cfg_limits.size == full_dim:
            return cfg_limits[idx]
        if cfg_limits.size == idx.size:
            return cfg_limits.copy()
        raise ValueError(
            "residual.action_limits length mismatch: "
            f"got {int(cfg_limits.size)}, expected env.action_dim={full_dim} "
            f"or residual_action_dim={int(idx.size)}"
        )

    raise ValueError(
        "residual.action_limits must be set. "
        "Expected length env.action_dim or residual_action_dim."
    )


def as_numpy_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != action_dim:
        raise ValueError(
            f"Residual action dim mismatch: got {action.shape[0]} expected {action_dim}"
        )
    return action


def as_numpy_action_chunk(
    action: np.ndarray, *, action_dim: int, chunk_horizon: int
) -> np.ndarray:
    flat = as_numpy_action(action, action_dim * chunk_horizon)
    return flat.reshape(chunk_horizon, action_dim)


def compose_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    alpha: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    base_action = np.asarray(base_action, dtype=np.float32)
    residual_action = np.asarray(residual_action, dtype=np.float32)

    clipped = np.clip(residual_action, -1.0, 1.0)
    residual_scale = validate_alpha(alpha, name="alpha", allow_zero=True)
    bounded = np.clip(clipped * residual_scale, -residual_scale, residual_scale)
    applied_delta = bounded * limits

    delta_full = np.zeros_like(base_action, dtype=np.float32)
    delta_full[indices] = applied_delta

    final_action = base_action + delta_full
    if clip_gripper and final_action.shape[0] > 0:
        final_action[-1] = np.clip(final_action[-1], -1.0, 1.0)

    return delta_full, final_action


def compose_residual_action_chunk(
    *,
    base_chunk: np.ndarray,
    residual_chunk: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    alpha: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    base_chunk_arr = np.asarray(base_chunk, dtype=np.float32)
    residual_chunk_arr = np.asarray(residual_chunk, dtype=np.float32)

    if base_chunk_arr.ndim != 2:
        raise ValueError(
            f"Unexpected base_chunk shape: {base_chunk_arr.shape}, expected 2-D chunk"
        )
    if (
        residual_chunk_arr.ndim != 2
        or residual_chunk_arr.shape[0] != base_chunk_arr.shape[0]
    ):
        raise ValueError(
            "Residual chunk must be 2D and share the same horizon as base_chunk: "
            f"{residual_chunk_arr.shape} vs {base_chunk_arr.shape}"
        )

    delta_chunk = np.zeros_like(base_chunk_arr, dtype=np.float32)
    final_chunk = np.zeros_like(base_chunk_arr, dtype=np.float32)
    for step_idx in range(base_chunk_arr.shape[0]):
        delta_step, final_step = compose_residual_action(
            base_action=base_chunk_arr[step_idx],
            residual_action=residual_chunk_arr[step_idx],
            indices=indices,
            limits=limits,
            alpha=alpha,
            clip_gripper=clip_gripper,
        )
        delta_chunk[step_idx] = delta_step
        final_chunk[step_idx] = final_step
    return delta_chunk, final_chunk
