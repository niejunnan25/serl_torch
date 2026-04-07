"""Residual action helpers shared across environments."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _validate_residual_alpha(alpha: float) -> float:
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value):
        raise ValueError(f"Residual alpha must be finite, got {alpha!r}")
    if alpha_value < 0.0:
        raise ValueError(f"Residual alpha must be >= 0, got {alpha_value}")
    return alpha_value


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


def as_numpy_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.shape[0] != int(action_dim):
        raise ValueError(
            f"Residual action dim mismatch: got {action_arr.shape[0]} expected {int(action_dim)}"
        )
    return action_arr


def as_numpy_action_chunk(
    action: np.ndarray, *, action_dim: int, chunk_horizon: int
) -> np.ndarray:
    flat = as_numpy_action(action, int(action_dim) * int(chunk_horizon))
    return flat.reshape(int(chunk_horizon), int(action_dim))


def compose_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    alpha: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    base_action_arr = np.asarray(base_action, dtype=np.float32)
    residual_action_arr = np.asarray(residual_action, dtype=np.float32)

    clipped = np.clip(residual_action_arr, -1.0, 1.0)
    residual_scale = _validate_residual_alpha(alpha)
    bounded = np.clip(clipped * residual_scale, -residual_scale, residual_scale)
    applied_delta = bounded * np.asarray(limits, dtype=np.float32)

    delta_full = np.zeros_like(base_action_arr, dtype=np.float32)
    delta_full[np.asarray(indices, dtype=np.int64)] = applied_delta

    final_action = base_action_arr + delta_full
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
