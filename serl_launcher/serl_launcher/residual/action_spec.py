"""Residual action-spec helpers shared across environments."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def resolve_action_mask(
    *,
    full_action_dim: int,
    action_mask: Optional[Sequence[object]] = None,
) -> np.ndarray:
    full_dim = int(full_action_dim)
    if full_dim <= 0:
        raise ValueError(f"full_action_dim must be positive, got {full_dim}")
    if action_mask is None:
        return np.ones((full_dim,), dtype=bool)

    mask_arr = np.asarray(list(action_mask), dtype=bool).reshape(-1)
    if mask_arr.size != full_dim:
        raise ValueError(
            "residual.action_mask length mismatch: "
            f"got {int(mask_arr.size)}, expected env.action_dim={full_dim}"
        )
    if not np.any(mask_arr):
        raise ValueError("residual.action_mask must enable at least one action dim")
    return mask_arr.astype(bool, copy=False)


def resolve_control_indices(
    *,
    full_action_dim: int,
    action_mask: Optional[Sequence[object]] = None,
) -> np.ndarray:
    mask_arr = resolve_action_mask(
        full_action_dim=int(full_action_dim),
        action_mask=action_mask,
    )
    return np.flatnonzero(mask_arr).astype(np.int64)


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

    if action_limits is None:
        raise ValueError(
            "residual.action_limits must be set. "
            "Expected length env.action_dim."
        )

    limits_arr = np.asarray(action_limits, dtype=np.float32).reshape(-1)
    if limits_arr.size != full_dim:
        raise ValueError(
            "residual.action_limits length mismatch: "
            f"got {int(limits_arr.size)}, expected env.action_dim={full_dim}"
        )
    if np.any(~np.isfinite(limits_arr)):
        raise ValueError(
            f"residual.action_limits contains non-finite values: {limits_arr.tolist()}"
        )
    if np.any(limits_arr < 0.0):
        raise ValueError(
            f"residual.action_limits must be >= 0 for all dims, got {limits_arr.tolist()}"
        )
    return limits_arr[idx].astype(np.float32, copy=False)
