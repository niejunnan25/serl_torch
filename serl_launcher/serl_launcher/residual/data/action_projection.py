"""Residual action projection helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def project_expert_action(
    *,
    expert_action: np.ndarray,
    base_action: np.ndarray,
    control_indices: np.ndarray,
    denom: np.ndarray,
    clip_residual_to_unit: bool,
) -> Tuple[np.ndarray, int]:
    base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    expert_action_arr = np.asarray(expert_action, dtype=np.float32).reshape(-1)
    control_indices_arr = np.asarray(control_indices, dtype=np.int64).reshape(-1)
    denom_arr = np.asarray(denom, dtype=np.float32).reshape(-1)

    raw_residual = (
        expert_action_arr[control_indices_arr] - base_action_arr[control_indices_arr]
    ) / denom_arr
    clipped_count = int(np.count_nonzero((raw_residual < -1.0) | (raw_residual > 1.0)))
    if clip_residual_to_unit:
        raw_residual = np.clip(raw_residual, -1.0, 1.0)

    projected = np.asarray(base_action_arr, dtype=np.float32).copy()
    projected[control_indices_arr] = base_action_arr[control_indices_arr] + (
        raw_residual * denom_arr
    )
    return projected.astype(np.float32), clipped_count
