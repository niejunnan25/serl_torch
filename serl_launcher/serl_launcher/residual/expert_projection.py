from __future__ import annotations

"""Residual-space projection helpers for offline expert actions."""

import numpy as np

from .typed_action import ResidualActionSpec

UNIT_RESIDUAL_EPS = 1.0e-6


def project_expert_action(
    *,
    expert_action: np.ndarray,
    base_action: np.ndarray,
    action_spec: ResidualActionSpec,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
    unit_residual_eps: float = UNIT_RESIDUAL_EPS,
) -> tuple[np.ndarray, int, bool]:
    expert_arr = np.asarray(expert_action, dtype=np.float32).reshape(-1)
    base_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if expert_arr.shape != base_arr.shape:
        raise ValueError(
            f"expert/base action shape mismatch: {expert_arr.shape} != {base_arr.shape}"
        )

    projected = np.asarray(base_arr, dtype=np.float32).copy()
    if action_spec.alpha <= 0.0:
        step_unrepresentable = bool(
            np.any(
                np.abs(
                    expert_arr[
                        np.asarray(action_spec.control_indices, dtype=np.int64).reshape(-1)
                    ]
                    - base_arr[
                        np.asarray(action_spec.control_indices, dtype=np.int64).reshape(-1)
                    ]
                )
                > float(unit_residual_eps)
            )
        )
        if action_spec.clip_gripper and projected.shape[0] > 0:
            projected[-1] = np.clip(projected[-1], -1.0, 1.0)
        return projected, 0, step_unrepresentable

    clipped_values = 0
    step_unrepresentable = False
    limits = np.asarray(action_spec.residual_limits, dtype=np.float32).reshape(-1)
    control_indices = np.asarray(action_spec.control_indices, dtype=np.int64).reshape(-1)
    denom = limits * float(action_spec.alpha) * float(expert_reference_scale)
    for local_idx, action_idx in enumerate(control_indices):
        scale = float(denom[local_idx])
        if (not np.isfinite(scale)) or scale <= 0.0:
            if abs(float(expert_arr[action_idx] - base_arr[action_idx])) > float(
                unit_residual_eps
            ):
                clipped_values += 1
                step_unrepresentable = True
            continue
        residual_value = float(expert_arr[action_idx] - base_arr[action_idx]) / scale
        if abs(residual_value) > (1.0 + float(unit_residual_eps)):
            clipped_values += 1
            step_unrepresentable = True
        if clip_residual_to_unit:
            residual_value = float(np.clip(residual_value, -1.0, 1.0))
        projected[action_idx] = base_arr[action_idx] + (residual_value * scale)

    if action_spec.clip_gripper and projected.shape[0] > 0:
        projected[-1] = np.clip(projected[-1], -1.0, 1.0)
    return projected.astype(np.float32, copy=False), int(clipped_values), bool(
        step_unrepresentable
    )


__all__ = [
    "UNIT_RESIDUAL_EPS",
    "project_expert_action",
]
