"""Arm-layout helpers for AgiBot residual training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

AGIBOT_ARM_ACTION_DIM = 7
AGIBOT_ROBOT_ACTION_DIM = 14

ARM_LAYOUT_LEFT = "left_arm"
ARM_LAYOUT_RIGHT = "right_arm"
ARM_LAYOUT_DUAL = "dual_arm"

_LEFT_SLICE = slice(0, AGIBOT_ARM_ACTION_DIM)
_RIGHT_SLICE = slice(AGIBOT_ARM_ACTION_DIM, AGIBOT_ROBOT_ACTION_DIM)
_DUAL_SLICE = slice(0, AGIBOT_ROBOT_ACTION_DIM)


@dataclass(frozen=True, slots=True)
class ArmLayoutSpec:
    name: str
    active_slice: slice
    action_dim: int
    robot_action_dim: int = AGIBOT_ROBOT_ACTION_DIM

    @property
    def is_single_arm(self) -> bool:
        return self.action_dim == AGIBOT_ARM_ACTION_DIM


def normalize_arm_layout(value: Any) -> str:
    layout = str(value or ARM_LAYOUT_DUAL).strip().lower()
    if layout in {"", "full", "dual", "dual_arm", "bimanual"}:
        return ARM_LAYOUT_DUAL
    if layout in {
        "left",
        "left_arm",
        "left_hand",
        "left_arm_camera_position",
        "left_hand_camera_position",
    }:
        return ARM_LAYOUT_LEFT
    if layout in {
        "right",
        "right_arm",
        "right_hand",
        "right_arm_camera_position",
        "right_hand_camera_position",
    }:
        return ARM_LAYOUT_RIGHT
    raise ValueError(
        "arm layout must be one of 'dual_arm', 'left_arm', or 'right_arm', "
        f"got {value!r}"
    )


def get_arm_layout_spec(value: Any) -> ArmLayoutSpec:
    layout = normalize_arm_layout(value)
    if layout == ARM_LAYOUT_LEFT:
        return ArmLayoutSpec(
            name=ARM_LAYOUT_LEFT,
            active_slice=_LEFT_SLICE,
            action_dim=AGIBOT_ARM_ACTION_DIM,
        )
    if layout == ARM_LAYOUT_RIGHT:
        return ArmLayoutSpec(
            name=ARM_LAYOUT_RIGHT,
            active_slice=_RIGHT_SLICE,
            action_dim=AGIBOT_ARM_ACTION_DIM,
        )
    return ArmLayoutSpec(
        name=ARM_LAYOUT_DUAL,
        active_slice=_DUAL_SLICE,
        action_dim=AGIBOT_ROBOT_ACTION_DIM,
    )


def validate_arm_layout_dims(
    *,
    arm_layout: Any,
    action_dim: int,
    robot_action_dim: int,
) -> None:
    spec = get_arm_layout_spec(arm_layout)
    if int(robot_action_dim) != AGIBOT_ROBOT_ACTION_DIM:
        raise ValueError(
            "AgiBot robot_action_dim is fixed to "
            f"{AGIBOT_ROBOT_ACTION_DIM}, got {int(robot_action_dim)}"
        )
    if int(action_dim) != int(spec.action_dim):
        raise ValueError(
            f"env.arm_layout={spec.name!r} requires env.action_dim={spec.action_dim}, "
            f"got {int(action_dim)}"
        )


def project_canonical_vector(value: Any, arm_layout: Any) -> np.ndarray:
    spec = get_arm_layout_spec(arm_layout)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) != AGIBOT_ROBOT_ACTION_DIM:
        raise ValueError(
            f"Canonical AgiBot vector must be {AGIBOT_ROBOT_ACTION_DIM}D, got {arr.shape}"
        )
    return np.asarray(arr[spec.active_slice], dtype=np.float32)


def project_vector_to_layout(
    value: Any,
    arm_layout: Any,
    *,
    source_name: str = "AgiBot vector",
) -> np.ndarray:
    spec = get_arm_layout_spec(arm_layout)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) == 30:
        arr = arr[-AGIBOT_ROBOT_ACTION_DIM:]
    if int(arr.shape[0]) == AGIBOT_ROBOT_ACTION_DIM:
        return project_canonical_vector(arr, spec.name)
    if int(arr.shape[0]) == AGIBOT_ARM_ACTION_DIM:
        if spec.name == ARM_LAYOUT_DUAL:
            raise ValueError(
                f"{source_name} is {AGIBOT_ARM_ACTION_DIM}D and cannot be used with "
                "env.arm_layout='dual_arm'"
            )
        return np.asarray(arr, dtype=np.float32)
    raise ValueError(
        f"{source_name} must be {AGIBOT_ARM_ACTION_DIM}D, "
        f"{AGIBOT_ROBOT_ACTION_DIM}D, or 30D, got {arr.shape}"
    )


def project_chunk_to_layout(
    value: Any,
    arm_layout: Any,
    *,
    source_name: str = "AgiBot action chunk",
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{source_name} must be rank-2, got {arr.shape}")
    return np.stack(
        [
            project_vector_to_layout(row, arm_layout, source_name=source_name)
            for row in arr
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def embed_logical_action(
    logical_action: Any,
    current_robot_state: Any,
    arm_layout: Any,
) -> np.ndarray:
    spec = get_arm_layout_spec(arm_layout)
    action = np.asarray(logical_action, dtype=np.float32).reshape(-1)
    if int(action.shape[0]) != int(spec.action_dim):
        raise ValueError(
            f"Logical action for env.arm_layout={spec.name!r} must be "
            f"{spec.action_dim}D, got {action.shape}"
        )
    if spec.name == ARM_LAYOUT_DUAL:
        return np.asarray(action, dtype=np.float32)

    robot_state = np.asarray(current_robot_state, dtype=np.float32).reshape(-1)
    if int(robot_state.shape[0]) != AGIBOT_ROBOT_ACTION_DIM:
        raise ValueError(
            f"Current robot state must be {AGIBOT_ROBOT_ACTION_DIM}D, got {robot_state.shape}"
        )
    physical_action = np.asarray(robot_state, dtype=np.float32).copy()
    physical_action[spec.active_slice] = action
    return np.asarray(physical_action, dtype=np.float32)


def embed_logical_action_chunk(
    logical_actions: Any,
    current_robot_state: Any,
    arm_layout: Any,
) -> np.ndarray:
    actions = np.asarray(logical_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Logical action chunk must be rank-2, got {actions.shape}")
    return np.stack(
        [
            embed_logical_action(action, current_robot_state, arm_layout)
            for action in actions
        ],
        axis=0,
    ).astype(np.float32, copy=False)


__all__ = [
    "AGIBOT_ARM_ACTION_DIM",
    "AGIBOT_ROBOT_ACTION_DIM",
    "ARM_LAYOUT_DUAL",
    "ARM_LAYOUT_LEFT",
    "ARM_LAYOUT_RIGHT",
    "ArmLayoutSpec",
    "embed_logical_action",
    "embed_logical_action_chunk",
    "get_arm_layout_spec",
    "normalize_arm_layout",
    "project_canonical_vector",
    "project_chunk_to_layout",
    "project_vector_to_layout",
    "validate_arm_layout_dims",
]
