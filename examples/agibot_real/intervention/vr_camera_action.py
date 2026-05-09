"""Quest VR intervention controller for AgiBot camera_position actions.

The real robot controller we verified by hand is most natural in robot base
frame: x+=front, y+=left, z+=up, and RPY follows the robot's right-hand-rule
base convention. The residual RL environment currently consumes 14D
``camera_position`` actions:

    left xyz + left axis-angle + left gripper
    right xyz + right axis-angle + right gripper

This module keeps the VR control state in base frame, then converts each target
back to the environment's camera-frame 14D action before execution or replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from ..robot.retargeter import BodyRetargeter
from ..robot.retargeter import build_joint_array
from ..robot.retargeter import slice_action_vector


VR_TO_BASE_SIGN = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
ROT_AXIS_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "rx": 0,
    "ry": 1,
    "rz": 2,
    "roll": 0,
    "pitch": 1,
    "yaw": 2,
}


@dataclass(frozen=True, slots=True)
class VRCameraActionConfig:
    hand: str = "right"
    max_delta: float = 0.10
    max_step: float = 0.006
    smoothing: float = 0.40
    command_deadband: float = 0.001
    max_rot_delta_deg: float = 35.0
    max_rot_step_deg: float = 2.0
    rot_smoothing: float = 0.12
    rotation_deadband_deg: float = 0.8
    rot_map: str = "-ry,-rz,rx"
    gripper_open: float = 0.0
    gripper_closed: float = 120.0
    gripper_deadband: float = 0.5


@dataclass(slots=True)
class VRCameraActionResult:
    camera_action: np.ndarray
    should_send: bool
    active: bool
    target_left_pos: np.ndarray
    target_left_euler: np.ndarray
    target_right_pos: np.ndarray
    target_right_euler: np.ndarray
    hand_action: np.ndarray
    info: dict[str, Any]


def parse_rot_map(rot_map: str) -> list[tuple[float, int]]:
    tokens = [token.strip().lower() for token in str(rot_map).split(",")]
    if len(tokens) != 3 or any(not token for token in tokens):
        raise ValueError(
            "rot_map must have exactly 3 comma-separated tokens, "
            "for example -ry,-rz,rx"
        )

    mapping: list[tuple[float, int]] = []
    for token in tokens:
        sign = 1.0
        if token[0] == "+":
            token = token[1:]
        elif token[0] == "-":
            sign = -1.0
            token = token[1:]
        if token not in ROT_AXIS_INDEX:
            valid = ", ".join(sorted(ROT_AXIS_INDEX))
            raise ValueError(f"Invalid rot_map axis '{token}'. Valid axes: {valid}")
        mapping.append((sign, ROT_AXIS_INDEX[token]))
    return mapping


def limit_vec_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector_arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector_arr))
    if norm > max_norm > 0.0:
        return vector_arr * (float(max_norm) / norm)
    return vector_arr


def trigger_to_gripper(
    trigger_value: Any,
    *,
    open_value: float,
    closed_value: float,
) -> float:
    trigger = float(np.clip(float(trigger_value or 0.0), 0.0, 1.0))
    return float(open_value + trigger * (closed_value - open_value))


def build_state_vec_from_robot_node(robot_node: Any) -> tuple[np.ndarray, dict[str, Any]]:
    joint_state = robot_node.get_joint_state()
    if joint_state is None:
        raise RuntimeError("Failed to read 16D arm/gripper joint state")

    head_states = np.asarray(robot_node.get_head_joint_states(), dtype=np.float32)
    waist_states = np.asarray(robot_node.get_waist_joint_states(), dtype=np.float32)
    arm_states = np.asarray(robot_node.get_arm_joint_states(), dtype=np.float32)
    joint_state_array = np.asarray(joint_state, dtype=np.float32)

    state_vec = np.zeros((1, 53), dtype=np.float32)
    state_vec[0, 28:35] = arm_states[:7]
    state_vec[0, 35:42] = arm_states[7:]
    state_vec[0, 42:43] = joint_state_array[7]
    state_vec[0, 43:44] = joint_state_array[15]
    state_vec[0, 51:53] = waist_states
    state_vec[0, 26:28] = head_states

    raw = {
        "joint_state_16": joint_state_array,
        "head_rad": head_states,
        "waist": waist_states,
        "arm_14": arm_states,
    }
    return state_vec, raw


def current_base_poses(
    retargeter: BodyRetargeter,
    state_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_slices = slice_action_vector(np.asarray(state_vec, dtype=np.float64)[0])
    waist = action_slices["waist"].copy()
    waist[0], waist[1] = waist[1], waist[0]
    action_slices["waist"] = waist

    q_left = build_joint_array(
        retargeter.kin_left_arm,
        action_slices,
        ("waist", "left_arm"),
    )
    q_right = build_joint_array(
        retargeter.kin_right_arm,
        action_slices,
        ("waist", "right_arm"),
    )
    t_base_left = retargeter.kin_left_arm.forward(q_left)
    t_base_right = retargeter.kin_right_arm.forward(q_right)

    left_pos = np.asarray(t_base_left[:3, 3], dtype=np.float64).reshape(3)
    right_pos = np.asarray(t_base_right[:3, 3], dtype=np.float64).reshape(3)
    left_euler = retargeter.R.from_matrix(t_base_left[:3, :3]).as_euler(
        "xyz",
        degrees=False,
    )
    right_euler = retargeter.R.from_matrix(t_base_right[:3, :3]).as_euler(
        "xyz",
        degrees=False,
    )
    return left_pos, left_euler, right_pos, right_euler


def _base_head_transform(
    retargeter: BodyRetargeter,
    state_vec: np.ndarray,
) -> np.ndarray:
    action_slices = slice_action_vector(np.asarray(state_vec, dtype=np.float64)[0])
    waist = action_slices["waist"].copy()
    waist[0], waist[1] = waist[1], waist[0]
    action_slices["waist"] = waist
    q_head = build_joint_array(retargeter.kin_head, action_slices, ("waist", "neck"))
    return np.asarray(retargeter.kin_head.forward(q_head), dtype=np.float64)


def _pose_matrix(
    retargeter: BodyRetargeter,
    pos: np.ndarray,
    euler: np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = retargeter.R.from_euler("xyz", euler).as_matrix()
    transform[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return transform


def base_targets_to_camera_action(
    *,
    retargeter: BodyRetargeter,
    state_vec: np.ndarray,
    left_pos: np.ndarray,
    left_euler: np.ndarray,
    left_gripper: float,
    right_pos: np.ndarray,
    right_euler: np.ndarray,
    right_gripper: float,
) -> np.ndarray:
    t_base_head = _base_head_transform(retargeter, state_vec)
    t_head_base = np.linalg.inv(t_base_head)

    t_cam_left = (
        retargeter.T_head_to_cam
        @ t_head_base
        @ _pose_matrix(retargeter, left_pos, left_euler)
    )
    t_cam_right = (
        retargeter.T_head_to_cam
        @ t_head_base
        @ _pose_matrix(retargeter, right_pos, right_euler)
    )

    left_cam_pos = np.asarray(t_cam_left[:3, 3], dtype=np.float64).reshape(3)
    right_cam_pos = np.asarray(t_cam_right[:3, 3], dtype=np.float64).reshape(3)
    left_axisangle = retargeter.R.from_matrix(t_cam_left[:3, :3]).as_rotvec()
    right_axisangle = retargeter.R.from_matrix(t_cam_right[:3, :3]).as_rotvec()

    return np.asarray(
        np.concatenate(
            [
                left_cam_pos,
                left_axisangle,
                np.asarray([left_gripper], dtype=np.float64),
                right_cam_pos,
                right_axisangle,
                np.asarray([right_gripper], dtype=np.float64),
            ]
        ),
        dtype=np.float32,
    )


def execute_camera_action(
    *,
    robot_node: Any,
    retargeter: BodyRetargeter,
    raw: dict[str, Any],
    camera_action: np.ndarray,
    trajectory_time: float,
) -> None:
    action_arr = np.asarray(camera_action, dtype=np.float32).reshape(14)
    left_pos = action_arr[:3].reshape(1, 3)
    left_aa = action_arr[3:6].reshape(1, 3)
    right_pos = action_arr[7:10].reshape(1, 3)
    right_aa = action_arr[10:13].reshape(1, 3)

    action_vec = np.zeros((1, 53), dtype=np.float32)
    action_vec[0, 51:53] = np.asarray(raw["waist"], dtype=np.float32)
    action_vec[0, 26:28] = np.asarray(raw["head_rad"], dtype=np.float32)

    (left_pos_base, left_euler), (right_pos_base, right_euler) = (
        retargeter.inverse_kinematics_from_camera_axisangle(
            left_pos,
            left_aa,
            right_pos,
            right_aa,
            action_vec,
        )
    )
    abs_action = np.concatenate(
        [
            left_pos_base[0],
            left_euler[0],
            right_pos_base[0],
            right_euler[0],
        ]
    )
    action_abs = {
        "observation_timestamp": int(time.time() * 1e9),
        "head_joint_states": np.rad2deg(raw["head_rad"]).tolist(),
        "waist_joint_states": raw["waist"].tolist(),
        "arm_joint_states": raw["arm_14"].tolist(),
        "arm_cmd": [abs_action.tolist()],
    }
    hand_action = np.asarray([action_arr[6], action_arr[13]], dtype=np.float32)
    robot_node.publish_abs_pose_command_and_hand(
        action_abs,
        hand_action,
        trajectory_reference_time=float(trajectory_time),
    )


class VRCameraActionController:
    """Stateful VR-to-camera_position action adapter."""

    def __init__(
        self,
        *,
        retargeter: BodyRetargeter,
        initial_state_vec: np.ndarray,
        initial_grippers: np.ndarray,
        config: VRCameraActionConfig | None = None,
    ) -> None:
        self.retargeter = retargeter
        self.config = config or VRCameraActionConfig()
        hand = str(self.config.hand).strip().lower()
        if hand not in {"right", "left"}:
            raise ValueError(f"Unsupported hand={self.config.hand!r}")
        self.hand = hand
        self.rot_map = parse_rot_map(self.config.rot_map)

        left_pos, left_euler, right_pos, right_euler = current_base_poses(
            retargeter,
            initial_state_vec,
        )
        self.target_left_pos = left_pos.copy()
        self.target_left_euler = left_euler.copy()
        self.target_right_pos = right_pos.copy()
        self.target_right_euler = right_euler.copy()
        self.commanded_left_pos = self.target_left_pos.copy()
        self.commanded_left_euler = self.target_left_euler.copy()
        self.commanded_right_pos = self.target_right_pos.copy()
        self.commanded_right_euler = self.target_right_euler.copy()

        self.hand_action = np.asarray(initial_grippers, dtype=np.float32).reshape(2)
        self.current_hand_idx = 1 if self.hand == "right" else 0
        self.commanded_gripper = float(self.hand_action[self.current_hand_idx])
        self.anchor_pos: np.ndarray | None = None
        self.anchor_euler: np.ndarray | None = None
        self.last_active = False

    def _active(self, signals: dict[str, Any]) -> bool:
        if self.hand == "right":
            return bool(dict(signals.get("button_states") or {}).get("RG", False))
        return bool(dict(signals.get("left_button_states") or {}).get("LG", False))

    def _vr_delta(self, signals: dict[str, Any]) -> np.ndarray:
        key = "position_delta" if self.hand == "right" else "left_position_delta"
        delta = np.asarray(signals.get(key, np.zeros(3)), dtype=np.float32).reshape(3)
        return delta * VR_TO_BASE_SIGN

    def _vr_rot_delta(self, signals: dict[str, Any]) -> np.ndarray:
        key = "rotation_delta" if self.hand == "right" else "left_rotation_delta"
        delta = np.asarray(signals.get(key, np.zeros(3)), dtype=np.float32).reshape(3)
        return np.asarray(
            [sign * float(delta[index]) for sign, index in self.rot_map],
            dtype=np.float32,
        )

    def update(
        self,
        *,
        signals: dict[str, Any],
        state_vec: np.ndarray,
    ) -> VRCameraActionResult:
        cfg = self.config
        active = self._active(signals)
        became_active = bool(active and not self.last_active)
        became_inactive = bool((not active) and self.last_active)
        if became_active:
            if self.hand == "right":
                self.anchor_pos = self.target_right_pos.copy()
                self.anchor_euler = self.target_right_euler.copy()
            else:
                self.anchor_pos = self.target_left_pos.copy()
                self.anchor_euler = self.target_left_euler.copy()
        elif became_inactive:
            self.anchor_pos = None
            self.anchor_euler = None

        trigger_key = "trigger" if self.hand == "right" else "left_trigger"
        self.hand_action[self.current_hand_idx] = trigger_to_gripper(
            signals.get(trigger_key, 0.0),
            open_value=float(cfg.gripper_open),
            closed_value=float(cfg.gripper_closed),
        )
        gripper_delta = abs(
            float(self.hand_action[self.current_hand_idx])
            - float(self.commanded_gripper)
        )

        pos_command_delta = 0.0
        rot_command_delta = 0.0
        max_delta = abs(float(cfg.max_delta))
        max_step = abs(float(cfg.max_step))
        smoothing = float(np.clip(float(cfg.smoothing), 0.0, 1.0))
        max_rot_delta = np.deg2rad(abs(float(cfg.max_rot_delta_deg)))
        max_rot_step = np.deg2rad(abs(float(cfg.max_rot_step_deg)))
        rot_smoothing = float(np.clip(float(cfg.rot_smoothing), 0.0, 1.0))

        if active and self.anchor_pos is not None and self.anchor_euler is not None:
            delta = limit_vec_norm(self._vr_delta(signals), max_delta)
            rot_delta = limit_vec_norm(self._vr_rot_delta(signals), max_rot_delta)
            desired_pos = self.anchor_pos + delta
            desired_euler = self.anchor_euler + rot_delta
            if self.hand == "right":
                filtered_pos = self.target_right_pos + smoothing * (
                    desired_pos - self.target_right_pos
                )
                self.target_right_pos = self.target_right_pos + limit_vec_norm(
                    filtered_pos - self.target_right_pos,
                    max_step,
                )
                filtered_euler = self.target_right_euler + rot_smoothing * (
                    desired_euler - self.target_right_euler
                )
                self.target_right_euler = self.target_right_euler + limit_vec_norm(
                    filtered_euler - self.target_right_euler,
                    max_rot_step,
                )
                pos_command_delta = float(
                    np.linalg.norm(self.target_right_pos - self.commanded_right_pos)
                )
                rot_command_delta = float(
                    np.linalg.norm(self.target_right_euler - self.commanded_right_euler)
                )
            else:
                filtered_pos = self.target_left_pos + smoothing * (
                    desired_pos - self.target_left_pos
                )
                self.target_left_pos = self.target_left_pos + limit_vec_norm(
                    filtered_pos - self.target_left_pos,
                    max_step,
                )
                filtered_euler = self.target_left_euler + rot_smoothing * (
                    desired_euler - self.target_left_euler
                )
                self.target_left_euler = self.target_left_euler + limit_vec_norm(
                    filtered_euler - self.target_left_euler,
                    max_rot_step,
                )
                pos_command_delta = float(
                    np.linalg.norm(self.target_left_pos - self.commanded_left_pos)
                )
                rot_command_delta = float(
                    np.linalg.norm(self.target_left_euler - self.commanded_left_euler)
                )

        camera_action = base_targets_to_camera_action(
            retargeter=self.retargeter,
            state_vec=state_vec,
            left_pos=self.target_left_pos,
            left_euler=self.target_left_euler,
            left_gripper=float(self.hand_action[0]),
            right_pos=self.target_right_pos,
            right_euler=self.target_right_euler,
            right_gripper=float(self.hand_action[1]),
        )
        should_send = bool(
            pos_command_delta >= abs(float(cfg.command_deadband))
            or rot_command_delta >= np.deg2rad(abs(float(cfg.rotation_deadband_deg)))
            or gripper_delta >= abs(float(cfg.gripper_deadband))
        )
        if should_send:
            if self.hand == "right":
                self.commanded_right_pos = self.target_right_pos.copy()
                self.commanded_right_euler = self.target_right_euler.copy()
            else:
                self.commanded_left_pos = self.target_left_pos.copy()
                self.commanded_left_euler = self.target_left_euler.copy()
            self.commanded_gripper = float(self.hand_action[self.current_hand_idx])
        self.last_active = bool(active)
        return VRCameraActionResult(
            camera_action=camera_action,
            should_send=should_send,
            active=bool(active),
            target_left_pos=self.target_left_pos.copy(),
            target_left_euler=self.target_left_euler.copy(),
            target_right_pos=self.target_right_pos.copy(),
            target_right_euler=self.target_right_euler.copy(),
            hand_action=self.hand_action.copy(),
            info={
                "hitl_intervention": bool(active),
                "hitl_active": bool(active),
                "hitl_became_active": bool(became_active),
                "hitl_became_inactive": bool(became_inactive),
                "hitl_source": "quest_vr",
                "hitl_hand": self.hand,
                "hitl_pos_command_delta": float(pos_command_delta),
                "hitl_rot_command_delta": float(rot_command_delta),
                "hitl_gripper_delta": float(gripper_delta),
            },
        )


__all__ = [
    "VRCameraActionConfig",
    "VRCameraActionController",
    "VRCameraActionResult",
    "base_targets_to_camera_action",
    "build_state_vec_from_robot_node",
    "current_base_poses",
    "execute_camera_action",
    "limit_vec_norm",
    "parse_rot_map",
    "trigger_to_gripper",
]
