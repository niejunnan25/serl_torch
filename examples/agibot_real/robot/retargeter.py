"""Self-contained kinematics helpers for AgiBot camera-position control."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
from typing import Optional

import numpy as np

ACTION_LAYOUT = {
    "neck": (26, 28),
    "left_arm": (28, 35),
    "right_arm": (35, 42),
    "left_hand": (42, 43),
    "right_hand": (43, 44),
    "waist": (51, 53),
}


def slice_action_vector(action: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    return {
        group: np.asarray(arr[start:end], dtype=np.float64)
        for group, (start, end) in ACTION_LAYOUT.items()
    }


def classify_joint(joint_name: str) -> str:
    name = joint_name.lower()
    if any(token in name for token in ("waist", "torso", "pelvis", "body_joint", "body_link")):
        return "waist"
    if "neck" in name or "head" in name:
        return "neck"
    if any(token in name for token in ("left", "l_", "_l", "arm_l")):
        if "hand" in name or "finger" in name:
            return "left_hand"
        return "left_arm"
    if any(token in name for token in ("right", "r_", "_r", "arm_r")):
        if "hand" in name or "finger" in name:
            return "right_hand"
        return "right_arm"
    raise KeyError(f"Unable to infer joint group for {joint_name!r}")


def _resolve_group_value_index(group: str, joint_name: str) -> Optional[int]:
    if group != "waist":
        return None
    match = re.search(r"joint(\d+)", joint_name.lower())
    if match is None:
        return None
    joint_id = int(match.group(1))
    if joint_id == 1:
        return 0
    if joint_id == 2:
        return 1
    return None


def build_joint_array(
    kin: object,
    action_slices: dict[str, np.ndarray],
    allowed_groups: Iterable[str],
) -> list[float]:
    allowed = set(allowed_groups)
    counters = {group: 0 for group in allowed}
    joint_names = kin.get_joint_names()
    arr = [0.0] * len(joint_names)
    for idx, joint_name in enumerate(joint_names):
        group = classify_joint(joint_name)
        if group not in allowed:
            raise KeyError(f"Joint group {group!r} is not allowed for {joint_name!r}")
        values = action_slices.get(group, np.asarray([], dtype=np.float64))
        offset = counters[group]
        mapped_index = _resolve_group_value_index(group, joint_name)
        value_index = mapped_index if mapped_index is not None else offset
        if value_index >= len(values):
            raise IndexError(f"Action slice for group {group!r} is too short")
        arr[idx] = float(values[value_index])
        counters[group] = offset + 1
    return arr


def load_camera_extrinsic_cam_to_head(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    extrinsic = data.get("extrinsic", data)
    rotation = np.asarray(extrinsic["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(extrinsic["translation_vector"], dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


class BodyRetargeter:
    """Convert between camera-space 14D poses and robot-base absolute poses."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        camera_extrinsic_path: str | Path,
        base_link: str = "base_link",
        head_tip_link: str = "head_link2",
        left_hand_tip_link: str = "arm_left_link7",
        right_hand_tip_link: str = "arm_right_link7",
    ) -> None:
        from scipy.spatial.transform import Rotation as R
        from urdf_parser_py.urdf import URDF
        from pykdl_utils.kdl_kinematics import KDLKinematics
        from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model

        self.R = R
        robot_urdf = URDF.from_xml_file(str(Path(urdf_path).resolve()))
        tree = kdl_tree_from_urdf_model(robot_urdf)
        self.kin_head = KDLKinematics(robot_urdf, base_link, head_tip_link, tree)
        self.kin_left_arm = KDLKinematics(robot_urdf, base_link, left_hand_tip_link, tree)
        self.kin_right_arm = KDLKinematics(robot_urdf, base_link, right_hand_tip_link, tree)
        self.T_cam_to_head = load_camera_extrinsic_cam_to_head(camera_extrinsic_path)
        self.T_head_to_cam = np.linalg.inv(self.T_cam_to_head)

    def process_kinematics(
        self,
        action_vectors: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        action_vectors = np.asarray(action_vectors, dtype=np.float64)
        squeeze_output = action_vectors.ndim == 1
        if squeeze_output:
            action_vectors = action_vectors[np.newaxis, :]
        batch_size = action_vectors.shape[0]

        left_positions = np.zeros((batch_size, 3), dtype=np.float64)
        right_positions = np.zeros((batch_size, 3), dtype=np.float64)
        left_axisangles = np.zeros((batch_size, 3), dtype=np.float64)
        right_axisangles = np.zeros((batch_size, 3), dtype=np.float64)

        for idx in range(batch_size):
            action_slices = slice_action_vector(action_vectors[idx])
            waist = action_slices["waist"].copy()
            waist[0], waist[1] = waist[1], waist[0]
            action_slices["waist"] = waist

            q_head = build_joint_array(self.kin_head, action_slices, ("waist", "neck"))
            q_left = build_joint_array(self.kin_left_arm, action_slices, ("waist", "left_arm"))
            q_right = build_joint_array(self.kin_right_arm, action_slices, ("waist", "right_arm"))

            t_base_head = self.kin_head.forward(q_head)
            t_base_left = self.kin_left_arm.forward(q_left)
            t_base_right = self.kin_right_arm.forward(q_right)

            t_head_left = np.linalg.inv(t_base_head) @ t_base_left
            t_head_right = np.linalg.inv(t_base_head) @ t_base_right
            t_cam_left = self.T_head_to_cam @ t_head_left
            t_cam_right = self.T_head_to_cam @ t_head_right

            left_positions[idx] = t_cam_left[:3, 3]
            right_positions[idx] = t_cam_right[:3, 3]
            left_axisangles[idx] = self.R.from_matrix(t_cam_left[:3, :3]).as_rotvec()
            right_axisangles[idx] = self.R.from_matrix(t_cam_right[:3, :3]).as_rotvec()

        if squeeze_output:
            return (
                left_positions[0],
                left_axisangles[0],
            ), (
                right_positions[0],
                right_axisangles[0],
            )
        return (left_positions, left_axisangles), (right_positions, right_axisangles)

    def inverse_kinematics_from_camera_axisangle(
        self,
        left_hand_pos: np.ndarray,
        left_hand_axisangle: np.ndarray,
        right_hand_pos: np.ndarray,
        right_hand_axisangle: np.ndarray,
        action_vector: np.ndarray,
        *,
        euler_order: str = "xyz",
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        left_hand_pos = np.asarray(left_hand_pos, dtype=np.float64)
        left_hand_axisangle = np.asarray(left_hand_axisangle, dtype=np.float64)
        right_hand_pos = np.asarray(right_hand_pos, dtype=np.float64)
        right_hand_axisangle = np.asarray(right_hand_axisangle, dtype=np.float64)
        action_vector = np.asarray(action_vector, dtype=np.float64)

        squeeze_output = left_hand_pos.ndim == 1
        if squeeze_output:
            left_hand_pos = left_hand_pos[np.newaxis, :]
            left_hand_axisangle = left_hand_axisangle[np.newaxis, :]
            right_hand_pos = right_hand_pos[np.newaxis, :]
            right_hand_axisangle = right_hand_axisangle[np.newaxis, :]
        if action_vector.ndim == 1:
            action_vector = action_vector[np.newaxis, :]

        batch_size = left_hand_pos.shape[0]
        left_pos_base = np.zeros((batch_size, 3), dtype=np.float64)
        right_pos_base = np.zeros((batch_size, 3), dtype=np.float64)
        left_euler = np.zeros((batch_size, 3), dtype=np.float64)
        right_euler = np.zeros((batch_size, 3), dtype=np.float64)

        for idx in range(batch_size):
            action_slices = slice_action_vector(action_vector[idx])
            waist = action_slices["waist"].copy()
            waist[0], waist[1] = waist[1], waist[0]
            action_slices["waist"] = waist
            q_head = build_joint_array(self.kin_head, action_slices, ("waist", "neck"))
            t_base_head = self.kin_head.forward(q_head)

            t_cam_left = np.eye(4, dtype=np.float64)
            t_cam_left[:3, :3] = self.R.from_rotvec(left_hand_axisangle[idx]).as_matrix()
            t_cam_left[:3, 3] = left_hand_pos[idx]
            t_cam_right = np.eye(4, dtype=np.float64)
            t_cam_right[:3, :3] = self.R.from_rotvec(right_hand_axisangle[idx]).as_matrix()
            t_cam_right[:3, 3] = right_hand_pos[idx]

            t_head_left = self.T_cam_to_head @ t_cam_left
            t_head_right = self.T_cam_to_head @ t_cam_right
            t_base_left = t_base_head @ t_head_left
            t_base_right = t_base_head @ t_head_right

            left_pos_base[idx] = t_base_left[:3, 3]
            right_pos_base[idx] = t_base_right[:3, 3]
            left_euler[idx] = self.R.from_matrix(t_base_left[:3, :3]).as_euler(
                euler_order,
                degrees=False,
            )
            right_euler[idx] = self.R.from_matrix(t_base_right[:3, :3]).as_euler(
                euler_order,
                degrees=False,
            )

        if squeeze_output:
            return (left_pos_base[0], left_euler[0]), (right_pos_base[0], right_euler[0])
        return (left_pos_base, left_euler), (right_pos_base, right_euler)

