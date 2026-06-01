"""Read current AgiBot left/right EE pose without commanding the robot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


AGIBOT_REAL_ROOT = Path(__file__).resolve().parents[1]
SERL_REPO_PARENT = Path(__file__).resolve().parents[4]
if str(SERL_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(SERL_REPO_PARENT))

from serl_torch.examples.agibot_real.robot.interface import AgiBotRobotNode
from serl_torch.examples.agibot_real.robot.retargeter import BodyRetargeter
from serl_torch.examples.agibot_real.robot.retargeter import build_joint_array
from serl_torch.examples.agibot_real.robot.retargeter import slice_action_vector


def _resolve_asset_path(explicit_path: str | None, *, default_name: str) -> str:
    if explicit_path:
        return str(Path(explicit_path).expanduser().resolve())
    return str(
        (
            AGIBOT_REAL_ROOT
            / "assets"
            / "G1"
            / default_name
        ).resolve()
    )


def _build_state_vec(robot_node: AgiBotRobotNode) -> tuple[np.ndarray, dict[str, Any]]:
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


def _compute_base_frame_poses(
    retargeter: BodyRetargeter,
    state_vec: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
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
    left_rpy = retargeter.R.from_matrix(t_base_left[:3, :3]).as_euler(
        "xyz",
        degrees=True,
    )
    right_rpy = retargeter.R.from_matrix(t_base_right[:3, :3]).as_euler(
        "xyz",
        degrees=True,
    )
    return (left_pos, left_rpy), (right_pos, right_rpy)


def _format_pose(label: str, pos: np.ndarray, rpy_deg: np.ndarray) -> str:
    return (
        f"{label} "
        f"xyz=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) "
        f"rpy_deg=({rpy_deg[0]:+.2f}, {rpy_deg[1]:+.2f}, {rpy_deg[2]:+.2f})"
    )


def _print_once(
    *,
    robot_node: AgiBotRobotNode,
    retargeter: BodyRetargeter,
    show_joints: bool,
) -> None:
    state_vec, raw = _build_state_vec(robot_node)
    (left_cam_pos, left_cam_axisangle), (
        right_cam_pos,
        right_cam_axisangle,
    ) = retargeter.process_kinematics(state_vec)
    left_cam_rpy = retargeter.R.from_rotvec(left_cam_axisangle[0]).as_euler(
        "xyz",
        degrees=True,
    )
    right_cam_rpy = retargeter.R.from_rotvec(right_cam_axisangle[0]).as_euler(
        "xyz",
        degrees=True,
    )
    (left_base_pos, left_base_rpy), (right_base_pos, right_base_rpy) = (
        _compute_base_frame_poses(retargeter, state_vec)
    )

    print("=" * 88)
    print(f"timestamp_ns={time.time_ns()}")
    print("[camera_frame]")
    print(_format_pose("left ", left_cam_pos[0], left_cam_rpy))
    print(_format_pose("right", right_cam_pos[0], right_cam_rpy))
    print("[base_frame]")
    print(_format_pose("left ", left_base_pos, left_base_rpy))
    print(_format_pose("right", right_base_pos, right_base_rpy))
    print(
        "gripper "
        f"left={float(raw['joint_state_16'][7]):+.4f} "
        f"right={float(raw['joint_state_16'][15]):+.4f}"
    )
    if show_joints:
        print(f"head_rad={np.array2string(raw['head_rad'], precision=4)}")
        print(f"waist={np.array2string(raw['waist'], precision=4)}")
        print(f"arm_14={np.array2string(raw['arm_14'], precision=4)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read current AgiBot EE pose without sending robot commands.",
    )
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--show-joints", action="store_true")
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--camera-extrinsic-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    urdf_path = _resolve_asset_path(args.urdf_path, default_name="model.urdf")
    camera_extrinsic_path = _resolve_asset_path(
        args.camera_extrinsic_path,
        default_name="head_extrinsic_ours.json",
    )
    retargeter = BodyRetargeter(
        urdf_path=urdf_path,
        camera_extrinsic_path=camera_extrinsic_path,
    )
    robot_node = AgiBotRobotNode(hz=float(args.hz))
    period = 1.0 / max(float(args.hz), 1e-6)
    try:
        while True:
            _print_once(
                robot_node=robot_node,
                retargeter=retargeter,
                show_joints=bool(args.show_joints),
            )
            if not bool(args.loop):
                break
            time.sleep(period)
    finally:
        robot_node.shutdown()


if __name__ == "__main__":
    main()
