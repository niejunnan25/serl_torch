"""Run Quest VR teleop through AgiBot camera_position actions.

This script is the SERL-side bridge test for HITL:

    Quest VR base-frame target -> 14D camera_position action -> robot command

Dry-run by default. Add --execute to command the robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.agibot_real.intervention import QuestVRClient
from serl_torch.examples.agibot_real.intervention import VRCameraActionConfig
from serl_torch.examples.agibot_real.intervention import VRCameraActionController
from serl_torch.examples.agibot_real.intervention import build_state_vec_from_robot_node
from serl_torch.examples.agibot_real.intervention import execute_camera_action
from serl_torch.examples.agibot_real.robot.interface import AgiBotRobotNode
from serl_torch.examples.agibot_real.robot.retargeter import BodyRetargeter


def _resolve_asset_path(explicit_path: str | None, *, default_name: str) -> str:
    if explicit_path:
        return str(Path(explicit_path).expanduser().resolve())
    return str(
        (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "G1"
            / default_name
        ).resolve()
    )


def _format_pose(label: str, pos: np.ndarray, euler: np.ndarray) -> str:
    rpy_deg = np.rad2deg(np.asarray(euler, dtype=np.float64).reshape(3))
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    return (
        f"{label} "
        f"xyz=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
        f"rpy_deg=({rpy_deg[0]:+.1f},{rpy_deg[1]:+.1f},{rpy_deg[2]:+.1f})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quest VR camera_position action bridge. Dry-run by default; "
            "add --execute to command robot."
        )
    )
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--trajectory-time", type=float, default=0.035)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--max-delta", type=float, default=0.10)
    parser.add_argument("--max-step", type=float, default=0.006)
    parser.add_argument("--smoothing", type=float, default=0.40)
    parser.add_argument("--command-deadband", type=float, default=0.001)
    parser.add_argument("--max-rot-delta-deg", type=float, default=35.0)
    parser.add_argument("--max-rot-step-deg", type=float, default=2.0)
    parser.add_argument("--rot-smoothing", type=float, default=0.12)
    parser.add_argument("--rotation-deadband-deg", type=float, default=0.8)
    parser.add_argument("--rot-map", default="-ry,-rz,rx")
    parser.add_argument("--gripper-open", type=float, default=0.0)
    parser.add_argument("--gripper-closed", type=float, default=120.0)
    parser.add_argument("--gripper-deadband", type=float, default=0.5)
    parser.add_argument("--scaling-factor", type=float, default=0.5)
    parser.add_argument("--control-freq", type=int, default=30)
    parser.add_argument("--coordinate-mapping", default="sim")
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--camera-extrinsic-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    period = 1.0 / max(float(args.hz), 1e-6)
    urdf_path = _resolve_asset_path(args.urdf_path, default_name="model.urdf")
    camera_extrinsic_path = _resolve_asset_path(
        args.camera_extrinsic_path,
        default_name="head_extrinsic_ours.json",
    )
    retargeter = BodyRetargeter(
        urdf_path=urdf_path,
        camera_extrinsic_path=camera_extrinsic_path,
    )
    print("Initializing AgiBotRobotNode...", flush=True)
    robot_node = AgiBotRobotNode(hz=float(args.hz))
    print("AgiBotRobotNode ready.", flush=True)

    try:
        state_vec, raw = build_state_vec_from_robot_node(robot_node)
        initial_grippers = np.asarray(
            [raw["joint_state_16"][7], raw["joint_state_16"][15]],
            dtype=np.float32,
        )
        controller = VRCameraActionController(
            retargeter=retargeter,
            initial_state_vec=state_vec,
            initial_grippers=initial_grippers,
            config=VRCameraActionConfig(
                hand=str(args.hand),
                max_delta=float(args.max_delta),
                max_step=float(args.max_step),
                smoothing=float(args.smoothing),
                command_deadband=float(args.command_deadband),
                max_rot_delta_deg=float(args.max_rot_delta_deg),
                max_rot_step_deg=float(args.max_rot_step_deg),
                rot_smoothing=float(args.rot_smoothing),
                rotation_deadband_deg=float(args.rotation_deadband_deg),
                rot_map=str(args.rot_map),
                gripper_open=float(args.gripper_open),
                gripper_closed=float(args.gripper_closed),
                gripper_deadband=float(args.gripper_deadband),
            ),
        )

        print("=" * 88, flush=True)
        print(
            f"VR camera-action client hand={args.hand} hz={float(args.hz):.2f} "
            f"execute={bool(args.execute)} trajectory_time={args.trajectory_time:.3f}s "
            f"rot_map={args.rot_map}",
            flush=True,
        )
        print("Hold RG for right hand or LG for left hand. Ctrl+C to stop.", flush=True)

        with QuestVRClient(
            scaling_factor=float(args.scaling_factor),
            motion_mode="xyzrxryrz",
            control_freq=int(args.control_freq),
            enable_visualization=False,
            controller_mode=str(args.hand),
            coordinate_mapping=str(args.coordinate_mapping),
        ) as vr_client:
            while True:
                loop_t0 = time.monotonic()
                state_vec, raw = build_state_vec_from_robot_node(robot_node)
                result = controller.update(
                    signals=vr_client.snapshot().signals,
                    state_vec=state_vec,
                )
                if bool(result.info["hitl_became_active"]):
                    print("camera-action teleop active", flush=True)
                elif bool(result.info["hitl_became_inactive"]):
                    print("camera-action teleop inactive", flush=True)

                if result.should_send:
                    if str(args.hand) == "right":
                        pose_line = _format_pose(
                            "right_target",
                            result.target_right_pos,
                            result.target_right_euler,
                        )
                        grip = float(result.hand_action[1])
                    else:
                        pose_line = _format_pose(
                            "left_target ",
                            result.target_left_pos,
                            result.target_left_euler,
                        )
                        grip = float(result.hand_action[0])
                    print(
                        f"{pose_line} gripper={grip:.1f} "
                        f"camera_action={np.array2string(result.camera_action, precision=4, suppress_small=True)}",
                        flush=True,
                    )
                    if bool(args.execute):
                        execute_camera_action(
                            robot_node=robot_node,
                            retargeter=retargeter,
                            raw=raw,
                            camera_action=result.camera_action,
                            trajectory_time=float(args.trajectory_time),
                        )

                sleep_s = period - (time.monotonic() - loop_t0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
    finally:
        if bool(args.shutdown):
            print("Shutting down AgiBotRobotNode...", flush=True)
            robot_node.shutdown()
            print("AgiBotRobotNode shutdown complete.", flush=True)
        else:
            print(
                "Skipping AgiBotRobotNode.shutdown() to avoid SDK release blocking.",
                flush=True,
            )


if __name__ == "__main__":
    main()
