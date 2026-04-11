"""Minimal AgiBot robot interface wrapper used by the real-robot env."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import ruckig
except ImportError:
    ruckig = None  # type: ignore[assignment]


class AgiBotRobotNode:
    """Thin wrapper around the external AgiBot DDS / controller SDK."""

    def __init__(self, *, hz: float = 20.0) -> None:
        try:
            from .sdk_bootstrap import ensure_repo_local_a2d_sdk

            ensure_repo_local_a2d_sdk()
            from a2d_sdk.robot import CosineCamera
            from a2d_sdk.robot import RobotController
            from a2d_sdk.robot import RobotDds
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Repo-local AgiBot SDK bootstrap failed. "
                "Make sure you are using Python 3.10 on the robot machine and "
                "that examples/agibot_real/vendor/a2d_sdk is present."
            ) from exc

        self.robot = RobotDds()
        self.camera = CosineCamera(["head", "hand_left", "hand_right"])
        self.robot_controller = RobotController()
        self.period = 1.0 / float(hz)
        self._t0 = time.time()
        self._wait_until_ready()

    def _wait_until_ready(self, *, timeout_sec: float = 10.0) -> None:
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            img, _ = self.camera.get_latest_image("head")
            pos, _ = self.robot.arm_joint_states()
            grip, _ = self.robot.gripper_states()
            if img is not None and pos is not None and grip is not None:
                return
            time.sleep(0.01)
        raise RuntimeError(
            "Timed out waiting for AgiBot camera / joint state readiness"
        )

    def _poll_state(
        self,
        getter: Any,
        *,
        length: int,
        timeout_sec: float = 2.0,
        interval_sec: float = 0.1,
    ) -> list[float]:
        deadline = time.time() + float(timeout_sec)
        last_vals = None
        while time.time() < deadline:
            vals, _ = getter()
            last_vals = vals
            try:
                values = list(vals)
            except Exception:  # noqa: BLE001
                values = []
            if len(values) == int(length):
                try:
                    return [float(v) for v in values]
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(interval_sec)
        raise RuntimeError(
            f"Robot state getter did not become ready: last={last_vals!r}"
        )

    def get_img_head(self) -> np.ndarray | None:
        img, _ = self.camera.get_latest_image("head")
        return img

    def get_img_left_wrist(self) -> np.ndarray | None:
        img, _ = self.camera.get_latest_image("hand_left")
        return img

    def get_img_right_wrist(self) -> np.ndarray | None:
        img, _ = self.camera.get_latest_image("hand_right")
        return img

    def get_joint_state(self) -> list[float] | None:
        pos, _ = self.robot.arm_joint_states()
        grip, _ = self.robot.gripper_states()
        if pos is None or grip is None:
            return None
        return list(pos[:7]) + [float(grip[0])] + list(pos[7:]) + [float(grip[1])]

    def get_head_joint_states(self) -> list[float]:
        head_states = self._poll_state(self.robot.head_joint_states, length=2)
        return np.deg2rad(head_states).tolist()

    def get_waist_joint_states(self) -> list[float]:
        return self._poll_state(self.robot.waist_joint_states, length=2)

    def get_arm_joint_states(self) -> list[float]:
        return self._poll_state(self.robot.arm_joint_states, length=14)

    def publish_joint_command_direct(self, action: np.ndarray) -> None:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != 16:
            raise ValueError(f"Expected 16D joint action, got {action_arr.shape}")
        target_positions = np.concatenate((action_arr[:7], action_arr[8:15]))
        self.robot.move_arm(target_positions)
        self.robot.move_gripper([float(action_arr[7]), float(action_arr[15])])

    def publish_joint_command_reset(self, action: np.ndarray) -> None:
        """Smooth move to 16D joint+gripper target (same idea as tangyili utils_robot.RobotNode).

        Uses ruckig when installed; otherwise one-shot ``publish_joint_command_direct``.
        """
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != 16:
            raise ValueError(f"Expected 16D joint action, got {action_arr.shape}")
        target_positions = np.concatenate((action_arr[:7], action_arr[8:15]))

        if ruckig is None:
            self.publish_joint_command_direct(action_arr)
            return

        current_positions, _ = self.robot.arm_joint_states()
        if not current_positions:
            raise RuntimeError("Failed to get arm joint states for reset trajectory")

        dof = 14
        interval = 0.01
        cur = list(map(float, current_positions))
        tgt = list(map(float, np.asarray(target_positions).reshape(-1)))
        if len(cur) != dof or len(tgt) != dof:
            raise RuntimeError(
                f"Expected {dof} arm joints, got cur={len(cur)} tgt={len(tgt)}"
            )

        try:
            rk = ruckig.Ruckig(dof, interval)
            rk_input = ruckig.InputParameter(dof)
            rk_output = ruckig.OutputParameter(dof)

            rk_input.current_position = cur
            rk_input.current_velocity = [0.0] * dof
            rk_input.current_acceleration = [0.0] * dof
            rk_input.target_position = tgt
            rk_input.target_velocity = [0.0] * dof
            rk_input.target_acceleration = [0.0] * dof
            rk_input.max_velocity = [2.0] * dof
            rk_input.max_acceleration = [1.0] * dof
            rk_input.max_jerk = [5.0] * dof

            while rk.update(rk_input, rk_output) == ruckig.Result.Working:
                self.robot.move_arm(rk_output.new_position)
                rk_output.pass_to_input(rk_input)
                time.sleep(interval)
        except Exception:
            self.publish_joint_command_direct(action_arr)
            return

        self.robot.move_gripper([float(action_arr[7]), float(action_arr[15])])

    def publish_head_command(self, target_positions: np.ndarray) -> None:
        self.robot.move_head(np.asarray(target_positions, dtype=np.float32).tolist())

    def publish_waist_command(self, target_positions: np.ndarray) -> None:
        self.robot.move_waist(np.asarray(target_positions, dtype=np.float32).tolist())

    def publish_abs_pose_command_and_hand(
        self,
        action: dict[str, Any],
        hand_action: np.ndarray,
        *,
        trajectory_reference_time: float,
    ) -> None:
        robot_states = {
            "head": action["head_joint_states"],
            "waist": action["waist_joint_states"],
            "arm": action["arm_joint_states"],
        }
        robot_actions = [
            {
                "left_arm": {
                    "action_data": abs_pose[:6],
                    "control_type": "ABS_POSE",
                },
                "right_arm": {
                    "action_data": abs_pose[6:12],
                    "control_type": "ABS_POSE",
                },
            }
            for abs_pose in action["arm_cmd"]
        ]
        self.robot_controller.trajectory_tracking_control(
            infer_timestamp=action["observation_timestamp"],
            robot_states=robot_states,
            robot_actions=robot_actions,
            robot_link="base_link",
            trajectory_reference_time=float(trajectory_reference_time),
        )
        hand_arr = np.asarray(hand_action, dtype=np.float32).reshape(-1)
        self.robot.move_gripper([float(hand_arr[0]), float(hand_arr[1])])

    def shutdown(self) -> None:
        self.robot.shutdown()
