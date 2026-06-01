"""Continuous Quest VR 6D teleop in robot base frame with SO3 rotation target.

Default mode is dry-run. Add --execute to command the robot.
Verified right-hand empirical defaults:
  translation: base x+=front, y+=left, z+=up
  rotation: roll=-VR pitch, pitch=-VR yaw, yaw=+VR roll

Unlike run_vr_base_pose_teleop.py, this script keeps the internal target
orientation as scipy Rotation and only converts to RPY at the command boundary.

python agibot_hitl_tests/run_vr_base_pose_so3_teleop.py \
  --hand right \
  --execute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


AGIBOT_REAL_ROOT = Path(__file__).resolve().parents[1]
SERL_REPO_PARENT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SERL_REPO_PARENT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from read_current_ee_pose import _build_state_vec
from read_current_ee_pose import _format_pose
from read_current_ee_pose import _resolve_asset_path
from serl_torch.examples.agibot_real.intervention import QuestVRClient
from serl_torch.examples.agibot_real.robot.interface import AgiBotRobotNode
from serl_torch.examples.agibot_real.robot.init_positions import get_task_initial_pose
from serl_torch.examples.agibot_real.robot.init_positions import (
    normalize_task_name_for_init_pose,
)
from serl_torch.examples.agibot_real.robot.retargeter import BodyRetargeter
from serl_torch.examples.agibot_real.robot.retargeter import build_joint_array
from serl_torch.examples.agibot_real.robot.retargeter import slice_action_vector


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Quest controller input and apply base-frame xyz+rpy teleop. "
            "Dry-run by default; add --execute to command robot."
        ),
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
    parser.add_argument(
        "--continuous-euler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Choose the RPY representation nearest to the previous command "
            "when converting SO3 targets to robot SDK xyz+rpy commands. This "
            "reduces 180/-180 branch jumps at the command boundary."
        ),
    )
    parser.add_argument(
        "--rot-map",
        default="-ry,-rz,rx",
        help=(
            "Map raw VR rotation delta to robot base RPY. "
            "Verified right-hand default: -ry,-rz,rx."
        ),
    )
    parser.add_argument("--gripper-open", type=float, default=0.0)
    parser.add_argument("--gripper-closed", type=float, default=120.0)
    parser.add_argument("--gripper-deadband", type=float, default=0.5)
    parser.add_argument("--task-name", default="perdon")
    parser.add_argument(
        "--reset-button",
        default="auto",
        help="Button used to reset to task initial pose. auto means B for right, Y for left.",
    )
    parser.add_argument("--reset-sleep-head-waist", type=float, default=2.0)
    parser.add_argument("--reset-sleep-arm", type=float, default=1.0)
    parser.add_argument("--scaling-factor", type=float, default=0.5)
    parser.add_argument("--control-freq", type=int, default=30)
    parser.add_argument("--coordinate-mapping", default="sim")
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--camera-extrinsic-path", default=None)
    parser.add_argument(
        "--control-mode",
        choices=("absolute", "incremental"),
        default="absolute",
        help=(
            "absolute: target = grip-anchor robot pose + current VR delta. "
            "incremental: target accumulates per-frame VR delta changes."
        ),
    )
    parser.add_argument(
        "--rotation-update",
        choices=("euler", "so3"),
        default="so3",
        help=(
            "Kept for CLI compatibility. This SO3 script always keeps the "
            "internal target orientation as a Rotation object."
        ),
    )
    parser.add_argument(
        "--so3-order",
        choices=("base", "local"),
        default="base",
        help=(
            "base: R_target = R_delta * R_anchor, matching the original HITL "
            "matrix update. local: R_target = R_anchor * R_delta."
        ),
    )
    parser.add_argument(
        "--rotation-calibration",
        default=None,
        help=(
            "Path to calibration JSON containing R_B_V. When set, rotation "
            "uses raw Quest controller matrices and applies "
            "Delta_R_B = R_B_V * Delta_R_V * R_B_V^-1. "
            "In this mode --rot-map is only kept for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--calibrated-rotation-mode",
        choices=("relative", "tool", "direct", "grip-local"),
        default="direct",
        help=(
            "When --rotation-calibration is set: direct uses "
            "R_B_E = R_B_Cmap * R_V_C. relative uses grip-time "
            "Delta_R_B * R_B_E0. tool uses the full calibrated formula "
            "R_B_E = R_B_V * R_V_C * R_C_E. If R_C_E is missing from the "
            "calibration file, tool mode computes a session R_C_E at grip time "
            "to avoid target jumps. grip-local uses only the corrected "
            "controller local relative rotation since grip press: "
            "R_B_E = R_B_E0 * (R_V_C0^-1 * R_V_C)."
        ),
    )
    parser.add_argument(
        "--grip-local-rot-map",
        default="rx,ry,rz",
        help=(
            "Only used with --calibrated-rotation-mode grip-local. Maps the "
            "controller local relative rotvec to robot local rotvec. Example: "
            "-rx,rz,ry means robot roll=-controller roll, "
            "robot pitch=controller yaw, robot yaw=controller pitch."
        ),
    )
    parser.add_argument(
        "--print-delta",
        action="store_true",
        help="Print mapped VR xyz/rpy deltas in addition to target pose.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show live target/actual xyz+rpy curves for the controlled EE.",
    )
    parser.add_argument("--plot-window", type=float, default=20.0)
    parser.add_argument("--plot-rate", type=float, default=10.0)
    parser.add_argument("--actual-rate", type=float, default=5.0)
    parser.add_argument(
        "--plot-save-path",
        default=None,
        help=(
            "Path used when pressing s to save the full live plot animation. "
            "Defaults to examples/agibot_real/scripts/vr_pose_debug_<timestamp>.gif."
        ),
    )
    parser.add_argument("--plot-save-fps", type=float, default=10.0)
    parser.add_argument(
        "--plot-save-max-frames",
        type=int,
        default=600,
        help="Maximum frames in saved animation; history is evenly downsampled if needed.",
    )
    return parser.parse_args()


def _active(signals: dict[str, Any], hand: str) -> bool:
    if hand == "right":
        return bool(dict(signals.get("button_states") or {}).get("RG", False))
    return bool(dict(signals.get("left_button_states") or {}).get("LG", False))


def _parse_rot_map(rot_map: str) -> list[tuple[float, int]]:
    tokens = [token.strip().lower() for token in str(rot_map).split(",")]
    if len(tokens) != 3 or any(not token for token in tokens):
        raise ValueError(
            "--rot-map must have exactly 3 comma-separated tokens, "
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
            raise ValueError(f"Invalid --rot-map axis '{token}'. Valid axes: {valid}")
        mapping.append((sign, ROT_AXIS_INDEX[token]))
    return mapping


def _apply_rot_map(delta: np.ndarray, mapping: list[tuple[float, int]]) -> np.ndarray:
    return np.asarray(
        [sign * float(delta[index]) for sign, index in mapping],
        dtype=np.float32,
    )


def _vr_delta(signals: dict[str, Any], hand: str) -> np.ndarray:
    key = "position_delta" if hand == "right" else "left_position_delta"
    delta = np.asarray(signals.get(key, np.zeros(3)), dtype=np.float32).reshape(3)
    return delta * VR_TO_BASE_SIGN


def _vr_rot_delta(
    signals: dict[str, Any],
    hand: str,
    mapping: list[tuple[float, int]],
) -> np.ndarray:
    key = "rotation_delta" if hand == "right" else "left_rotation_delta"
    delta = np.asarray(signals.get(key, np.zeros(3)), dtype=np.float32).reshape(3)
    return _apply_rot_map(delta, mapping)


def _controller_rotation_matrix(raw_controller: Any, hand: str) -> np.ndarray | None:
    transforms, _buttons = raw_controller.vr.get_transformations_and_buttons()
    key = "r" if hand == "right" else "l"
    if not transforms or key not in transforms:
        return None
    transform = np.asarray(transforms[key], dtype=np.float64)
    if transform.shape != (4, 4):
        return None
    correction = (
        raw_controller.init_transfer_right_hand
        if hand == "right"
        else raw_controller.init_transfer_left_hand
    )
    return np.asarray(transform[:3, :3], dtype=np.float64) @ np.asarray(
        correction,
        dtype=np.float64,
    )


def _load_rotation_calibration(path: str | None) -> Rotation | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(
        payload.get("R_B_Cmap", payload["R_B_V"]),
        dtype=np.float64,
    ).reshape(3, 3)
    return Rotation.from_matrix(matrix)


def _load_tool_offset(path: str | None) -> Rotation | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = payload.get("R_C_E")
    if matrix is None:
        return None
    return Rotation.from_matrix(np.asarray(matrix, dtype=np.float64).reshape(3, 3))


def _limit_vec_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > max_norm > 0.0:
        return vector * (max_norm / norm)
    return vector


def _copy_rotation(rotation: Rotation) -> Rotation:
    return Rotation.from_quat(rotation.as_quat())


def _rotation_step_towards(
    current: Rotation,
    desired: Rotation,
    *,
    smoothing: float,
    max_step_rad: float,
) -> Rotation:
    relative = current.inv() * desired
    rotvec = relative.as_rotvec()
    step_rotvec = np.asarray(rotvec, dtype=np.float64) * float(smoothing)
    step_rotvec = _limit_vec_norm(step_rotvec, float(max_step_rad))
    return current * Rotation.from_rotvec(step_rotvec)


def _wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _nearest_xyz_euler(rotation: Rotation, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64).reshape(3)
    base = np.asarray(rotation.as_euler("xyz", degrees=False), dtype=np.float64)
    alternate = np.asarray(
        [base[0] + np.pi, np.pi - base[1], base[2] + np.pi],
        dtype=np.float64,
    )

    candidates: list[np.ndarray] = []
    for seed in (base, alternate):
        nearest = reference + _wrap_to_pi(seed - reference)
        candidates.append(np.asarray(nearest, dtype=np.float64))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.append(nearest + 2.0 * np.pi * np.array([dx, dy, dz]))

    return min(candidates, key=lambda euler: float(np.linalg.norm(euler - reference)))


def _rotation_to_command_euler(
    rotation: Rotation,
    reference: np.ndarray,
    *,
    continuous: bool,
) -> np.ndarray:
    if continuous:
        return _nearest_xyz_euler(rotation, reference)
    return np.asarray(rotation.as_euler("xyz", degrees=False), dtype=np.float64)


def _trigger_to_gripper(
    signals: dict[str, Any],
    hand: str,
    *,
    open_value: float,
    closed_value: float,
) -> float:
    key = "trigger" if hand == "right" else "left_trigger"
    trigger = float(np.clip(float(signals.get(key, 0.0) or 0.0), 0.0, 1.0))
    return float(open_value + trigger * (closed_value - open_value))


def _button_pressed(signals: dict[str, Any], hand: str, button: str) -> bool:
    button = str(button).strip()
    if hand == "right":
        return bool(dict(signals.get("button_states") or {}).get(button, False))
    return bool(dict(signals.get("left_button_states") or {}).get(button, False))


def _current_base_poses(
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


def _make_action(
    *,
    left_pos: np.ndarray,
    left_euler: np.ndarray,
    right_pos: np.ndarray,
    right_euler: np.ndarray,
) -> np.ndarray:
    return np.concatenate([left_pos, left_euler, right_pos, right_euler])


def _make_action_abs(
    raw: dict[str, Any],
    abs_action: np.ndarray,
) -> dict[str, Any]:
    return {
        "observation_timestamp": int(time.time() * 1e9),
        "head_joint_states": np.rad2deg(raw["head_rad"]).tolist(),
        "waist_joint_states": raw["waist"].tolist(),
        "arm_joint_states": raw["arm_14"].tolist(),
        "arm_cmd": [abs_action.tolist()],
    }


class _LivePosePlot:
    def __init__(self, *, hand: str, window_s: float) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.window_s = max(float(window_s), 1.0)
        self.t: list[float] = []
        self.target: list[np.ndarray] = []
        self.actual: list[np.ndarray] = []
        self.full_t: list[float] = []
        self.full_target: list[np.ndarray] = []
        self.full_actual: list[np.ndarray] = []

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 6), num=f"{hand} EE pose")
        self.fig.suptitle(f"{hand} EE pose: target vs actual")
        labels = ["x m", "y m", "z m", "roll deg", "pitch deg", "yaw deg"]
        self.target_lines = []
        self.actual_lines = []
        for axis, label in zip(self.axes.reshape(-1), labels):
            target_line, = axis.plot([], [], label="target", linewidth=1.6)
            actual_line, = axis.plot([], [], label="actual", linewidth=1.2)
            axis.set_title(label)
            axis.grid(True)
            axis.legend(loc="upper right")
            self.target_lines.append(target_line)
            self.actual_lines.append(actual_line)
        self.fig.tight_layout()
        plt.show(block=False)

    def update(
        self,
        *,
        t_s: float,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        actual_pos: np.ndarray | None,
        actual_euler: np.ndarray | None,
    ) -> None:
        target_pose = np.concatenate(
            [np.asarray(target_pos, dtype=np.float64), np.rad2deg(target_euler)]
        )
        if actual_pos is None or actual_euler is None:
            actual_pose = np.full(6, np.nan, dtype=np.float64)
        else:
            actual_pose = np.concatenate(
                [np.asarray(actual_pos, dtype=np.float64), np.rad2deg(actual_euler)]
            )

        self.t.append(float(t_s))
        self.target.append(target_pose)
        self.actual.append(actual_pose)
        self.full_t.append(float(t_s))
        self.full_target.append(target_pose.copy())
        self.full_actual.append(actual_pose.copy())
        keep_from = float(t_s) - self.window_s
        while self.t and self.t[0] < keep_from:
            self.t.pop(0)
            self.target.pop(0)
            self.actual.pop(0)

        t = np.asarray(self.t, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        actual = np.asarray(self.actual, dtype=np.float64)
        for idx, axis in enumerate(self.axes.reshape(-1)):
            self.target_lines[idx].set_data(t, target[:, idx])
            self.actual_lines[idx].set_data(t, actual[:, idx])
            axis.relim()
            axis.autoscale_view()
            if t.size:
                axis.set_xlim(max(0.0, float(t[-1]) - self.window_s), float(t[-1]) + 0.1)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def save_animation(
        self,
        path: str | Path,
        *,
        fps: float,
        max_frames: int,
    ) -> Path:
        from matplotlib.animation import FuncAnimation
        from matplotlib.animation import PillowWriter

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.full_t:
            self.fig.savefig(path.with_suffix(".png"), dpi=160)
            return path.with_suffix(".png")

        t = np.asarray(self.full_t, dtype=np.float64)
        target = np.asarray(self.full_target, dtype=np.float64)
        actual = np.asarray(self.full_actual, dtype=np.float64)
        frame_indices = np.arange(t.shape[0], dtype=np.int64)
        if max_frames > 0 and frame_indices.size > max_frames:
            frame_indices = np.linspace(
                0,
                frame_indices.size - 1,
                max_frames,
                dtype=np.int64,
            )

        fig, axes = self.plt.subplots(2, 3, figsize=(12, 6))
        fig.suptitle("EE pose: target vs actual")
        labels = ["x m", "y m", "z m", "roll deg", "pitch deg", "yaw deg"]
        target_lines = []
        actual_lines = []
        for idx, (axis, label) in enumerate(zip(axes.reshape(-1), labels)):
            target_line, = axis.plot([], [], label="target", linewidth=1.6)
            actual_line, = axis.plot([], [], label="actual", linewidth=1.2)
            axis.set_title(label)
            axis.grid(True)
            axis.legend(loc="upper right")
            y = np.concatenate([target[:, idx], actual[:, idx]])
            finite = y[np.isfinite(y)]
            if finite.size:
                y_min = float(np.min(finite))
                y_max = float(np.max(finite))
                pad = max((y_max - y_min) * 0.08, 1e-3)
                axis.set_ylim(y_min - pad, y_max + pad)
            axis.set_xlim(float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0] + 1.0))
            target_lines.append(target_line)
            actual_lines.append(actual_line)
        fig.tight_layout()

        def update(frame_idx: int):
            idx = int(frame_indices[frame_idx])
            x = t[: idx + 1]
            for axis_idx in range(6):
                target_lines[axis_idx].set_data(x, target[: idx + 1, axis_idx])
                actual_lines[axis_idx].set_data(x, actual[: idx + 1, axis_idx])
            return [*target_lines, *actual_lines]

        animation = FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=1000.0 / max(float(fps), 1e-6),
            blit=False,
        )
        animation.save(path, writer=PillowWriter(fps=max(float(fps), 1e-6)))
        self.plt.close(fig)
        return path


class _TerminalKeyReader:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self.fd: int | None = None
        self.old_settings: list[Any] | None = None

    def __enter__(self) -> "_TerminalKeyReader":
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self) -> str | None:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)


def main() -> None:
    args = _parse_args()
    period = 1.0 / max(float(args.hz), 1e-6)
    max_delta = abs(float(args.max_delta))
    max_step = abs(float(args.max_step))
    smoothing = float(np.clip(float(args.smoothing), 0.0, 1.0))
    command_deadband = abs(float(args.command_deadband))
    max_rot_delta = np.deg2rad(abs(float(args.max_rot_delta_deg)))
    max_rot_step = np.deg2rad(abs(float(args.max_rot_step_deg)))
    rot_smoothing = float(np.clip(float(args.rot_smoothing), 0.0, 1.0))
    rotation_deadband = np.deg2rad(abs(float(args.rotation_deadband_deg)))
    gripper_deadband = abs(float(args.gripper_deadband))
    rot_map = _parse_rot_map(str(args.rot_map))
    grip_local_rot_map = _parse_rot_map(str(args.grip_local_rot_map))
    r_b_v_calib = _load_rotation_calibration(args.rotation_calibration)
    r_c_e_file = _load_tool_offset(args.rotation_calibration)
    if r_b_v_calib is not None and r_c_e_file is None:
        print(
            "rotation calibration has no R_C_E; tool mode will compute a "
            "session tool offset at grip time.",
            flush=True,
        )
    reset_button = str(args.reset_button)
    if reset_button == "auto":
        reset_button = "B" if str(args.hand) == "right" else "Y"
    plotter: _LivePosePlot | None = None
    if bool(args.plot):
        try:
            plotter = _LivePosePlot(
                hand=str(args.hand),
                window_s=float(args.plot_window),
            )
        except Exception as exc:
            print(f"Live plot disabled: {type(exc).__name__}: {exc}", flush=True)

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

    def reset_to_initial_pose() -> tuple[np.ndarray, dict[str, Any]]:
        init_key = normalize_task_name_for_init_pose(str(args.task_name))
        head_action, waist_action, joint_action = get_task_initial_pose(init_key)
        print(
            f"reset to initial pose: task_name={args.task_name} init_key={init_key}",
            flush=True,
        )
        robot_node.publish_head_command(np.asarray(head_action, dtype=np.float32))
        robot_node.publish_waist_command(np.asarray(waist_action, dtype=np.float32))
        time.sleep(float(args.reset_sleep_head_waist))
        robot_node.publish_joint_command_reset(
            np.asarray(joint_action, dtype=np.float32).reshape(-1)
        )
        time.sleep(float(args.reset_sleep_arm))
        return _build_state_vec(robot_node)

    anchor_pos: np.ndarray | None = None
    anchor_rot: Rotation | None = None
    anchor_vr_rot: Rotation | None = None
    session_c_e: Rotation | None = None
    last_vr_delta: np.ndarray | None = None
    last_vr_rot_delta: np.ndarray | None = None
    last_active = False
    last_reset_pressed = False
    start_t = time.monotonic()
    last_plot_t = 0.0
    last_actual_t = 0.0
    actual_left_pos: np.ndarray | None = None
    actual_left_euler: np.ndarray | None = None
    actual_right_pos: np.ndarray | None = None
    actual_right_euler: np.ndarray | None = None

    try:
        state_vec, raw = _build_state_vec(robot_node)
        left_pos, left_euler, right_pos, right_euler = _current_base_poses(
            retargeter,
            state_vec,
        )
        target_left_pos = left_pos.copy()
        target_right_pos = right_pos.copy()
        target_left_euler = left_euler.copy()
        target_right_euler = right_euler.copy()
        target_left_rot = Rotation.from_euler("xyz", target_left_euler)
        target_right_rot = Rotation.from_euler("xyz", target_right_euler)
        commanded_left_pos = target_left_pos.copy()
        commanded_right_pos = target_right_pos.copy()
        commanded_left_euler = target_left_euler.copy()
        commanded_right_euler = target_right_euler.copy()

        hand_action = np.array(
            [
                float(raw["joint_state_16"][7]),
                float(raw["joint_state_16"][15]),
            ],
            dtype=np.float32,
        )
        current_hand_idx = 1 if str(args.hand) == "right" else 0
        commanded_gripper = float(hand_action[current_hand_idx])

        print("=" * 88, flush=True)
        print(
            f"VR base pose teleop hand={args.hand} hz={float(args.hz):.2f} "
            f"trajectory_time={float(args.trajectory_time):.3f}s "
            f"execute={bool(args.execute)} max_step={max_step:.3f} "
            f"smoothing={smoothing:.2f} max_rot_step={args.max_rot_step_deg:.1f}deg "
            f"rot_smoothing={rot_smoothing:.2f} rot_map={args.rot_map} "
            f"so3_order={args.so3_order} "
            f"rotation_calibration={args.rotation_calibration or 'none'} "
            f"calibrated_rotation_mode={args.calibrated_rotation_mode} "
            f"continuous_euler={bool(args.continuous_euler)} "
            f"gripper_open={args.gripper_open:.1f} "
            f"gripper_closed={args.gripper_closed:.1f} "
            f"reset_button={reset_button} task={args.task_name}",
            flush=True,
        )
        print("[initial base_frame]", flush=True)
        print(_format_pose("left ", left_pos, np.rad2deg(left_euler)), flush=True)
        print(_format_pose("right", right_pos, np.rad2deg(right_euler)), flush=True)
        print(
            "Hold RG for right hand or LG for left hand. "
            f"Trigger controls gripper even when grip is released. "
            f"Press {reset_button} or keyboard r to reset. "
            "Press keyboard s to save plot and stop. Ctrl+C to stop.",
            flush=True,
        )

        with QuestVRClient(
            scaling_factor=float(args.scaling_factor),
            motion_mode="xyzrxryrz",
            control_freq=int(args.control_freq),
            enable_visualization=False,
            controller_mode=str(args.hand),
            coordinate_mapping=str(args.coordinate_mapping),
        ) as vr_client, _TerminalKeyReader() as key_reader:
            while True:
                loop_t0 = time.monotonic()
                keyboard_reset = False
                key = key_reader.read_key()
                if key == "s":
                    if plotter is not None:
                        save_path = args.plot_save_path
                        if save_path is None:
                            stamp = time.strftime("%Y%m%d_%H%M%S")
                            save_path = SCRIPT_DIR / f"vr_pose_debug_{stamp}.gif"
                        saved = plotter.save_animation(
                            save_path,
                            fps=float(args.plot_save_fps),
                            max_frames=int(args.plot_save_max_frames),
                        )
                        print(f"Saved live plot animation to {saved}", flush=True)
                    else:
                        print("No live plot is enabled; nothing to save.", flush=True)
                    print("Stopped by keyboard s.", flush=True)
                    break
                if key == "r":
                    keyboard_reset = True

                snapshot = vr_client.snapshot()
                signals = snapshot.signals
                is_active = _active(signals, str(args.hand))
                reset_pressed = _button_pressed(signals, str(args.hand), reset_button)

                if keyboard_reset or (reset_pressed and not last_reset_pressed):
                    if bool(args.execute):
                        state_vec, raw = reset_to_initial_pose()
                    else:
                        print("DRY RUN: reset skipped without --execute", flush=True)
                        state_vec, raw = _build_state_vec(robot_node)
                    left_pos, left_euler, right_pos, right_euler = _current_base_poses(
                        retargeter,
                        state_vec,
                    )
                    target_left_pos = left_pos.copy()
                    target_right_pos = right_pos.copy()
                    target_left_euler = left_euler.copy()
                    target_right_euler = right_euler.copy()
                    target_left_rot = Rotation.from_euler("xyz", target_left_euler)
                    target_right_rot = Rotation.from_euler("xyz", target_right_euler)
                    commanded_left_pos = target_left_pos.copy()
                    commanded_right_pos = target_right_pos.copy()
                    commanded_left_euler = target_left_euler.copy()
                    commanded_right_euler = target_right_euler.copy()
                    hand_action[:] = np.array(
                        [
                            float(raw["joint_state_16"][7]),
                            float(raw["joint_state_16"][15]),
                        ],
                        dtype=np.float32,
                    )
                    commanded_gripper = float(hand_action[current_hand_idx])
                    anchor_pos = None
                    anchor_rot = None
                    anchor_vr_rot = None
                    session_c_e = None
                    last_vr_delta = None
                    last_vr_rot_delta = None
                    last_active = False
                    print("[after reset base_frame]", flush=True)
                    print(
                        _format_pose(
                            "left ",
                            target_left_pos,
                            np.rad2deg(target_left_euler),
                        ),
                        flush=True,
                    )
                    print(
                        _format_pose(
                            "right",
                            target_right_pos,
                            np.rad2deg(target_right_euler),
                        ),
                        flush=True,
                    )
                    last_reset_pressed = bool(reset_pressed)
                    sleep_s = period - (time.monotonic() - loop_t0)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue
                last_reset_pressed = bool(reset_pressed)

                if is_active and not last_active:
                    if str(args.hand) == "right":
                        anchor_pos = target_right_pos.copy()
                        anchor_rot = _copy_rotation(target_right_rot)
                    else:
                        anchor_pos = target_left_pos.copy()
                        anchor_rot = _copy_rotation(target_left_rot)
                    if r_b_v_calib is not None:
                        current_vr_matrix = _controller_rotation_matrix(
                            vr_client.raw_controller,
                            str(args.hand),
                        )
                        if current_vr_matrix is None:
                            anchor_vr_rot = None
                            print(
                                "pose teleop active, but no raw VR rotation yet",
                                flush=True,
                            )
                        else:
                            anchor_vr_rot = Rotation.from_matrix(current_vr_matrix)
                            if str(args.calibrated_rotation_mode) == "tool":
                                robot_start_rot = (
                                    target_right_rot
                                    if str(args.hand) == "right"
                                    else target_left_rot
                                )
                                session_c_e = (
                                    r_b_v_calib * anchor_vr_rot
                                ).inv() * robot_start_rot
                            else:
                                session_c_e = None
                    last_vr_delta = np.zeros(3, dtype=np.float32)
                    last_vr_rot_delta = np.zeros(3, dtype=np.float32)
                    print("pose teleop active", flush=True)
                elif not is_active and last_active:
                    anchor_pos = None
                    anchor_rot = None
                    anchor_vr_rot = None
                    session_c_e = None
                    last_vr_delta = None
                    last_vr_rot_delta = None
                    print("pose teleop inactive", flush=True)

                hand_action[current_hand_idx] = _trigger_to_gripper(
                    signals,
                    str(args.hand),
                    open_value=float(args.gripper_open),
                    closed_value=float(args.gripper_closed),
                )
                gripper_delta = abs(
                    float(hand_action[current_hand_idx]) - float(commanded_gripper)
                )
                pos_command_delta = 0.0
                rot_command_delta = 0.0

                if is_active and anchor_pos is not None and anchor_rot is not None:
                    delta = _limit_vec_norm(
                        _vr_delta(signals, str(args.hand)),
                        max_delta,
                    )
                    if r_b_v_calib is not None:
                        current_vr_matrix = _controller_rotation_matrix(
                            vr_client.raw_controller,
                            str(args.hand),
                        )
                        if current_vr_matrix is None or anchor_vr_rot is None:
                            rot_delta = np.zeros(3, dtype=np.float64)
                            desired_rot = _copy_rotation(anchor_rot)
                        else:
                            current_vr_rot = Rotation.from_matrix(current_vr_matrix)
                            if str(args.calibrated_rotation_mode) == "direct":
                                raw_desired_rot = r_b_v_calib * current_vr_rot
                                relative_to_current = (
                                    anchor_rot.inv() * raw_desired_rot
                                )
                                rot_delta = _limit_vec_norm(
                                    relative_to_current.as_rotvec(),
                                    max_rot_delta,
                                )
                                desired_rot = anchor_rot * Rotation.from_rotvec(
                                    rot_delta,
                                )
                            elif str(args.calibrated_rotation_mode) == "tool":
                                r_c_e = r_c_e_file or session_c_e
                                if r_c_e is None:
                                    desired_rot = _copy_rotation(anchor_rot)
                                    rot_delta = np.zeros(3, dtype=np.float64)
                                else:
                                    raw_desired_rot = (
                                        r_b_v_calib * current_vr_rot * r_c_e
                                    )
                                    relative_to_current = (
                                        anchor_rot.inv() * raw_desired_rot
                                    )
                                    rot_delta = _limit_vec_norm(
                                        relative_to_current.as_rotvec(),
                                        max_rot_delta,
                                    )
                                    desired_rot = anchor_rot * Rotation.from_rotvec(
                                        rot_delta,
                                    )
                            elif str(args.calibrated_rotation_mode) == "grip-local":
                                delta_c = anchor_vr_rot.inv() * current_vr_rot
                                rot_delta = _limit_vec_norm(
                                    _apply_rot_map(
                                        np.asarray(
                                            delta_c.as_rotvec(),
                                            dtype=np.float64,
                                        ),
                                        grip_local_rot_map,
                                    ),
                                    max_rot_delta,
                                )
                                desired_rot = anchor_rot * Rotation.from_rotvec(
                                    rot_delta,
                                )
                            else:
                                delta_v = current_vr_rot * anchor_vr_rot.inv()
                                delta_b = r_b_v_calib * delta_v * r_b_v_calib.inv()
                                rot_delta = _limit_vec_norm(
                                    delta_b.as_rotvec(),
                                    max_rot_delta,
                                )
                                delta_b = Rotation.from_rotvec(rot_delta)
                                if str(args.so3_order) == "local":
                                    desired_rot = anchor_rot * delta_b
                                else:
                                    desired_rot = delta_b * anchor_rot
                    else:
                        rot_delta = _limit_vec_norm(
                            _vr_rot_delta(signals, str(args.hand), rot_map),
                            max_rot_delta,
                        )

                    if str(args.control_mode) == "incremental":
                        if last_vr_delta is None:
                            last_vr_delta = np.zeros(3, dtype=np.float32)
                        if last_vr_rot_delta is None:
                            last_vr_rot_delta = np.zeros(3, dtype=np.float32)
                        delta_step = np.asarray(delta - last_vr_delta, dtype=np.float64)
                        rot_delta_step = np.asarray(
                            rot_delta - last_vr_rot_delta,
                            dtype=np.float64,
                        )
                        last_vr_delta = delta.copy()
                        last_vr_rot_delta = rot_delta.copy()
                        if str(args.hand) == "right":
                            desired_pos = target_right_pos + delta_step
                            if r_b_v_calib is None:
                                step_rot = Rotation.from_rotvec(rot_delta_step)
                                if str(args.so3_order) == "local":
                                    desired_rot = target_right_rot * step_rot
                                else:
                                    desired_rot = step_rot * target_right_rot
                        else:
                            desired_pos = target_left_pos + delta_step
                            if r_b_v_calib is None:
                                step_rot = Rotation.from_rotvec(rot_delta_step)
                                if str(args.so3_order) == "local":
                                    desired_rot = target_left_rot * step_rot
                                else:
                                    desired_rot = step_rot * target_left_rot
                    else:
                        desired_pos = anchor_pos + delta
                        if r_b_v_calib is None:
                            delta_rot = Rotation.from_rotvec(rot_delta)
                            if str(args.so3_order) == "local":
                                desired_rot = anchor_rot * delta_rot
                            else:
                                desired_rot = delta_rot * anchor_rot
                    if str(args.hand) == "right":
                        filtered_pos = (
                            target_right_pos
                            + smoothing * (desired_pos - target_right_pos)
                        )
                        step = _limit_vec_norm(filtered_pos - target_right_pos, max_step)
                        target_right_pos = target_right_pos + step

                        target_right_rot = _rotation_step_towards(
                            target_right_rot,
                            desired_rot,
                            smoothing=rot_smoothing,
                            max_step_rad=max_rot_step,
                        )
                        target_right_euler = _rotation_to_command_euler(
                            target_right_rot,
                            target_right_euler,
                            continuous=bool(args.continuous_euler),
                        )
                        pos_command_delta = float(
                            np.linalg.norm(target_right_pos - commanded_right_pos)
                        )
                        rot_command_delta = float(
                            (Rotation.from_euler("xyz", commanded_right_euler).inv() * target_right_rot).magnitude()
                        )
                        target_pos = target_right_pos
                        target_euler = target_right_euler
                    else:
                        filtered_pos = (
                            target_left_pos
                            + smoothing * (desired_pos - target_left_pos)
                        )
                        step = _limit_vec_norm(filtered_pos - target_left_pos, max_step)
                        target_left_pos = target_left_pos + step

                        target_left_rot = _rotation_step_towards(
                            target_left_rot,
                            desired_rot,
                            smoothing=rot_smoothing,
                            max_step_rad=max_rot_step,
                        )
                        target_left_euler = _rotation_to_command_euler(
                            target_left_rot,
                            target_left_euler,
                            continuous=bool(args.continuous_euler),
                        )
                        pos_command_delta = float(
                            np.linalg.norm(target_left_pos - commanded_left_pos)
                        )
                        rot_command_delta = float(
                            (Rotation.from_euler("xyz", commanded_left_euler).inv() * target_left_rot).magnitude()
                        )
                        target_pos = target_left_pos
                        target_euler = target_left_euler

                    print(
                        f"{args.hand} xyz=({target_pos[0]:+.3f},"
                        f"{target_pos[1]:+.3f},{target_pos[2]:+.3f}) "
                        f"rpy_deg=({np.rad2deg(target_euler[0]):+.1f},"
                        f"{np.rad2deg(target_euler[1]):+.1f},"
                        f"{np.rad2deg(target_euler[2]):+.1f}) "
                        f"gripper={hand_action[current_hand_idx]:.1f}"
                        + (
                            f" delta_xyz=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}) "
                            f"delta_rpy_deg=({np.rad2deg(rot_delta[0]):+.1f},"
                            f"{np.rad2deg(rot_delta[1]):+.1f},"
                            f"{np.rad2deg(rot_delta[2]):+.1f})"
                            if bool(args.print_delta)
                            else ""
                        ),
                        flush=True,
                    )

                should_send = (
                    pos_command_delta >= command_deadband
                    or rot_command_delta >= rotation_deadband
                    or gripper_delta >= gripper_deadband
                )
                if bool(args.execute) and should_send:
                    abs_action = _make_action(
                        left_pos=target_left_pos,
                        left_euler=target_left_euler,
                        right_pos=target_right_pos,
                        right_euler=target_right_euler,
                    )
                    robot_node.publish_abs_pose_command_and_hand(
                        _make_action_abs(raw, abs_action),
                        hand_action,
                        trajectory_reference_time=float(args.trajectory_time),
                    )
                    if str(args.hand) == "right":
                        commanded_right_pos = target_right_pos.copy()
                        commanded_right_euler = target_right_euler.copy()
                    else:
                        commanded_left_pos = target_left_pos.copy()
                        commanded_left_euler = target_left_euler.copy()
                    commanded_gripper = float(hand_action[current_hand_idx])

                now = time.monotonic()
                if plotter is not None and now - last_plot_t >= 1.0 / max(
                    float(args.plot_rate),
                    1e-6,
                ):
                    if now - last_actual_t >= 1.0 / max(float(args.actual_rate), 1e-6):
                        try:
                            prev_actual_left_euler = (
                                actual_left_euler.copy()
                                if actual_left_euler is not None
                                else None
                            )
                            prev_actual_right_euler = (
                                actual_right_euler.copy()
                                if actual_right_euler is not None
                                else None
                            )
                            state_vec, _actual_raw = _build_state_vec(robot_node)
                            (
                                actual_left_pos,
                                actual_left_euler,
                                actual_right_pos,
                                actual_right_euler,
                            ) = _current_base_poses(retargeter, state_vec)
                            if (
                                bool(args.continuous_euler)
                                and prev_actual_left_euler is not None
                            ):
                                actual_left_euler = _nearest_xyz_euler(
                                    Rotation.from_euler("xyz", actual_left_euler),
                                    prev_actual_left_euler,
                                )
                            if (
                                bool(args.continuous_euler)
                                and prev_actual_right_euler is not None
                            ):
                                actual_right_euler = _nearest_xyz_euler(
                                    Rotation.from_euler("xyz", actual_right_euler),
                                    prev_actual_right_euler,
                                )
                            last_actual_t = now
                        except Exception as exc:
                            print(
                                f"actual pose read failed: {type(exc).__name__}: {exc}",
                                flush=True,
                            )

                    if str(args.hand) == "right":
                        plotter.update(
                            t_s=now - start_t,
                            target_pos=target_right_pos,
                            target_euler=target_right_euler,
                            actual_pos=actual_right_pos,
                            actual_euler=actual_right_euler,
                        )
                    else:
                        plotter.update(
                            t_s=now - start_t,
                            target_pos=target_left_pos,
                            target_euler=target_left_euler,
                            actual_pos=actual_left_pos,
                            actual_euler=actual_left_euler,
                        )
                    last_plot_t = now

                last_active = bool(is_active)
                sleep_s = period - (time.monotonic() - loop_t0)
                if sleep_s > 0:
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
