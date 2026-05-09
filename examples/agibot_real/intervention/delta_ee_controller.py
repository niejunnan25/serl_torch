"""Pure-software Quest delta-EE to AgiBot 14D action adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


_DEFAULT_LEFT_POS = np.array([0.56, 0.25, 0.03], dtype=np.float32)
_DEFAULT_RIGHT_POS = np.array([0.56, -0.25, 0.03], dtype=np.float32)
_DEFAULT_ROTATION = Rotation.from_quat([0.0, -0.709, 0.0, 0.705])
_POSITION_DIRECTION_SIGN = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
_ROTATION_DIRECTION_SIGN = np.array([-1.0, -1.0, -1.0], dtype=np.float32)


@dataclass
class EETargetState:
    pos: np.ndarray
    rot: Rotation
    gripper: float = 100.0

    def copy(self) -> "EETargetState":
        return EETargetState(
            pos=np.asarray(self.pos, dtype=np.float32).copy(),
            rot=Rotation.from_quat(self.rot.as_quat()),
            gripper=float(self.gripper),
        )


class DeltaEETeleopController:
    """Maintains absolute EE targets from Quest relative motion signals."""

    def __init__(self) -> None:
        self.left = EETargetState(pos=_DEFAULT_LEFT_POS.copy(), rot=_DEFAULT_ROTATION)
        self.right = EETargetState(pos=_DEFAULT_RIGHT_POS.copy(), rot=_DEFAULT_ROTATION)
        self._left_anchor: EETargetState | None = None
        self._right_anchor: EETargetState | None = None
        self._last_left_active = False
        self._last_right_active = False

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def _trigger_to_gripper(value: Any) -> float:
        trigger = float(np.clip(float(value or 0.0), 0.0, 1.0))
        return float((1.0 - trigger) * 100.0)

    @staticmethod
    def _as_vec3(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape(3)

    @staticmethod
    def _correct_position_direction(value: Any) -> np.ndarray:
        return DeltaEETeleopController._as_vec3(value) * _POSITION_DIRECTION_SIGN

    @staticmethod
    def _correct_rotation_direction(value: Any) -> np.ndarray:
        return DeltaEETeleopController._as_vec3(value) * _ROTATION_DIRECTION_SIGN

    @staticmethod
    def _active(buttons: dict[str, Any], key: str) -> bool:
        return bool(dict(buttons or {}).get(key, False))

    def _update_hand(
        self,
        *,
        current: EETargetState,
        anchor: EETargetState | None,
        active: bool,
        last_active: bool,
        pos_delta: Any,
        rot_delta: Any,
        gripper: float,
    ) -> tuple[EETargetState, EETargetState | None]:
        current.gripper = float(gripper)
        if active and not last_active:
            anchor = current.copy()
        if active and anchor is not None:
            current.pos = anchor.pos + self._correct_position_direction(pos_delta)
            current.rot = anchor.rot * Rotation.from_rotvec(
                self._correct_rotation_direction(rot_delta)
            )
        if not active:
            anchor = None
        return current, anchor

    def update(self, signals: dict[str, Any]) -> np.ndarray:
        right_buttons = dict(signals.get("button_states") or {})
        left_buttons = dict(signals.get("left_button_states") or {})
        right_active = self._active(right_buttons, "RG")
        left_active = self._active(left_buttons, "LG")

        self.right, self._right_anchor = self._update_hand(
            current=self.right,
            anchor=self._right_anchor,
            active=right_active,
            last_active=self._last_right_active,
            pos_delta=signals.get("position_delta", np.zeros(3, dtype=np.float32)),
            rot_delta=signals.get("rotation_delta", np.zeros(3, dtype=np.float32)),
            gripper=self._trigger_to_gripper(signals.get("trigger", 0.0)),
        )
        self.left, self._left_anchor = self._update_hand(
            current=self.left,
            anchor=self._left_anchor,
            active=left_active,
            last_active=self._last_left_active,
            pos_delta=signals.get("left_position_delta", np.zeros(3, dtype=np.float32)),
            rot_delta=signals.get("left_rotation_delta", np.zeros(3, dtype=np.float32)),
            gripper=self._trigger_to_gripper(signals.get("left_trigger", 0.0)),
        )
        self._last_right_active = bool(right_active)
        self._last_left_active = bool(left_active)
        return self.action14()

    def action14(self) -> np.ndarray:
        left_rotvec = self.left.rot.as_rotvec().astype(np.float32)
        right_rotvec = self.right.rot.as_rotvec().astype(np.float32)
        return np.concatenate(
            [
                self.left.pos.astype(np.float32),
                left_rotvec,
                np.array([self.left.gripper], dtype=np.float32),
                self.right.pos.astype(np.float32),
                right_rotvec,
                np.array([self.right.gripper], dtype=np.float32),
            ]
        ).astype(np.float32)

    def format_status(self) -> str:
        left_rpy = self.left.rot.as_euler("xyz", degrees=True)
        right_rpy = self.right.rot.as_euler("xyz", degrees=True)
        left_active = "on" if self._last_left_active else "off"
        right_active = "on" if self._last_right_active else "off"
        return (
            "L "
            f"active={left_active} "
            f"xyz=({self.left.pos[0]:+.3f},{self.left.pos[1]:+.3f},{self.left.pos[2]:+.3f}) "
            f"rpy=({left_rpy[0]:+.1f},{left_rpy[1]:+.1f},{left_rpy[2]:+.1f}) "
            f"grip={self.left.gripper:.1f} | "
            "R "
            f"active={right_active} "
            f"xyz=({self.right.pos[0]:+.3f},{self.right.pos[1]:+.3f},{self.right.pos[2]:+.3f}) "
            f"rpy=({right_rpy[0]:+.1f},{right_rpy[1]:+.1f},{right_rpy[2]:+.1f}) "
            f"grip={self.right.gripper:.1f}"
        )


__all__ = ["DeltaEETeleopController", "EETargetState"]
