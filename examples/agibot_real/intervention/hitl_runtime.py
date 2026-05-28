"""Runtime wrapper for human-in-the-loop AgiBot actor control."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .vr_camera_action import VRCameraActionConfig
from .vr_camera_action import VRCameraActionController
from .vr_camera_action import build_state_vec_from_robot_node
from .vr_client import QuestVRClient


@dataclass(slots=True)
class HitlActionChunk:
    actions: np.ndarray
    infos: list[dict[str, Any]]


class QuestVRHitlRuntime:
    """Generate camera_position action chunks from Quest VR when active."""

    def __init__(
        self,
        *,
        env: Any,
        cfg: Any,
        logger: Any,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.logger = logger
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.hand = str(getattr(cfg, "hand", "right"))
        self.vr_chunk_horizon = max(1, int(getattr(cfg, "vr_chunk_horizon", 1)))
        self._vr_client: QuestVRClient | None = None
        self._controller: VRCameraActionController | None = None
        self._last_active = False

        if self.enabled:
            self._vr_client = QuestVRClient(
                scaling_factor=float(getattr(cfg, "scaling_factor", 0.5)),
                motion_mode="xyzrxryrz",
                control_freq=int(getattr(cfg, "control_freq", 30)),
                enable_visualization=False,
                controller_mode=self.hand,
                coordinate_mapping=str(getattr(cfg, "coordinate_mapping", "sim")),
            )
            self._vr_client.__enter__()
            state_vec, raw = build_state_vec_from_robot_node(env.robot_node)
            initial_grippers = np.asarray(
                [raw["joint_state_16"][7], raw["joint_state_16"][15]],
                dtype=np.float32,
            )
            self._controller = VRCameraActionController(
                retargeter=env.retargeter,
                initial_state_vec=state_vec,
                initial_grippers=initial_grippers,
                config=VRCameraActionConfig(
                    hand=self.hand,
                    max_delta=float(getattr(cfg, "max_delta", 0.10)),
                    max_step=float(getattr(cfg, "max_step", 0.006)),
                    smoothing=float(getattr(cfg, "smoothing", 0.40)),
                    command_deadband=float(getattr(cfg, "command_deadband", 0.001)),
                    max_rot_delta_deg=float(
                        getattr(cfg, "max_rot_delta_deg", 35.0)
                    ),
                    max_rot_step_deg=float(getattr(cfg, "max_rot_step_deg", 2.0)),
                    rot_smoothing=float(getattr(cfg, "rot_smoothing", 0.12)),
                    rotation_deadband_deg=float(
                        getattr(cfg, "rotation_deadband_deg", 0.8)
                    ),
                    rot_map=str(getattr(cfg, "rot_map", "-ry,-rz,rx")),
                    gripper_open=float(getattr(cfg, "gripper_open", 0.0)),
                    gripper_closed=float(getattr(cfg, "gripper_closed", 120.0)),
                    gripper_deadband=float(getattr(cfg, "gripper_deadband", 0.5)),
                ),
            )
            self.logger.info(
                "HITL Quest VR enabled: hand=%s vr_chunk_horizon=%s",
                self.hand,
                int(self.vr_chunk_horizon),
            )

    def close(self) -> None:
        if self._vr_client is not None:
            self._vr_client.__exit__(None, None, None)
            self._vr_client = None

    def _snapshot_active(self, signals: dict[str, Any]) -> bool:
        right_active = bool(dict(signals.get("button_states") or {}).get("RG", False))
        left_active = bool(
            dict(signals.get("left_button_states") or {}).get("LG", False)
        )
        if self.hand == "right":
            return right_active
        if self.hand == "left":
            return left_active
        return bool(right_active or left_active)

    def poll_action_chunk(self) -> HitlActionChunk | None:
        if (not self.enabled) or self._vr_client is None or self._controller is None:
            return None

        snapshot = self._vr_client.snapshot()
        if not self._snapshot_active(snapshot.signals):
            if self._last_active:
                self.logger.info("HITL VR inactive; returning control to RL")
            self._last_active = False
            return None

        if not self._last_active:
            self.logger.info("HITL VR active; using Quest VR action source")
        self._last_active = True

        actions: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        period = 1.0 / max(float(getattr(self.env, "hz", 20.0)), 1.0)
        for step_idx in range(self.vr_chunk_horizon):
            loop_t0 = time.monotonic()
            state_vec, _raw = build_state_vec_from_robot_node(self.env.robot_node)
            signals = snapshot.signals if step_idx == 0 else self._vr_client.snapshot().signals
            result = self._controller.update(
                signals=signals,
                state_vec=state_vec,
                raw_controller=self._vr_client.raw_controller,
            )
            actions.append(np.asarray(result.camera_action, dtype=np.float32).reshape(14))
            info = dict(result.info)
            info["hitl_chunk_step"] = int(step_idx)
            info["hitl_vr_chunk_horizon"] = int(self.vr_chunk_horizon)
            infos.append(info)
            sleep_s = period - (time.monotonic() - loop_t0)
            if sleep_s > 0.0 and step_idx < self.vr_chunk_horizon - 1:
                time.sleep(sleep_s)

        return HitlActionChunk(
            actions=np.asarray(actions, dtype=np.float32).reshape(-1, 14),
            infos=infos,
        )


__all__ = ["HitlActionChunk", "QuestVRHitlRuntime"]
