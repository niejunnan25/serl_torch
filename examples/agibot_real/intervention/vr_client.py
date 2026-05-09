"""Thin adapter around the vendored Quest controller input stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


_QUEST_ROOT = Path(__file__).resolve().parent / "quest"
_QUEST_TELEOP_ROOT = _QUEST_ROOT / "Teleoperation"
_QUEST_OCULUS_ROOT = _QUEST_ROOT / "oculus_reader"

for _path in (_QUEST_ROOT, _QUEST_TELEOP_ROOT, _QUEST_OCULUS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@dataclass(frozen=True)
class VRSignalSnapshot:
    """Stable copy of one Quest controller signal frame."""

    timestamp: float
    signals: dict[str, Any]

    @property
    def age_sec(self) -> float:
        return max(0.0, time.time() - float(self.timestamp))

    @property
    def recent(self) -> bool:
        return self.age_sec < 0.2


def _copy_signal_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {str(k): _copy_signal_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_signal_value(v) for v in value)
    return value


class QuestVRClient:
    """Lifecycle wrapper for ``VRTeleoperationController``.

    This keeps the rest of ``agibot_real`` independent from the copied HITL
    directory layout. The controller itself is intentionally still the original
    Quest implementation.
    """

    def __init__(
        self,
        *,
        scaling_factor: float = 0.5,
        motion_mode: str = "xyzrxryrz",
        control_freq: int = 30,
        enable_visualization: bool = False,
        controller_mode: str = "both",
        coordinate_mapping: str = "sim",
    ) -> None:
        from Teleoperation.vr_teleoperation_controller import (
            VRTeleoperationController,
        )

        self._controller = VRTeleoperationController(
            scaling_factor=float(scaling_factor),
            motion_mode=str(motion_mode),
            control_freq=int(control_freq),
            enable_visualization=bool(enable_visualization),
            controller_mode=str(controller_mode),
        )
        self._controller.set_coordinate_mapping(str(coordinate_mapping))
        self._started = False

    @property
    def raw_controller(self) -> Any:
        return self._controller

    def start(self) -> None:
        if self._started:
            return
        self._controller.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._controller.stop()
        self._started = False

    def snapshot(self) -> VRSignalSnapshot:
        signals = {
            str(k): _copy_signal_value(v)
            for k, v in dict(self._controller.delta_control_signals).items()
        }
        timestamp_value = signals.get("timestamp", time.time())
        try:
            timestamp = float(timestamp_value)
        except (TypeError, ValueError):
            timestamp = time.time()
        return VRSignalSnapshot(timestamp=timestamp, signals=signals)

    def __enter__(self) -> "QuestVRClient":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


__all__ = ["QuestVRClient", "VRSignalSnapshot"]
