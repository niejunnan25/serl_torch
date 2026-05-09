"""Print Quest VR controller signal snapshots for bring-up."""

from __future__ import annotations

import argparse
import time
from typing import Any

from serl_torch.examples.agibot_real.intervention import QuestVRClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Quest controller delta signals without commanding robot.",
    )
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--scaling-factor", type=float, default=0.5)
    parser.add_argument("--control-freq", type=int, default=30)
    parser.add_argument("--controller-mode", default="both")
    parser.add_argument("--coordinate-mapping", default="sim")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--raw-buttons",
        action="store_true",
        help="Also print OculusReader raw parsed button dict.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _flatten_button_state(signals: dict[str, Any]) -> dict[str, Any]:
    right_buttons = dict(signals.get("button_states") or {})
    left_buttons = dict(signals.get("left_button_states") or {})
    return {
        "A": bool(right_buttons.get("A", False)),
        "B": bool(right_buttons.get("B", False)),
        "RG": bool(right_buttons.get("RG", False)),
        "RTr": bool(right_buttons.get("RTr", False)),
        "right_trigger": float(signals.get("trigger", 0.0) or 0.0),
        "right_grip": float(signals.get("grip", 0.0) or 0.0),
        "right_joystick": right_buttons.get("rightJS", signals.get("joystick")),
        "X": bool(left_buttons.get("X", False)),
        "Y": bool(left_buttons.get("Y", False)),
        "LG": bool(left_buttons.get("LG", False)),
        "LTr": bool(left_buttons.get("LTr", False)),
        "left_trigger": float(signals.get("left_trigger", 0.0) or 0.0),
        "left_grip": float(signals.get("left_grip", 0.0) or 0.0),
        "left_joystick": left_buttons.get("leftJS", signals.get("left_joystick")),
    }


def _changed_events(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    if previous is None:
        return []
    events: list[str] = []
    for key, value in current.items():
        old = previous.get(key)
        if isinstance(value, bool):
            if bool(old) != bool(value):
                events.append(f"{key}={'pressed' if value else 'released'}")
            continue
        if isinstance(value, float):
            old_float = 0.0 if old is None else float(old)
            if abs(float(value) - old_float) >= 0.05:
                events.append(f"{key}={float(value):.2f}")
            continue
        if old != value:
            events.append(f"{key}={value}")
    return events


def main() -> None:
    args = _parse_args()
    period = 1.0 / max(float(args.hz), 1e-6)
    start_time = time.time()
    previous_buttons: dict[str, Any] | None = None

    with QuestVRClient(
        scaling_factor=float(args.scaling_factor),
        control_freq=int(args.control_freq),
        enable_visualization=bool(args.visualize),
        controller_mode=str(args.controller_mode),
        coordinate_mapping=str(args.coordinate_mapping),
    ) as client:
        while True:
            snapshot = client.snapshot()
            signals = snapshot.signals
            buttons = _flatten_button_state(signals)
            events = _changed_events(previous_buttons, buttons)
            previous_buttons = dict(buttons)
            payload = {
                "age_sec": round(snapshot.age_sec, 4),
                "recent": snapshot.recent,
                "events": events,
                "buttons": _jsonable(buttons),
                "right_pos_delta": _jsonable(signals.get("position_delta")),
                "right_rot_delta": _jsonable(signals.get("rotation_delta")),
                "left_pos_delta": _jsonable(signals.get("left_position_delta")),
                "left_rot_delta": _jsonable(signals.get("left_rotation_delta")),
            }
            if bool(args.raw_buttons):
                raw_vr = getattr(client.raw_controller, "vr", None)
                payload["raw_buttons"] = _jsonable(
                    dict(getattr(raw_vr, "last_buttons", {}) or {})
                )
            print(payload)
            if float(args.duration_sec) > 0.0:
                if time.time() - start_time >= float(args.duration_sec):
                    break
            time.sleep(period)


if __name__ == "__main__":
    main()
