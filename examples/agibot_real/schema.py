"""Shared AgiBot schema helpers for runtime and training recipes."""
from __future__ import annotations

import re
from typing import Iterable
from typing import Tuple

AGIBOT_IMAGE_SLOT_KEYS = ("image_rgb_0", "image_rgb_1", "image_rgb_2")
AGIBOT_DEFAULT_IMAGE_KEYS = AGIBOT_IMAGE_SLOT_KEYS
AGIBOT_CONTROLLER_STATES = (
    "WAIT_READY",
    "RUNNING",
    "PAUSED",
    "RESETTING",
    "EPISODE_DONE",
)
AGIBOT_TERMINAL_SIGNALS = (
    "success",
    "fail",
    "reset",
    "timeout",
    "hook",
)
AGIBOT_IMAGE_VIEW_TO_SLOT = {
    "image": "image_rgb_0",
    "head_image": "image_rgb_0",
    "front_image": "image_rgb_0",
    "image_rgb_0": "image_rgb_0",
    "wrist_left_image": "image_rgb_1",
    "left_wrist_image": "image_rgb_1",
    "image_rgb_1": "image_rgb_1",
    "wrist_right_image": "image_rgb_2",
    "right_wrist_image": "image_rgb_2",
    "image_rgb_2": "image_rgb_2",
}


def resolve_agibot_image_key(image_key: str) -> str:
    key = str(image_key)
    resolved = AGIBOT_IMAGE_VIEW_TO_SLOT.get(key, None)
    if resolved is None:
        raise KeyError(
            f"Unsupported AgiBot image key {key!r}. "
            f"Expected one of {sorted(AGIBOT_IMAGE_VIEW_TO_SLOT)}"
        )
    return resolved


def resolve_agibot_image_keys(image_keys: Iterable[str]) -> Tuple[str, ...]:
    resolved_keys = []
    seen = set()
    for image_key in image_keys:
        resolved = resolve_agibot_image_key(str(image_key))
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_keys.append(resolved)
    if not resolved_keys:
        raise ValueError("At least one AgiBot image key is required")
    return tuple(resolved_keys)


def sanitize_agibot_task_name(task_name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(task_name).strip()).strip("_").lower()
    return token or "default"


def build_agibot_task_key(task_name: str) -> str:
    return f"agibot_real_{sanitize_agibot_task_name(task_name)}"
