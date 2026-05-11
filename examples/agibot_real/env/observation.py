"""Observation helpers for the canonical AgiBot residual training flow."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .schema import resolve_agibot_image_keys

AGIBOT_STATE_DIM = 14
AGIBOT_ARM_STATE_DIM = 7
AGIBOT_LEFT_ARM_STATE_SLICE = slice(0, 7)
AGIBOT_RIGHT_ARM_STATE_SLICE = slice(7, 14)
AGIBOT_JOYRA_STATE_DIM = 18
RESIDUAL_IMAGE_HEIGHT = 224
RESIDUAL_IMAGE_WIDTH = 224


def _find_first_key(obs: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(
        f"Missing keys {candidates} in observation. Available keys: {sorted(obs.keys())}"
    )


def _maybe_find_first_key(obs: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    return None


def _trim_alpha(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {array.shape}")
    return np.ascontiguousarray(array)


def _resize_with_pad(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    array = _trim_alpha(image)
    if array.shape[:2] == (height, width):
        return array

    pil_image = Image.fromarray(array)
    current_width, current_height = pil_image.size
    resize_ratio = max(current_width / width, current_height / height)
    resized_width = max(1, int(current_width / resize_ratio))
    resized_height = max(1, int(current_height / resize_ratio))
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (width, height), 0)
    pad_width = max(0, (width - resized_width) // 2)
    pad_height = max(0, (height - resized_height) // 2)
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas, dtype=np.uint8)


def build_agibot_state(obs: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(
        _find_first_key(obs, ("state/pose", "observation/state", "pose")),
        dtype=np.float32,
    ).reshape(-1)
    if int(pose.shape[0]) != AGIBOT_STATE_DIM:
        raise ValueError(
            f"AgiBot camera-position state must be {AGIBOT_STATE_DIM}D, got {pose.shape}"
        )
    return np.asarray(pose, dtype=np.float32)


def build_agibot_right_arm_state(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        build_agibot_state(obs)[AGIBOT_RIGHT_ARM_STATE_SLICE],
        dtype=np.float32,
    )


def build_agibot_joyra_state(obs: dict[str, Any]) -> np.ndarray | None:
    head = _maybe_find_first_key(obs, ("state/head", "head_state", "head"))
    waist = _maybe_find_first_key(obs, ("state/waist", "waist_state", "waist"))
    if head is None or waist is None:
        return None

    pose = build_agibot_state(obs)
    head_array = np.asarray(head, dtype=np.float32).reshape(-1)
    waist_array = np.asarray(waist, dtype=np.float32).reshape(-1)
    if int(head_array.shape[0]) != 2:
        raise ValueError(f"AgiBot JoyRA head state must be 2D, got {head_array.shape}")
    if int(waist_array.shape[0]) != 2:
        raise ValueError(
            f"AgiBot JoyRA waist state must be 2D, got {waist_array.shape}"
        )

    joyra_state = np.concatenate([pose, head_array, waist_array], axis=0)
    if int(joyra_state.shape[0]) != AGIBOT_JOYRA_STATE_DIM:
        raise ValueError(
            f"AgiBot JoyRA state must be {AGIBOT_JOYRA_STATE_DIM}D, got {joyra_state.shape}"
        )
    return np.asarray(joyra_state, dtype=np.float32)


def extract_agibot_policy_images(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "image_rgb_0": _trim_alpha(
            _find_first_key(obs, ("image/head", "head_image", "observation/image"))
        ),
        "image_rgb_1": _trim_alpha(
            _find_first_key(
                obs,
                (
                    "image/left_wrist",
                    "left_wrist_image",
                    "observation/wrist_left_image",
                ),
            )
        ),
        "image_rgb_2": _trim_alpha(
            _find_first_key(
                obs,
                (
                    "image/right_wrist",
                    "right_wrist_image",
                    "observation/wrist_right_image",
                ),
            )
        ),
    }


def extract_agibot_residual_images(
    obs: dict[str, Any],
    *,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    resolved_image_keys = resolve_agibot_image_keys(image_keys)
    policy_images = extract_agibot_policy_images(obs)
    return {
        key: _resize_with_pad(
            policy_images[key],
            height=RESIDUAL_IMAGE_HEIGHT,
            width=RESIDUAL_IMAGE_WIDTH,
        )
        for key in resolved_image_keys
    }


__all__ = [
    "AGIBOT_ARM_STATE_DIM",
    "AGIBOT_JOYRA_STATE_DIM",
    "AGIBOT_LEFT_ARM_STATE_SLICE",
    "AGIBOT_RIGHT_ARM_STATE_SLICE",
    "AGIBOT_STATE_DIM",
    "RESIDUAL_IMAGE_HEIGHT",
    "RESIDUAL_IMAGE_WIDTH",
    "build_agibot_joyra_state",
    "build_agibot_right_arm_state",
    "build_agibot_state",
    "extract_agibot_policy_images",
    "extract_agibot_residual_images",
]
