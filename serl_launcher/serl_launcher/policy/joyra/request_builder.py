"""Helpers for building JoyRA requests from canonical policy inputs."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

from serl_launcher.policy.base import PolicyInput

JOYRA_IMAGE_HEIGHT = 224
JOYRA_IMAGE_WIDTH = 224


def _trim_alpha(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = _trim_alpha(img)
    if arr.shape[-3:-1] == (height, width):
        return np.asarray(arr, dtype=np.uint8)
    pil_img = Image.fromarray(arr)
    cur_w, cur_h = pil_img.size
    ratio = max(cur_w / width, cur_h / height)
    new_w = int(cur_w / ratio)
    new_h = int(cur_h / ratio)
    resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_w = max(0, (width - new_w) // 2)
    pad_h = max(0, (height - new_h) // 2)
    canvas.paste(resized, (pad_w, pad_h))
    return np.asarray(canvas, dtype=np.uint8)


def _resolve_primary_image(policy_input: PolicyInput) -> np.ndarray:
    if "image_rgb_0" not in policy_input.images:
        raise KeyError("PolicyInput.images must include 'image_rgb_0' for JoyRA")
    return _resize_with_pad(
        np.asarray(policy_input.images["image_rgb_0"], dtype=np.uint8),
        JOYRA_IMAGE_HEIGHT,
        JOYRA_IMAGE_WIDTH,
    )


def _resolve_optional_image(
    policy_input: PolicyInput,
    *,
    slot_name: str,
    primary_image: np.ndarray,
) -> np.ndarray:
    image = policy_input.images.get(slot_name, None)
    if image is None or (not bool(policy_input.image_mask.get(slot_name, False))):
        return np.zeros_like(primary_image, dtype=np.uint8)
    return _resize_with_pad(
        np.asarray(image, dtype=np.uint8),
        JOYRA_IMAGE_HEIGHT,
        JOYRA_IMAGE_WIDTH,
    )


def build_joyra_request(policy_input: PolicyInput) -> Dict[str, Any]:
    primary_image = _resolve_primary_image(policy_input)
    joyra_state = policy_input.metadata.get("joyra_state", policy_input.state)
    return {
        "observation/image": primary_image,
        "observation/wrist_left_image": _resolve_optional_image(
            policy_input,
            slot_name="image_rgb_1",
            primary_image=primary_image,
        ),
        "observation/wrist_right_image": _resolve_optional_image(
            policy_input,
            slot_name="image_rgb_2",
            primary_image=primary_image,
        ),
        "observation/state": np.asarray(joyra_state, dtype=np.float32),
        "prompt": str(policy_input.prompt),
    }


def build_joyra_batch_request(
    policy_inputs: Sequence[PolicyInput],
) -> Dict[str, Any]:
    if not policy_inputs:
        raise ValueError("policy_inputs must be non-empty for JoyRA batch infer")
    return {
        "examples": [build_joyra_request(policy_input) for policy_input in policy_inputs]
    }
