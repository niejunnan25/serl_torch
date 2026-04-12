"""LIBERO environment observation parsing helpers."""
from __future__ import annotations

import math
from typing import Any
from typing import Hashable
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image

LIBERO_STATE_DIM = 8
RESIDUAL_IMAGE_HEIGHT = 224
RESIDUAL_IMAGE_WIDTH = 224
LIBERO_IMAGE_SLOT_KEYS = ("image_rgb_0", "image_rgb_1", "image_rgb_2")
LIBERO_DEFAULT_IMAGE_KEYS = ("image_rgb_0", "image_rgb_1")
LIBERO_IMAGE_VIEW_TO_SLOT = {
    "image": "image_rgb_0",
    "wrist_image": "image_rgb_1",
    "image_rgb_0": "image_rgb_0",
    "image_rgb_1": "image_rgb_1",
    "image_rgb_2": "image_rgb_2",
}


def _resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape[-3:-1] == (height, width):
        return arr
    pil_img = Image.fromarray(arr)
    cur_w, cur_h = pil_img.size
    ratio = max(cur_w / float(width), cur_h / float(height))
    new_w = int(cur_w / ratio)
    new_h = int(cur_h / ratio)
    resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_w = max(0, (width - new_w) // 2)
    pad_h = max(0, (height - new_h) // 2)
    canvas.paste(resized, (pad_w, pad_h))
    return np.asarray(canvas, dtype=np.uint8)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float32).copy()
    quat_arr[3] = np.clip(quat_arr[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat_arr[3] * quat_arr[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat_arr[:3] * 2.0 * math.acos(float(quat_arr[3])) / den).astype(
        np.float32
    )


def _find_first_key(obs: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(
        f"Missing keys {candidates} in observation. Available keys: {list(obs.keys())}"
    )


def resolve_libero_image_key(image_key: str) -> str:
    key = str(image_key)
    resolved = LIBERO_IMAGE_VIEW_TO_SLOT.get(key, None)
    if resolved is None:
        raise KeyError(
            f"Unsupported LIBERO image key {key!r}. "
            f"Expected one of {sorted(LIBERO_IMAGE_VIEW_TO_SLOT)}"
        )
    return resolved


def resolve_libero_image_keys(image_keys: Iterable[str]) -> Tuple[str, ...]:
    resolved_keys = []
    seen = set()
    for image_key in image_keys:
        resolved = resolve_libero_image_key(str(image_key))
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_keys.append(resolved)
    if not resolved_keys:
        raise ValueError("At least one LIBERO image key is required")
    return tuple(resolved_keys)


def build_libero_state(
    obs: dict[str, Any],
    *,
    obs_cache: Optional[Any] = None,
    cache_key: Optional[Hashable] = None,
) -> np.ndarray:
    del obs_cache, cache_key
    if "robot0_eef_quat" in obs:
        eef_ori = _quat2axisangle(obs["robot0_eef_quat"])
    else:
        eef_ori = np.asarray(
            _find_first_key(obs, ("robot0_eef_axis_angle", "ee_ori", "eef_axis_angle")),
            dtype=np.float32,
        )
    return np.concatenate(
        (
            np.asarray(
                _find_first_key(obs, ("robot0_eef_pos", "ee_pos", "eef_pos")),
                dtype=np.float32,
            ),
            np.asarray(eef_ori, dtype=np.float32),
            np.asarray(
                _find_first_key(
                    obs, ("robot0_gripper_qpos", "gripper_states", "gripper_qpos")
                ),
                dtype=np.float32,
            ),
        ),
        axis=-1,
    ).astype(np.float32)


def _preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    rotated = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[::-1, ::-1])
    return _resize_with_pad(rotated, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH)


def extract_libero_images(
    obs: dict[str, Any],
    *,
    obs_cache: Optional[Any] = None,
    cache_key: Optional[Hashable] = None,
) -> dict[str, np.ndarray]:
    del obs_cache, cache_key
    image_rgb_0 = _preprocess_rgb(
        _find_first_key(obs, ("agentview_image", "agentview_rgb", "image", "front_rgb"))
    )
    image_rgb_1 = _preprocess_rgb(
        _find_first_key(
            obs,
            (
                "robot0_eye_in_hand_image",
                "eye_in_hand_rgb",
                "wrist_image",
                "hand_rgb",
            ),
        )
    )
    return {
        "image_rgb_0": image_rgb_0,
        "image_rgb_1": image_rgb_1,
        "image_rgb_2": np.zeros_like(image_rgb_0, dtype=np.uint8),
    }


__all__ = [
    "LIBERO_DEFAULT_IMAGE_KEYS",
    "LIBERO_IMAGE_SLOT_KEYS",
    "LIBERO_IMAGE_VIEW_TO_SLOT",
    "LIBERO_STATE_DIM",
    "RESIDUAL_IMAGE_HEIGHT",
    "RESIDUAL_IMAGE_WIDTH",
    "build_libero_state",
    "extract_libero_images",
    "resolve_libero_image_key",
    "resolve_libero_image_keys",
]
