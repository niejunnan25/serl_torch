"""Lightweight LIBERO observation helpers for reference-style training."""
from __future__ import annotations

import math
from typing import Any
from typing import Hashable
from typing import Optional

import numpy as np
from PIL import Image
try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from ..schema import resolve_libero_image_keys

LIBERO_STATE_DIM = 8
RESIDUAL_IMAGE_HEIGHT = 224
RESIDUAL_IMAGE_WIDTH = 224


def _resolve_residual_alpha(alpha: float) -> float:
    residual_alpha = float(alpha)
    if (not math.isfinite(residual_alpha)) or residual_alpha < 0.0:
        raise ValueError(
            f"residual.alpha must be finite and >= 0.0, got {alpha!r}"
        )
    return residual_alpha


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
    return (quat_arr[:3] * 2.0 * math.acos(float(quat_arr[3])) / den).astype(np.float32)


def _find_first_key(obs: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(
        f"Missing keys {candidates} in observation. Available keys: {list(obs.keys())}"
    )


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


def extract_residual_images(
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


def build_fused_residual_observation(
    *,
    obs: dict[str, Any],
    base_action: np.ndarray,
    image_keys: tuple[str, ...],
    alpha: float,
) -> dict[str, np.ndarray]:
    residual_alpha = _resolve_residual_alpha(alpha)
    state_core = build_libero_state(obs)
    base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    fused_state = np.concatenate(
        (
            state_core,
            base_action_arr,
            np.asarray([residual_alpha], dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    images = extract_residual_images(obs)
    residual_obs: dict[str, np.ndarray] = {
        "state": np.expand_dims(fused_state, axis=0).astype(np.float32),
        "base_action": np.expand_dims(base_action_arr, axis=0).astype(np.float32),
        "alpha": np.asarray([[residual_alpha]], dtype=np.float32),
    }
    for key in resolve_libero_image_keys(image_keys):
        if key not in images:
            raise KeyError(
                f"Unsupported image key {key!r}. Available keys: {list(images.keys())}"
            )
        residual_obs[key] = np.expand_dims(np.asarray(images[key]).copy(), axis=0)
    return residual_obs


def build_residual_sample_obs(
    *,
    action_dim: int,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    sample_obs: dict[str, np.ndarray] = {
        "state": np.zeros((1, LIBERO_STATE_DIM + int(action_dim) + 1), dtype=np.float32),
        "base_action": np.zeros((1, int(action_dim)), dtype=np.float32),
        "alpha": np.zeros((1, 1), dtype=np.float32),
    }
    for key in resolve_libero_image_keys(image_keys):
        sample_obs[key] = np.zeros(
            (1, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH, 3),
            dtype=np.uint8,
        )
    return sample_obs


def build_chunk_residual_sample_obs(
    *,
    action_dim: int,
    chunk_horizon: int,
    image_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if int(action_dim) <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if int(chunk_horizon) <= 0:
        raise ValueError(f"chunk_horizon must be positive, got {chunk_horizon}")
    sample_obs: dict[str, np.ndarray] = {
        "state": np.zeros(
            (
                1,
                LIBERO_STATE_DIM
                + int(action_dim)
                + int(chunk_horizon) * int(action_dim)
                + 1,
            ),
            dtype=np.float32,
        ),
        "base_action": np.zeros((1, int(action_dim)), dtype=np.float32),
        "base_action_chunk": np.zeros(
            (1, int(chunk_horizon), int(action_dim)),
            dtype=np.float32,
        ),
        "alpha": np.zeros((1, 1), dtype=np.float32),
    }
    for key in resolve_libero_image_keys(image_keys):
        sample_obs[key] = np.zeros(
            (1, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH, 3),
            dtype=np.uint8,
        )
    return sample_obs


def build_residual_observation_space(
    *,
    sample_obs: dict[str, np.ndarray],
    image_keys: tuple[str, ...],
) -> gym.spaces.Dict:
    resolved_image_keys = set(resolve_libero_image_keys(image_keys))
    spaces: dict[str, gym.Space] = {}
    for key, value in sample_obs.items():
        value_arr = np.asarray(value)
        if key in resolved_image_keys:
            spaces[key] = gym.spaces.Box(
                low=0,
                high=255,
                shape=value_arr.shape,
                dtype=np.uint8,
            )
        else:
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=value_arr.shape,
                dtype=np.float32,
            )
    return gym.spaces.Dict(spaces)


def build_chunk_residual_observation_space(
    *,
    sample_obs: dict[str, np.ndarray],
    image_keys: tuple[str, ...],
) -> gym.spaces.Dict:
    return build_residual_observation_space(
        sample_obs=sample_obs,
        image_keys=image_keys,
    )
