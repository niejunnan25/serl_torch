"""LIBERO-specific runtime observation adapters."""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np
from PIL import Image
from serl_launcher.residual.observation import build_residual_step_obs_from_core

from ..schema import resolve_libero_image_keys

RESIDUAL_IMAGE_HEIGHT = 224
RESIDUAL_IMAGE_WIDTH = 224


def _lru_get(cache: "OrderedDict[Hashable, Any]", key: Hashable) -> Any:
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _lru_set(
    cache: "OrderedDict[Hashable, Any]", key: Hashable, value: Any, *, limit: int
) -> Any:
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max(1, int(limit)):
        cache.popitem(last=False)
    return value


def _resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[-3:-1] == (height, width):
        return np.asarray(img, dtype=np.uint8)
    pil_img = Image.fromarray(np.asarray(img, dtype=np.uint8))
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


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float32).copy()
    quat_arr[3] = np.clip(quat_arr[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat_arr[3] * quat_arr[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat_arr[:3] * 2.0 * math.acos(float(quat_arr[3])) / den).astype(
        np.float32
    )


def _find_first_key(obs: Dict[str, Any], candidates: Tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(
        f"Missing keys {candidates} in observation. Available keys: {list(obs.keys())}"
    )


def _compute_libero_state(obs: Dict[str, Any]) -> np.ndarray:
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


def _compute_residual_images(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    image_rgb_0 = _preprocess_rgb(
        _find_first_key(
            obs, ("agentview_image", "agentview_rgb", "image", "front_rgb")
        )
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


class _ObservationIdentityKey:
    """Object-identity cache key that keeps the original observation alive."""

    __slots__ = ("_obj", "_obj_id")

    def __init__(self, obj: Dict[str, Any]) -> None:
        self._obj = obj
        self._obj_id = id(obj)

    def __hash__(self) -> int:
        return self._obj_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ObservationIdentityKey):
            return False
        return self._obj is other._obj


class LiberoObservationCache:
    """Lightweight LRU cache for repeated LIBERO observation preprocessing."""

    def __init__(
        self, *, max_obs_entries: int = 64, max_step_obs_entries: int = 256
    ) -> None:
        self.max_obs_entries = max(1, int(max_obs_entries))
        self.max_step_obs_entries = max(1, int(max_step_obs_entries))
        self._lock = threading.RLock()
        self._image_cache: "OrderedDict[Hashable, Dict[str, np.ndarray]]" = (
            OrderedDict()
        )
        self._state_cache: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._step_obs_cache: "OrderedDict[Hashable, Dict[str, np.ndarray]]" = (
            OrderedDict()
        )

    def clear(self) -> None:
        with self._lock:
            self._image_cache.clear()
            self._state_cache.clear()
            self._step_obs_cache.clear()

    def _resolve_key(
        self, obs: Dict[str, Any], cache_key: Optional[Hashable]
    ) -> Hashable:
        if cache_key is not None:
            return cache_key
        return _ObservationIdentityKey(obs)

    def get_images(
        self,
        obs: Dict[str, Any],
        *,
        cache_key: Optional[Hashable] = None,
    ) -> Dict[str, np.ndarray]:
        with self._lock:
            obs_key = self._resolve_key(obs, cache_key)
            cached = _lru_get(self._image_cache, obs_key)
            if cached is not None:
                return cached
            images = _compute_residual_images(obs)
            return _lru_set(
                self._image_cache, obs_key, images, limit=self.max_obs_entries
            )

    def get_state(
        self,
        obs: Dict[str, Any],
        *,
        cache_key: Optional[Hashable] = None,
    ) -> np.ndarray:
        with self._lock:
            obs_key = self._resolve_key(obs, cache_key)
            cached = _lru_get(self._state_cache, obs_key)
            if cached is not None:
                return cached
            state = _compute_libero_state(obs)
            return _lru_set(
                self._state_cache, obs_key, state, limit=self.max_obs_entries
            )

    def build_residual_step_obs(
        self,
        obs: Dict[str, Any],
        base_action: np.ndarray,
        *,
        image_keys: Tuple[str, ...],
        stack_horizon: int = 1,
        cache_key: Optional[Hashable] = None,
        action_dim: Optional[int] = None,
        base_action_chunk: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        with self._lock:
            image_keys = resolve_libero_image_keys(image_keys)
            if int(stack_horizon) != 1:
                raise ValueError(
                    f"Only stack_horizon=1 is currently supported, got {stack_horizon}"
                )

            obs_key = self._resolve_key(obs, cache_key)
            base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
            base_action_chunk_arr = (
                np.asarray(base_action_chunk, dtype=np.float32)
                if base_action_chunk is not None
                else None
            )
            fused_key = (
                obs_key,
                base_action_arr.tobytes(),
                None
                if base_action_chunk_arr is None
                else base_action_chunk_arr.tobytes(),
                None if alpha is None else float(alpha),
                image_keys,
                int(stack_horizon),
                None if action_dim is None else int(action_dim),
            )
            cached = _lru_get(self._step_obs_cache, fused_key)
            if cached is not None:
                return cached

            core = build_residual_step_core(
                obs,
                image_keys=image_keys,
                obs_cache=self,
                cache_key=cache_key,
            )
            obs_out = build_residual_step_obs_from_core(
                core,
                base_action=base_action_arr,
                base_action_chunk=base_action_chunk_arr,
                alpha=alpha,
                stack_horizon=int(stack_horizon),
            )
            if action_dim is not None and base_action_arr.shape[0] != int(action_dim):
                raise ValueError(
                    f"Unexpected base action shape: {base_action_arr.shape}, expected ({int(action_dim)},)"
                )
            return _lru_set(
                self._step_obs_cache,
                fused_key,
                obs_out,
                limit=self.max_step_obs_entries,
            )


def build_libero_state(
    obs: Dict[str, Any],
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> np.ndarray:
    if obs_cache is not None:
        return obs_cache.get_state(obs, cache_key=cache_key)
    return np.asarray(_compute_libero_state(obs), dtype=np.float32)


def extract_residual_images(
    obs: Dict[str, Any],
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    if obs_cache is not None:
        return obs_cache.get_images(obs, cache_key=cache_key)
    return _compute_residual_images(obs)


def build_residual_step_core(
    obs: Dict[str, Any],
    *,
    image_keys: Tuple[str, ...],
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    """Return preprocessed LIBERO step components before base-action fusion."""
    state_core = build_libero_state(
        obs,
        obs_cache=obs_cache,
        cache_key=cache_key,
    )
    resolved_image_keys = resolve_libero_image_keys(image_keys)
    images_all = extract_residual_images(
        obs,
        obs_cache=obs_cache,
        cache_key=cache_key,
    )
    missing_keys = [key for key in resolved_image_keys if key not in images_all]
    if missing_keys:
        raise KeyError(
            f"Unsupported image key(s): {missing_keys}. "
            f"Available keys: {list(images_all.keys())}"
        )
    payload: Dict[str, np.ndarray] = {
        "state_core": np.asarray(state_core, dtype=np.float32)
    }
    for key in resolved_image_keys:
        payload[key] = np.array(images_all[key], copy=True)
    return payload


def build_residual_step_obs(
    obs: Dict[str, Any],
    base_action: np.ndarray,
    image_keys: Tuple[str, ...],
    stack_horizon: int = 1,
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
    action_dim: Optional[int] = None,
    base_action_chunk: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    if obs_cache is not None:
        return obs_cache.build_residual_step_obs(
            obs,
            base_action,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            cache_key=cache_key,
            action_dim=action_dim,
            base_action_chunk=base_action_chunk,
            alpha=alpha,
        )

    core = build_residual_step_core(
        obs,
        image_keys=image_keys,
    )
    base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if action_dim is not None and base_action_arr.shape[0] != int(action_dim):
        raise ValueError(
            f"Unexpected base action shape: {base_action_arr.shape}, expected ({int(action_dim)},)"
        )
    base_action_chunk_arr = (
        np.asarray(base_action_chunk, dtype=np.float32)
        if base_action_chunk is not None
        else None
    )
    return build_residual_step_obs_from_core(
        core,
        base_action=base_action_arr,
        base_action_chunk=base_action_chunk_arr,
        alpha=alpha,
        stack_horizon=int(stack_horizon),
    )
