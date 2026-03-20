"""Observation preprocessing for LIBERO residual RL and OpenPI."""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np
from PIL import Image

from ..data.normalizer import StateActionNormalizer
from ..utils.constants import LIBERO_ACTION_DIM

RESIDUAL_IMAGE_HEIGHT = 224
RESIDUAL_IMAGE_WIDTH = 224


def _lru_get(cache: "OrderedDict[Hashable, Any]", key: Hashable) -> Any:
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _lru_set(cache: "OrderedDict[Hashable, Any]", key: Hashable, value: Any, *, limit: int) -> Any:
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
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _find_first_key(obs: Dict[str, Any], candidates: Tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"Missing keys {candidates} in observation. Available keys: {list(obs.keys())}")


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
                _find_first_key(obs, ("robot0_gripper_qpos", "gripper_states", "gripper_qpos")),
                dtype=np.float32,
            ),
        ),
        axis=-1,
    ).astype(np.float32)


def _preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    rotated = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[::-1, ::-1])
    return _resize_with_pad(rotated, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH)


def _compute_residual_images(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {
        "image": _preprocess_rgb(
            _find_first_key(obs, ("agentview_image", "agentview_rgb", "image", "front_rgb"))
        ),
        "wrist_image": _preprocess_rgb(
            _find_first_key(
                obs,
                ("robot0_eye_in_hand_image", "eye_in_hand_rgb", "wrist_image", "hand_rgb"),
            )
        ),
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

    def __init__(self, *, max_obs_entries: int = 64, max_step_obs_entries: int = 256) -> None:
        self.max_obs_entries = max(1, int(max_obs_entries))
        self.max_step_obs_entries = max(1, int(max_step_obs_entries))
        self._lock = threading.RLock()
        self._image_cache: "OrderedDict[Hashable, Dict[str, np.ndarray]]" = OrderedDict()
        self._state_cache: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._normalized_state_cache: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._step_obs_cache: "OrderedDict[Hashable, Dict[str, np.ndarray]]" = OrderedDict()

    def clear(self) -> None:
        with self._lock:
            self._image_cache.clear()
            self._state_cache.clear()
            self._normalized_state_cache.clear()
            self._step_obs_cache.clear()

    def _resolve_key(self, obs: Dict[str, Any], cache_key: Optional[Hashable]) -> Hashable:
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
            return _lru_set(self._image_cache, obs_key, images, limit=self.max_obs_entries)

    def get_state(
        self,
        obs: Dict[str, Any],
        *,
        normalizer: Optional[StateActionNormalizer] = None,
        cache_key: Optional[Hashable] = None,
    ) -> np.ndarray:
        with self._lock:
            obs_key = self._resolve_key(obs, cache_key)
            norm_key = None if normalizer is None else id(normalizer)
            if norm_key is None:
                cached = _lru_get(self._state_cache, obs_key)
                if cached is not None:
                    return cached
                state = _compute_libero_state(obs)
                return _lru_set(self._state_cache, obs_key, state, limit=self.max_obs_entries)

            cache_token = (obs_key, norm_key)
            cached = _lru_get(self._normalized_state_cache, cache_token)
            if cached is not None:
                return cached

            state = self.get_state(obs, cache_key=cache_key)
            normalized_state = normalizer.normalize_state(state)
            return _lru_set(
                self._normalized_state_cache,
                cache_token,
                np.asarray(normalized_state, dtype=np.float32),
                limit=self.max_obs_entries,
            )

    def build_residual_step_obs(
        self,
        obs: Dict[str, Any],
        base_action: np.ndarray,
        *,
        image_keys: Tuple[str, ...],
        stack_horizon: int = 1,
        normalizer: Optional[StateActionNormalizer] = None,
        cache_key: Optional[Hashable] = None,
    ) -> Dict[str, np.ndarray]:
        with self._lock:
            image_keys = tuple(image_keys)
            if stack_horizon != 1:
                raise ValueError(f"Only stack_horizon=1 is currently supported, got {stack_horizon}")

            obs_key = self._resolve_key(obs, cache_key)
            base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
            fused_key = (
                obs_key,
                base_action_arr.tobytes(),
                image_keys,
                int(stack_horizon),
                None if normalizer is None else id(normalizer),
            )
            cached = _lru_get(self._step_obs_cache, fused_key)
            if cached is not None:
                return cached

            if base_action_arr.shape[0] != LIBERO_ACTION_DIM:
                raise ValueError(
                    f"Unexpected base action shape: {base_action_arr.shape}, expected ({LIBERO_ACTION_DIM},)"
                )

            state = self.get_state(obs, normalizer=normalizer, cache_key=cache_key)
            if normalizer is not None:
                base_action_norm = normalizer.normalize_action(base_action_arr)
            else:
                base_action_norm = base_action_arr

            fused_state = np.concatenate(
                [state, np.asarray(base_action_norm, dtype=np.float32)],
                axis=-1,
            ).astype(np.float32)
            images_all = self.get_images(obs, cache_key=cache_key)
            missing_keys = [key for key in image_keys if key not in images_all]
            if missing_keys:
                raise KeyError(
                    f"Unsupported image key(s): {missing_keys}. "
                    f"Available keys: {list(images_all.keys())}"
                )

            stacked = {key: np.expand_dims(images_all[key], axis=0) for key in image_keys}
            stacked["state"] = np.expand_dims(fused_state, axis=0)
            return _lru_set(self._step_obs_cache, fused_key, stacked, limit=self.max_step_obs_entries)


def build_libero_state(
    obs: Dict[str, Any],
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
    normalizer: Optional[StateActionNormalizer] = None,
) -> np.ndarray:
    if obs_cache is not None:
        return obs_cache.get_state(obs, normalizer=normalizer, cache_key=cache_key)
    state = _compute_libero_state(obs)
    if normalizer is not None:
        state = normalizer.normalize_state(state)
    return np.asarray(state, dtype=np.float32)


def extract_residual_images(
    obs: Dict[str, Any],
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    if obs_cache is not None:
        return obs_cache.get_images(obs, cache_key=cache_key)
    return _compute_residual_images(obs)


def build_residual_step_obs(
    obs: Dict[str, Any],
    base_action: np.ndarray,
    image_keys: Tuple[str, ...],
    stack_horizon: int = 1,
    normalizer: Optional[StateActionNormalizer] = None,
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    if obs_cache is not None:
        return obs_cache.build_residual_step_obs(
            obs,
            base_action,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            normalizer=normalizer,
            cache_key=cache_key,
        )

    state = build_libero_state(obs, normalizer=normalizer)
    base_action = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if base_action.shape[0] != LIBERO_ACTION_DIM:
        raise ValueError(
            f"Unexpected base action shape: {base_action.shape}, expected ({LIBERO_ACTION_DIM},)"
        )
    if normalizer is not None:
        base_action_norm = normalizer.normalize_action(base_action)
    else:
        base_action_norm = base_action

    fused_state = np.concatenate([state, base_action_norm], axis=-1).astype(np.float32)
    images_all = extract_residual_images(obs)
    missing_keys = [key for key in image_keys if key not in images_all]
    if missing_keys:
        raise KeyError(
            f"Unsupported image key(s): {missing_keys}. "
            f"Available keys: {list(images_all.keys())}"
        )
    if stack_horizon != 1:
        raise ValueError(f"Only stack_horizon=1 is currently supported, got {stack_horizon}")

    stacked = {key: np.expand_dims(images_all[key], axis=0) for key in image_keys}
    stacked["state"] = np.expand_dims(fused_state, axis=0)
    return stacked
