"""AgiBot-specific runtime observation adapters."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image
from serl_launcher.data.normalizer import StateActionNormalizer
from serl_launcher.residual.observation import build_residual_step_obs_from_core
from serl_launcher.residual.observation import normalize_residual_observation_state_mode

from ..schema import resolve_agibot_image_keys

RESIDUAL_IMAGE_HEIGHT = 224
RESIDUAL_IMAGE_WIDTH = 224


def _lru_get(cache: "OrderedDict[Hashable, Any]", key: Hashable) -> Any:
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return value


def _lru_set(
    cache: "OrderedDict[Hashable, Any]",
    key: Hashable,
    value: Any,
    *,
    limit: int,
) -> Any:
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max(1, int(limit)):
        cache.popitem(last=False)
    return value


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


def _find_first_key(obs: Dict[str, Any], candidates: Tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(
        f"Missing keys {candidates} in observation. Available keys: {list(obs.keys())}"
    )


def _maybe_find_first_key(obs: Dict[str, Any], candidates: Tuple[str, ...]) -> Any:
    for key in candidates:
        if key in obs:
            return obs[key]
    return None


def _compute_agibot_state(obs: Dict[str, Any]) -> np.ndarray:
    pose = np.asarray(
        _find_first_key(obs, ("state/pose", "observation/state", "pose")),
        dtype=np.float32,
    ).reshape(-1)
    if pose.shape[0] != 14:
        raise ValueError(f"AgiBot camera-position state must be 14D, got {pose.shape}")
    return pose.astype(np.float32)


def _compute_agibot_joyra_state(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    pose = _compute_agibot_state(obs)
    head = _maybe_find_first_key(obs, ("state/head", "head_state", "head"))
    waist = _maybe_find_first_key(obs, ("state/waist", "waist_state", "waist"))
    if head is None or waist is None:
        return None

    head_arr = np.asarray(head, dtype=np.float32).reshape(-1)
    waist_arr = np.asarray(waist, dtype=np.float32).reshape(-1)
    if head_arr.shape[0] != 2:
        raise ValueError("AgiBot JoyRA head state must be 2D, got " f"{head_arr.shape}")
    if waist_arr.shape[0] != 2:
        raise ValueError(
            "AgiBot JoyRA waist state must be 2D, got " f"{waist_arr.shape}"
        )
    return np.concatenate([pose, head_arr, waist_arr], axis=0).astype(np.float32)


def _compute_policy_images(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
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


def _compute_residual_images(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    policy_images = _compute_policy_images(obs)
    return {
        key: _resize_with_pad(value, RESIDUAL_IMAGE_HEIGHT, RESIDUAL_IMAGE_WIDTH)
        for key, value in policy_images.items()
    }


class _ObservationIdentityKey:
    __slots__ = ("_obj", "_obj_id")

    def __init__(self, obj: Dict[str, Any]) -> None:
        self._obj = obj
        self._obj_id = id(obj)

    def __hash__(self) -> int:
        return self._obj_id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ObservationIdentityKey) and self._obj is other._obj


class AgiBotObservationCache:
    """Lightweight LRU cache for repeated AgiBot observation preprocessing."""

    def __init__(
        self,
        *,
        max_obs_entries: int = 64,
        max_step_obs_entries: int = 256,
    ) -> None:
        self.max_obs_entries = max(1, int(max_obs_entries))
        self.max_step_obs_entries = max(1, int(max_step_obs_entries))
        self._lock = threading.RLock()
        self._policy_image_cache: "OrderedDict[Hashable, Dict[str, np.ndarray]]" = (
            OrderedDict()
        )
        self._residual_image_cache: "OrderedDict[Hashable, Dict[str, np.ndarray]]" = (
            OrderedDict()
        )
        self._state_cache: "OrderedDict[Hashable, np.ndarray]" = OrderedDict()
        self._normalized_state_cache: "OrderedDict[Hashable, np.ndarray]" = (
            OrderedDict()
        )
        self._step_obs_cache: "OrderedDict[Hashable, Dict[str, np.ndarray]]" = (
            OrderedDict()
        )

    def clear(self) -> None:
        with self._lock:
            self._policy_image_cache.clear()
            self._residual_image_cache.clear()
            self._state_cache.clear()
            self._normalized_state_cache.clear()
            self._step_obs_cache.clear()

    def _resolve_key(
        self,
        obs: Dict[str, Any],
        cache_key: Optional[Hashable],
    ) -> Hashable:
        if cache_key is not None:
            return cache_key
        return _ObservationIdentityKey(obs)

    def get_policy_images(
        self,
        obs: Dict[str, Any],
        *,
        cache_key: Optional[Hashable] = None,
    ) -> Dict[str, np.ndarray]:
        with self._lock:
            obs_key = self._resolve_key(obs, cache_key)
            cached = _lru_get(self._policy_image_cache, obs_key)
            if cached is not None:
                return cached
            images = _compute_policy_images(obs)
            return _lru_set(
                self._policy_image_cache,
                obs_key,
                images,
                limit=self.max_obs_entries,
            )

    def get_residual_images(
        self,
        obs: Dict[str, Any],
        *,
        cache_key: Optional[Hashable] = None,
    ) -> Dict[str, np.ndarray]:
        with self._lock:
            obs_key = self._resolve_key(obs, cache_key)
            cached = _lru_get(self._residual_image_cache, obs_key)
            if cached is not None:
                return cached
            images = _compute_residual_images(obs)
            return _lru_set(
                self._residual_image_cache,
                obs_key,
                images,
                limit=self.max_obs_entries,
            )

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
                state = _compute_agibot_state(obs)
                return _lru_set(
                    self._state_cache,
                    obs_key,
                    state,
                    limit=self.max_obs_entries,
                )

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
        action_dim: Optional[int] = None,
        base_action_chunk: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        state_mode: str = "fused",
    ) -> Dict[str, np.ndarray]:
        with self._lock:
            image_keys = resolve_agibot_image_keys(image_keys)
            normalized_state_mode = normalize_residual_observation_state_mode(
                state_mode
            )
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
                normalized_state_mode,
                None if normalizer is None else id(normalizer),
                None if action_dim is None else int(action_dim),
            )
            cached = _lru_get(self._step_obs_cache, fused_key)
            if cached is not None:
                return cached

            core = build_residual_step_core(
                obs,
                image_keys=image_keys,
                normalizer=normalizer,
                obs_cache=self,
                cache_key=cache_key,
            )
            obs_out = build_residual_step_obs_from_core(
                core,
                base_action=base_action_arr,
                base_action_chunk=base_action_chunk_arr,
                alpha=alpha,
                normalizer=normalizer,
                state_mode=normalized_state_mode,
                stack_horizon=int(stack_horizon),
            )
            if action_dim is not None and base_action_arr.shape[0] != int(action_dim):
                raise ValueError(
                    f"Unexpected base action shape: {base_action_arr.shape}, "
                    f"expected ({int(action_dim)},)"
                )
            return _lru_set(
                self._step_obs_cache,
                fused_key,
                obs_out,
                limit=self.max_step_obs_entries,
            )


def build_agibot_state(
    obs: Dict[str, Any],
    *,
    obs_cache: Optional[AgiBotObservationCache] = None,
    cache_key: Optional[Hashable] = None,
    normalizer: Optional[StateActionNormalizer] = None,
) -> np.ndarray:
    if obs_cache is not None:
        return obs_cache.get_state(obs, normalizer=normalizer, cache_key=cache_key)
    state = _compute_agibot_state(obs)
    if normalizer is not None:
        state = normalizer.normalize_state(state)
    return np.asarray(state, dtype=np.float32)


def build_agibot_joyra_state(
    obs: Dict[str, Any],
) -> Optional[np.ndarray]:
    joyra_state = _compute_agibot_joyra_state(obs)
    if joyra_state is None:
        return None
    return np.asarray(joyra_state, dtype=np.float32)


def extract_policy_images(
    obs: Dict[str, Any],
    *,
    obs_cache: Optional[AgiBotObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    if obs_cache is not None:
        return obs_cache.get_policy_images(obs, cache_key=cache_key)
    return _compute_policy_images(obs)


def extract_residual_images(
    obs: Dict[str, Any],
    *,
    obs_cache: Optional[AgiBotObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    if obs_cache is not None:
        return obs_cache.get_residual_images(obs, cache_key=cache_key)
    return _compute_residual_images(obs)


def build_residual_step_core(
    obs: Dict[str, Any],
    *,
    image_keys: Tuple[str, ...],
    normalizer: Optional[StateActionNormalizer] = None,
    obs_cache: Optional[AgiBotObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    state_core = build_agibot_state(
        obs,
        obs_cache=obs_cache,
        cache_key=cache_key,
        normalizer=normalizer,
    )
    resolved_image_keys = resolve_agibot_image_keys(image_keys)
    images_all = extract_residual_images(obs, obs_cache=obs_cache, cache_key=cache_key)
    missing_keys = [key for key in resolved_image_keys if key not in images_all]
    if missing_keys:
        raise KeyError(
            f"Unsupported image key(s): {missing_keys}. "
            f"Available keys: {list(images_all.keys())}"
        )
    payload: Dict[str, np.ndarray] = {
        "state_core": np.asarray(state_core, dtype=np.float32),
    }
    for key in resolved_image_keys:
        payload[key] = np.array(images_all[key], copy=True)
    return payload


def build_residual_step_obs(
    obs: Dict[str, Any],
    base_action: np.ndarray,
    image_keys: Tuple[str, ...],
    stack_horizon: int = 1,
    normalizer: Optional[StateActionNormalizer] = None,
    *,
    obs_cache: Optional[AgiBotObservationCache] = None,
    cache_key: Optional[Hashable] = None,
    action_dim: Optional[int] = None,
    base_action_chunk: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    state_mode: str = "fused",
) -> Dict[str, np.ndarray]:
    if obs_cache is not None:
        return obs_cache.build_residual_step_obs(
            obs,
            base_action,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            normalizer=normalizer,
            cache_key=cache_key,
            action_dim=action_dim,
            base_action_chunk=base_action_chunk,
            alpha=alpha,
            state_mode=state_mode,
        )

    core = build_residual_step_core(
        obs,
        image_keys=image_keys,
        normalizer=normalizer,
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
        normalizer=normalizer,
        state_mode=state_mode,
        stack_horizon=int(stack_horizon),
    )
