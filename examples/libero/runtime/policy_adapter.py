"""LIBERO runtime adapters for canonical policy inputs."""
from __future__ import annotations

from typing import Any, Dict, Hashable, Optional

import numpy as np

from serl_launcher.policy.base import PolicyInput

from .obs_adapter import LiberoObservationCache
from .obs_adapter import build_libero_state
from .obs_adapter import extract_residual_images


def build_libero_policy_input(
    obs: Dict[str, Any],
    prompt: str,
    *,
    obs_cache: Optional[LiberoObservationCache] = None,
    cache_key: Optional[Hashable] = None,
) -> PolicyInput:
    images = extract_residual_images(obs, obs_cache=obs_cache, cache_key=cache_key)
    state = build_libero_state(obs, obs_cache=obs_cache, cache_key=cache_key)
    image_mask = {
        "image_rgb_0": True,
        "image_rgb_1": True,
        "image_rgb_2": False,
    }
    return PolicyInput(
        prompt=str(prompt),
        state=np.asarray(state, dtype=np.float32),
        images={
            "image_rgb_0": np.asarray(images["image_rgb_0"], dtype=np.uint8),
            "image_rgb_1": np.asarray(images["image_rgb_1"], dtype=np.uint8),
            "image_rgb_2": np.asarray(images["image_rgb_2"], dtype=np.uint8),
        },
        image_mask=image_mask,
        metadata={},
    )
