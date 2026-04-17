"""LIBERO policy-input builders derived from raw environment observations."""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Sequence

import numpy as np

from serl_launcher.policy.base import PolicyInput

from .observation import build_libero_state
from .observation import extract_libero_images


def build_libero_policy_input(
    obs: Dict[str, Any],
    prompt: str,
    *,
    obs_cache: Optional[Any] = None,
    cache_key: Optional[Hashable] = None,
) -> PolicyInput:
    images = extract_libero_images(obs, obs_cache=obs_cache, cache_key=cache_key)
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


def build_libero_policy_inputs(
    observations: Sequence[Dict[str, Any]],
    prompt: str,
) -> list[PolicyInput]:
    return [
        build_libero_policy_input(obs, prompt)
        for obs in observations
    ]


__all__ = ["build_libero_policy_input", "build_libero_policy_inputs"]
