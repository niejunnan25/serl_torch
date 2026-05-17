"""LIBERO policy-input builders derived from parsed observation parts."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from serl_launcher.policy.base import PolicyInput


def build_libero_policy_input(
    *,
    prompt: str,
    state: np.ndarray,
    images: Mapping[str, np.ndarray],
) -> PolicyInput:
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


__all__ = ["build_libero_policy_input"]
