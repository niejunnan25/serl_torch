"""Canonical policy-input helpers for AgiBot residual training."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from serl_launcher.policy.base import PolicyInput

from .arm_layout import ARM_LAYOUT_DUAL
from .arm_layout import normalize_arm_layout
from .observation import build_agibot_layout_joyra_state
from .observation import build_agibot_layout_state
from .observation import extract_agibot_policy_images


def build_agibot_policy_input(
    obs: dict[str, Any],
    prompt: str,
    *,
    arm_layout: str = ARM_LAYOUT_DUAL,
) -> PolicyInput:
    images = extract_agibot_policy_images(obs)
    layout = normalize_arm_layout(arm_layout)
    state = build_agibot_layout_state(obs, arm_layout=layout)
    joyra_state = build_agibot_layout_joyra_state(obs, arm_layout=layout)
    metadata: dict[str, Any] = {"openpi_layout": "dual_wrist"}
    if joyra_state is not None:
        metadata["joyra_state"] = np.asarray(joyra_state, dtype=np.float32)

    return PolicyInput(
        prompt=str(prompt),
        state=np.asarray(state, dtype=np.float32),
        images={
            "image_rgb_0": np.asarray(images["image_rgb_0"], dtype=np.uint8),
            "image_rgb_1": np.asarray(images["image_rgb_1"], dtype=np.uint8),
            "image_rgb_2": np.asarray(images["image_rgb_2"], dtype=np.uint8),
        },
        image_mask={
            "image_rgb_0": True,
            "image_rgb_1": True,
            "image_rgb_2": True,
        },
        metadata=metadata,
    )


def build_agibot_policy_inputs(
    observations: Sequence[dict[str, Any]],
    prompt: str,
    *,
    arm_layout: str = ARM_LAYOUT_DUAL,
) -> tuple[PolicyInput, ...]:
    return tuple(
        build_agibot_policy_input(obs=obs, prompt=prompt, arm_layout=arm_layout)
        for obs in observations
    )


__all__ = ["build_agibot_policy_input", "build_agibot_policy_inputs"]
