"""Canonical policy-input helpers for AgiBot residual training."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from serl_launcher.policy.base import PolicyInput

from .observation import build_agibot_joyra_state
from .observation import build_agibot_state
from .observation import extract_agibot_policy_images


def build_agibot_policy_input(
    obs: dict[str, Any],
    prompt: str,
    *,
    state_slice: slice | None = None,
) -> PolicyInput:
    images = extract_agibot_policy_images(obs)
    full_state = build_agibot_state(obs)
    state = full_state if state_slice is None else full_state[state_slice]
    joyra_state = build_agibot_joyra_state(obs)
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
    state_slice: slice | None = None,
) -> tuple[PolicyInput, ...]:
    return tuple(
        build_agibot_policy_input(obs=obs, prompt=prompt, state_slice=state_slice)
        for obs in observations
    )


__all__ = ["build_agibot_policy_input", "build_agibot_policy_inputs"]
