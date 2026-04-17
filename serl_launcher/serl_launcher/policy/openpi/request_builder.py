"""Helpers for building OpenPI requests from canonical policy inputs."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from serl_launcher.policy.base import PolicyInput


def _resolve_primary_image(policy_input: PolicyInput) -> np.ndarray:
    if "image_rgb_0" not in policy_input.images:
        raise KeyError("PolicyInput.images must include 'image_rgb_0' for OpenPI")
    return np.asarray(policy_input.images["image_rgb_0"], dtype=np.uint8)


def _resolve_wrist_image(policy_input: PolicyInput) -> np.ndarray:
    primary = _resolve_primary_image(policy_input)
    wrist_mask = bool(policy_input.image_mask.get("image_rgb_1", False))
    wrist = policy_input.images.get("image_rgb_1", None)
    if wrist is None or (not wrist_mask):
        return np.zeros_like(primary, dtype=np.uint8)
    wrist_arr = np.asarray(wrist, dtype=np.uint8)
    if wrist_arr.shape != primary.shape:
        raise ValueError(
            "OpenPI wrist image must match primary image shape, got "
            f"{wrist_arr.shape} vs {primary.shape}"
        )
    return wrist_arr


def _resolve_dual_wrist_image(
    policy_input: PolicyInput,
    *,
    slot_name: str,
) -> np.ndarray:
    image = policy_input.images.get(slot_name, None)
    if image is None or (not bool(policy_input.image_mask.get(slot_name, False))):
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return np.asarray(image, dtype=np.uint8)


def build_openpi_request(policy_input: PolicyInput) -> Dict[str, Any]:
    openpi_layout = str(policy_input.metadata.get("openpi_layout", "")).strip().lower()
    if openpi_layout == "dual_wrist":
        return {
            "observation/image": _resolve_primary_image(policy_input),
            "observation/wrist_left_image": _resolve_dual_wrist_image(
                policy_input,
                slot_name="image_rgb_1",
            ),
            "observation/wrist_right_image": _resolve_dual_wrist_image(
                policy_input,
                slot_name="image_rgb_2",
            ),
            "observation/state": np.asarray(policy_input.state, dtype=np.float32),
            "prompt": str(policy_input.prompt),
        }
    return {
        "observation/image": _resolve_primary_image(policy_input),
        "observation/wrist_image": _resolve_wrist_image(policy_input),
        "observation/state": np.asarray(policy_input.state, dtype=np.float32),
        "prompt": str(policy_input.prompt),
    }


def build_openpi_batch_request(
    policy_inputs: Sequence[PolicyInput],
) -> Dict[str, Any]:
    if not policy_inputs:
        raise ValueError("policy_inputs must be non-empty for OpenPI batch requests")
    return {
        "examples": [build_openpi_request(policy_input) for policy_input in policy_inputs]
    }
