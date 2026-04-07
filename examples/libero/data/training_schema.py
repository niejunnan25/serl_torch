"""LIBERO-specific conversion into the unified residual training schema."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..policy.observation import extract_residual_images


def build_libero_training_arrays(
    *,
    agentview_rgb: np.ndarray,
    eye_in_hand_rgb: np.ndarray,
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_states: np.ndarray,
) -> Dict[str, Any]:
    agentview_arr = np.asarray(agentview_rgb, dtype=np.uint8)
    eye_in_hand_arr = np.asarray(eye_in_hand_rgb, dtype=np.uint8)
    ee_pos_arr = np.asarray(ee_pos, dtype=np.float32)
    ee_ori_arr = np.asarray(ee_ori, dtype=np.float32)
    gripper_arr = np.asarray(gripper_states, dtype=np.float32)

    if agentview_arr.ndim != 4 or eye_in_hand_arr.ndim != 4:
        raise ValueError(
            "LIBERO training images must have rank 4 [T,H,W,C], got "
            f"agentview={agentview_arr.shape} eye_in_hand={eye_in_hand_arr.shape}"
        )
    if int(agentview_arr.shape[0]) != int(eye_in_hand_arr.shape[0]):
        raise ValueError(
            "LIBERO image streams must have the same length, got "
            f"{int(agentview_arr.shape[0])} and {int(eye_in_hand_arr.shape[0])}"
        )

    num_steps = int(agentview_arr.shape[0])
    processed_image = []
    processed_wrist = []
    for frame_idx in range(num_steps):
        images = extract_residual_images(
            {
                "agentview_rgb": agentview_arr[frame_idx],
                "eye_in_hand_rgb": eye_in_hand_arr[frame_idx],
            }
        )
        processed_image.append(np.asarray(images["image"], dtype=np.uint8))
        processed_wrist.append(np.asarray(images["wrist_image"], dtype=np.uint8))

    image_rgb_0 = np.asarray(processed_image, dtype=np.uint8)
    image_rgb_1 = np.asarray(processed_wrist, dtype=np.uint8)
    image_rgb_2 = np.zeros_like(image_rgb_0)
    state = np.concatenate(
        (
            ee_pos_arr.astype(np.float32),
            ee_ori_arr.astype(np.float32),
            gripper_arr.astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)

    return {
        "image_rgb_0": image_rgb_0,
        "image_rgb_1": image_rgb_1,
        "image_rgb_2": image_rgb_2,
        "image_mask": {
            "image_rgb_0": True,
            "image_rgb_1": True,
            "image_rgb_2": False,
        },
        "state": state,
    }
