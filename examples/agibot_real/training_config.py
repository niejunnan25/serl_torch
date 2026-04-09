"""AgiBot residual-training data recipes."""
from __future__ import annotations

from typing import Any
from typing import Dict

import numpy as np

from serl_launcher.residual.data.action_projection import project_expert_action
from serl_launcher.residual.data.config import ResidualDataConfig
from serl_launcher.residual.data.config import register_residual_data_config
from serl_launcher.residual.data.recipe import build_common_repack_structure
from serl_launcher.residual.data.recipe import make_residual_data_config
from serl_launcher.residual.data.schema import LIBERO_RESIDUAL_TRAINING_SCHEMA
from serl_launcher.residual.data.transforms import delete_by_path
from serl_launcher.residual.data.transforms import get_by_path
from serl_launcher.residual.data.transforms import set_by_path

from .runtime.obs_adapter import build_agibot_state
from .runtime.obs_adapter import extract_residual_images
from .schema import AGIBOT_IMAGE_SLOT_KEYS

AGIBOT_TRAINING_IMAGE_VIEWS = {
    slot_key: slot_key for slot_key in AGIBOT_IMAGE_SLOT_KEYS
}

AGIBOT_IMAGE_RAW_MAPPING = {
    "head_image": "head_image",
    "left_wrist_image": "left_wrist_image",
    "right_wrist_image": "right_wrist_image",
}

AGIBOT_STATE_RAW_MAPPING = {
    "pose": "pose",
}

AGIBOT_OFFLINE_ACTION_MAPPING = {
    "base_chunks": "base_chunks",
    "expert": "actions",
    "alpha": "alpha",
}

AGIBOT_ONLINE_ACTION_MAPPING = {
    "base_chunks": "base_chunks",
    "final": "actions",
    "alpha": "alpha",
}


def _build_agibot_observation(data: Dict[str, Any]) -> Dict[str, Any]:
    image_raw = get_by_path(data, "observation/image_raw")
    state_raw = get_by_path(data, "observation/state_raw")

    head_image = np.asarray(image_raw["head_image"], dtype=np.uint8)
    left_wrist_image = np.asarray(image_raw["left_wrist_image"], dtype=np.uint8)
    right_wrist_image = np.asarray(image_raw["right_wrist_image"], dtype=np.uint8)
    pose = np.asarray(state_raw["pose"], dtype=np.float32)

    if head_image.ndim != 4 or left_wrist_image.ndim != 4 or right_wrist_image.ndim != 4:
        raise ValueError(
            "AgiBot training images must have rank 4 [T,H,W,C], got "
            f"head={head_image.shape} left={left_wrist_image.shape} right={right_wrist_image.shape}"
        )
    if int(head_image.shape[0]) != int(left_wrist_image.shape[0]) or int(head_image.shape[0]) != int(right_wrist_image.shape[0]):
        raise ValueError("AgiBot image streams must have the same temporal length")
    if pose.ndim != 2 or int(pose.shape[1]) != 14:
        raise ValueError(f"AgiBot pose tensor must have shape [T,14], got {pose.shape}")

    image_rgb_0 = []
    image_rgb_1 = []
    image_rgb_2 = []
    state = []
    num_steps = int(head_image.shape[0])
    for frame_idx in range(num_steps):
        obs_frame = {
            "image/head": head_image[frame_idx],
            "image/left_wrist": left_wrist_image[frame_idx],
            "image/right_wrist": right_wrist_image[frame_idx],
            "state/pose": pose[frame_idx],
        }
        images = extract_residual_images(obs_frame)
        image_rgb_0.append(np.asarray(images["image_rgb_0"], dtype=np.uint8))
        image_rgb_1.append(np.asarray(images["image_rgb_1"], dtype=np.uint8))
        image_rgb_2.append(np.asarray(images["image_rgb_2"], dtype=np.uint8))
        state.append(np.asarray(build_agibot_state(obs_frame), dtype=np.float32))

    set_by_path(data, "observation/image/image_rgb_0", np.asarray(image_rgb_0, dtype=np.uint8))
    set_by_path(data, "observation/image/image_rgb_1", np.asarray(image_rgb_1, dtype=np.uint8))
    set_by_path(data, "observation/image/image_rgb_2", np.asarray(image_rgb_2, dtype=np.uint8))
    set_by_path(
        data,
        "observation/image_mask",
        {
            "image_rgb_0": True,
            "image_rgb_1": True,
            "image_rgb_2": True,
        },
    )
    set_by_path(data, "observation/state", np.asarray(state, dtype=np.float32))
    delete_by_path(data, "observation/image_raw")
    delete_by_path(data, "observation/state_raw")
    return data


def _project_offline_expert_actions(data: Dict[str, Any]) -> Dict[str, Any]:
    expert_actions = np.asarray(get_by_path(data, "action/expert"), dtype=np.float32)
    base_chunks = np.asarray(get_by_path(data, "action/base_chunks"), dtype=np.float32)
    alpha = float(get_by_path(data, "action/alpha"))
    projection = dict(get_by_path(data, "metadata/projection"))

    control_indices = np.asarray(projection["control_indices"], dtype=np.int64)
    residual_limits = np.asarray(projection["residual_limits"], dtype=np.float32)
    expert_reference_scale = float(projection["expert_reference_scale"])
    clip_residual_to_unit = bool(projection["clip_residual_to_unit"])
    denom = residual_limits * alpha * expert_reference_scale
    chunk_horizon = int(base_chunks.shape[1])

    projected_actions = []
    clipped_total = 0
    for step_idx in range(expert_actions.shape[0]):
        chunk_start = int((step_idx // chunk_horizon) * chunk_horizon)
        chunk_index = int(chunk_start // chunk_horizon)
        step_in_chunk = int(step_idx - chunk_start)
        base_action = np.asarray(base_chunks[chunk_index][step_in_chunk], dtype=np.float32)
        expert_action = np.asarray(expert_actions[step_idx], dtype=np.float32).reshape(-1)
        if expert_action.shape[0] == 18:
            expert_action = expert_action[:14]
        projected_action, clipped_count = project_expert_action(
            expert_action=expert_action,
            base_action=base_action,
            control_indices=control_indices,
            denom=denom,
            clip_residual_to_unit=clip_residual_to_unit,
        )
        projected_actions.append(projected_action)
        clipped_total += int(clipped_count)

    projection["clipped_values"] = int(clipped_total)
    set_by_path(data, "metadata/projection", projection)
    set_by_path(data, "action/final", np.asarray(projected_actions, dtype=np.float32))
    return data


AGIBOT_OFFLINE_TRAINING_CONFIG = make_residual_data_config(
    name="agibot_offline_training",
    schema=LIBERO_RESIDUAL_TRAINING_SCHEMA,
    repack_structure=build_common_repack_structure(
        image_raw_mapping=AGIBOT_IMAGE_RAW_MAPPING,
        state_raw_mapping=AGIBOT_STATE_RAW_MAPPING,
        action_mapping=AGIBOT_OFFLINE_ACTION_MAPPING,
    ),
    image_views=AGIBOT_TRAINING_IMAGE_VIEWS,
    transforms=(
        _build_agibot_observation,
        _project_offline_expert_actions,
    ),
)


AGIBOT_ONLINE_TRAINING_CONFIG = make_residual_data_config(
    name="agibot_online_training",
    schema=LIBERO_RESIDUAL_TRAINING_SCHEMA,
    repack_structure=build_common_repack_structure(
        image_raw_mapping=AGIBOT_IMAGE_RAW_MAPPING,
        state_raw_mapping=AGIBOT_STATE_RAW_MAPPING,
        action_mapping=AGIBOT_ONLINE_ACTION_MAPPING,
    ),
    image_views=AGIBOT_TRAINING_IMAGE_VIEWS,
    transforms=(_build_agibot_observation,),
)


AGIBOT_RESIDUAL_BASE_CONFIG = register_residual_data_config(
    ResidualDataConfig(
        name="agibot_residual_training",
        schema=LIBERO_RESIDUAL_TRAINING_SCHEMA,
        image_views=AGIBOT_TRAINING_IMAGE_VIEWS,
    )
)

