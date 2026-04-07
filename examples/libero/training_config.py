"""LIBERO residual-training data recipes."""
from __future__ import annotations

from typing import Any, Dict

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

from .runtime.obs_adapter import build_libero_state, extract_residual_images
from .schema import LIBERO_IMAGE_SLOT_KEYS

LIBERO_TRAINING_IMAGE_VIEWS = {
    slot_key: slot_key for slot_key in LIBERO_IMAGE_SLOT_KEYS
}

LIBERO_IMAGE_RAW_MAPPING = {
    "agentview_rgb": "agentview_rgb",
    "eye_in_hand_rgb": "eye_in_hand_rgb",
}

LIBERO_STATE_RAW_MAPPING = {
    "ee_pos": "ee_pos",
    "ee_ori": "ee_ori",
    "gripper_states": "gripper_states",
}

LIBERO_OFFLINE_ACTION_MAPPING = {
    "base_chunks": "base_chunks",
    "expert": "actions",
    "alpha": "alpha",
}

LIBERO_ONLINE_ACTION_MAPPING = {
    "base_chunks": "base_chunks",
    "final": "actions",
    "alpha": "alpha",
}


def _build_libero_observation(data: Dict[str, Any]) -> Dict[str, Any]:
    image_raw = get_by_path(data, "observation/image_raw")
    state_raw = get_by_path(data, "observation/state_raw")

    agentview_rgb = np.asarray(image_raw["agentview_rgb"], dtype=np.uint8)
    eye_in_hand_rgb = np.asarray(image_raw["eye_in_hand_rgb"], dtype=np.uint8)
    ee_pos = np.asarray(state_raw["ee_pos"], dtype=np.float32)
    ee_ori = np.asarray(state_raw["ee_ori"], dtype=np.float32)
    gripper_states = np.asarray(state_raw["gripper_states"], dtype=np.float32)

    if agentview_rgb.ndim != 4 or eye_in_hand_rgb.ndim != 4:
        raise ValueError(
            "LIBERO training images must have rank 4 [T,H,W,C], got "
            f"agentview={agentview_rgb.shape} eye_in_hand={eye_in_hand_rgb.shape}"
        )
    if int(agentview_rgb.shape[0]) != int(eye_in_hand_rgb.shape[0]):
        raise ValueError(
            "LIBERO image streams must have the same length, got "
            f"{int(agentview_rgb.shape[0])} and {int(eye_in_hand_rgb.shape[0])}"
        )

    image_rgb_0 = []
    image_rgb_1 = []
    image_rgb_2 = []
    state = []
    num_steps = int(agentview_rgb.shape[0])
    for frame_idx in range(num_steps):
        obs_frame = {
            "agentview_rgb": agentview_rgb[frame_idx],
            "eye_in_hand_rgb": eye_in_hand_rgb[frame_idx],
            "ee_pos": ee_pos[frame_idx],
            "ee_ori": ee_ori[frame_idx],
            "gripper_states": gripper_states[frame_idx],
        }
        images = extract_residual_images(obs_frame)
        image_rgb_0.append(np.asarray(images["image_rgb_0"], dtype=np.uint8))
        image_rgb_1.append(np.asarray(images["image_rgb_1"], dtype=np.uint8))
        image_rgb_2.append(np.asarray(images["image_rgb_2"], dtype=np.uint8))
        state.append(np.asarray(build_libero_state(obs_frame), dtype=np.float32))

    set_by_path(
        data,
        "observation/image/image_rgb_0",
        np.asarray(image_rgb_0, dtype=np.uint8),
    )
    set_by_path(
        data,
        "observation/image/image_rgb_1",
        np.asarray(image_rgb_1, dtype=np.uint8),
    )
    set_by_path(
        data,
        "observation/image/image_rgb_2",
        np.asarray(image_rgb_2, dtype=np.uint8),
    )
    set_by_path(
        data,
        "observation/image_mask",
        {
            "image_rgb_0": True,
            "image_rgb_1": True,
            "image_rgb_2": False,
        },
    )
    set_by_path(
        data,
        "observation/state",
        np.asarray(state, dtype=np.float32),
    )
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
        projected_action, clipped_count = project_expert_action(
            expert_action=np.asarray(expert_actions[step_idx], dtype=np.float32),
            base_action=base_action,
            control_indices=control_indices,
            denom=denom,
            clip_residual_to_unit=clip_residual_to_unit,
        )
        projected_actions.append(projected_action)
        clipped_total += int(clipped_count)

    projection["clipped_values"] = int(clipped_total)
    set_by_path(data, "metadata/projection", projection)
    set_by_path(
        data,
        "action/final",
        np.asarray(projected_actions, dtype=np.float32),
    )
    return data


LIBERO_OFFLINE_TRAINING_CONFIG = make_residual_data_config(
    name="libero_offline_training",
    schema=LIBERO_RESIDUAL_TRAINING_SCHEMA,
    repack_structure=build_common_repack_structure(
        image_raw_mapping=LIBERO_IMAGE_RAW_MAPPING,
        state_raw_mapping=LIBERO_STATE_RAW_MAPPING,
        action_mapping=LIBERO_OFFLINE_ACTION_MAPPING,
    ),
    image_views=LIBERO_TRAINING_IMAGE_VIEWS,
    transforms=(
        _build_libero_observation,
        _project_offline_expert_actions,
    ),
)


LIBERO_ONLINE_TRAINING_CONFIG = make_residual_data_config(
    name="libero_online_training",
    schema=LIBERO_RESIDUAL_TRAINING_SCHEMA,
    repack_structure=build_common_repack_structure(
        image_raw_mapping=LIBERO_IMAGE_RAW_MAPPING,
        state_raw_mapping=LIBERO_STATE_RAW_MAPPING,
        action_mapping=LIBERO_ONLINE_ACTION_MAPPING,
    ),
    image_views=LIBERO_TRAINING_IMAGE_VIEWS,
    transforms=(_build_libero_observation,),
)


LIBERO_RESIDUAL_BASE_CONFIG = register_residual_data_config(ResidualDataConfig(
    name="libero_residual_training",
    schema=LIBERO_RESIDUAL_TRAINING_SCHEMA,
    image_views=LIBERO_TRAINING_IMAGE_VIEWS,
))
