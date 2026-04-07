"""Helpers for building residual-data recipes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from serl_launcher.residual.data.config import ResidualDataConfig
from serl_launcher.residual.data.config import register_residual_data_config
from serl_launcher.residual.data.schema import ResidualTrainingSchema
from serl_launcher.residual.data.transforms import CallableTransform, Group, RepackTransform

_DEFAULT_EPISODE_MAPPING = {
    "source": "source",
    "suite_name": "suite_name",
    "task_id": "task_id",
    "task_key": "task_key",
    "task_description": "task_description",
    "index": "episode_index",
    "steps": "episode_steps",
    "return": "episode_return",
    "success": "episode_success",
}

_DEFAULT_TRAJECTORY_MAPPING = {
    "rewards": "rewards",
    "dones": "dones",
}


def build_common_repack_structure(
    *,
    image_raw_mapping: Mapping[str, str],
    state_raw_mapping: Mapping[str, str],
    action_mapping: Mapping[str, str],
    prompt_source: str = "prompt",
    episode_mapping: Mapping[str, str] | None = None,
    trajectory_mapping: Mapping[str, str] | None = None,
    metadata_source: str = "metadata",
) -> dict[str, Any]:
    return {
        "prompt": str(prompt_source),
        "observation": {
            "image_raw": {str(key): str(value) for key, value in image_raw_mapping.items()},
            "state_raw": {str(key): str(value) for key, value in state_raw_mapping.items()},
        },
        "action": {str(key): str(value) for key, value in action_mapping.items()},
        "trajectory": {
            str(key): str(value)
            for key, value in (trajectory_mapping or _DEFAULT_TRAJECTORY_MAPPING).items()
        },
        "episode": {
            str(key): str(value)
            for key, value in (episode_mapping or _DEFAULT_EPISODE_MAPPING).items()
        },
        "metadata": str(metadata_source),
    }


def make_residual_data_config(
    *,
    name: str,
    schema: ResidualTrainingSchema,
    repack_structure: Mapping[str, Any],
    image_views: Mapping[str, str],
    transforms: Sequence[Any] = (),
    register: bool = True,
) -> ResidualDataConfig:
    data_config = ResidualDataConfig(
        name=str(name),
        schema=schema,
        repack_transforms=Group(inputs=(RepackTransform(dict(repack_structure)),)),
        data_transforms=Group(
            inputs=tuple(CallableTransform(transform) for transform in transforms)
        ),
        image_views={str(key): str(value) for key, value in image_views.items()},
    )
    if register:
        return register_residual_data_config(data_config)
    return data_config

