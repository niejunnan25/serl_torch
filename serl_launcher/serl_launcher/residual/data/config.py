"""Declarative residual-data configs inspired by OpenPI's DataConfig."""
from __future__ import annotations

import dataclasses
import difflib
from typing import Mapping

from serl_launcher.residual.data.schema import ResidualTrainingSchema
from serl_launcher.residual.data.transforms import Group


@dataclasses.dataclass(frozen=True)
class ResidualDataConfig:
    """Configures how raw episodes are repacked into residual-training payloads."""

    name: str
    schema: ResidualTrainingSchema
    repack_transforms: Group = dataclasses.field(default_factory=Group)
    data_transforms: Group = dataclasses.field(default_factory=Group)
    image_views: Mapping[str, str] = dataclasses.field(default_factory=dict)


_CONFIGS: dict[str, ResidualDataConfig] = {}


def register_residual_data_config(config: ResidualDataConfig) -> ResidualDataConfig:
    if config.name in _CONFIGS:
        raise ValueError(f"Residual data config already registered: {config.name!r}")
    _CONFIGS[config.name] = config
    return config


def get_residual_data_config(name: str) -> ResidualDataConfig:
    if name not in _CONFIGS:
        closest = difflib.get_close_matches(name, _CONFIGS.keys(), n=1, cutoff=0.0)
        suffix = f". Closest match: {closest[0]!r}" if closest else ""
        raise KeyError(f"Unknown residual data config {name!r}{suffix}")
    return _CONFIGS[name]


def list_residual_data_configs() -> tuple[str, ...]:
    return tuple(sorted(_CONFIGS))
