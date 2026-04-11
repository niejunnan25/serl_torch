"""AgiBot data/task bindings used by learner and materialization entrypoints."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig
from serl_launcher.residual.train.bindings import ResidualDataBindings

from ..config import resolve_agibot_cfg_image_keys
from ..config import resolve_agibot_cfg_task_key
from ..training_config import AGIBOT_RESIDUAL_BASE_CONFIG


@dataclass
class AgiBotDataBindings(ResidualDataBindings):
    image_keys: tuple[str, ...]
    task_key: str
    data_config: Any


def build_agibot_data_bindings(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
) -> AgiBotDataBindings:
    task_key = resolve_agibot_cfg_task_key(cfg)

    return AgiBotDataBindings(
        image_keys=tuple(resolve_agibot_cfg_image_keys(cfg)),
        task_key=task_key,
        data_config=AGIBOT_RESIDUAL_BASE_CONFIG,
    )
