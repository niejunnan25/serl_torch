"""LIBERO data/task bindings used by learner and materialization entrypoints."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from serl_launcher.data.normalizer import StateActionNormalizer
from serl_launcher.data.normalizer import load_normalizer
from serl_launcher.residual.train.bindings import ResidualDataBindings

from ..config import resolve_libero_cfg_image_keys
from ..training_config import LIBERO_RESIDUAL_BASE_CONFIG


@dataclass
class LiberoDataBindings(ResidualDataBindings):
    image_keys: tuple[str, ...]
    normalizer: StateActionNormalizer | None
    task_key: str
    data_config: Any


def build_libero_data_bindings(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
) -> LiberoDataBindings:
    normalizer: StateActionNormalizer | None = None
    norm_cfg = cfg.get("normalization", None)
    task_key = f"{cfg.task.suite_name}_task_{int(cfg.task.task_id)}"
    if norm_cfg is not None and bool(norm_cfg.get("enabled", False)):
        stats_dir = norm_cfg.get(
            "stats_dir",
            str(Path(__file__).resolve().parents[1] / "data" / "stats"),
        )
        normalizer = load_normalizer(task_key, stats_dir=stats_dir)
        if normalizer is not None:
            logger.info("Loaded normalizer for task_key=%s", task_key)

    return LiberoDataBindings(
        image_keys=tuple(resolve_libero_cfg_image_keys(cfg)),
        normalizer=normalizer,
        task_key=task_key,
        data_config=LIBERO_RESIDUAL_BASE_CONFIG,
    )
