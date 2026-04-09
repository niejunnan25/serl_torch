"""AgiBot-specific config parsing helpers."""
from __future__ import annotations

from omegaconf import DictConfig

from .schema import build_agibot_task_key
from .schema import resolve_agibot_image_keys


def resolve_agibot_cfg_image_keys(cfg: DictConfig) -> tuple[str, ...]:
    image_keys_cfg = cfg.residual.get("image_keys", None)
    source = image_keys_cfg if image_keys_cfg is not None else cfg.sac.image_keys
    return resolve_agibot_image_keys(str(k) for k in source)


def resolve_agibot_cfg_task_key(cfg: DictConfig) -> str:
    task_cfg = cfg.get("task", {})
    explicit_task_key = task_cfg.get("task_key", None)
    if explicit_task_key is not None:
        task_key = str(explicit_task_key).strip()
        if task_key:
            return task_key
    return build_agibot_task_key(task_cfg.get("name", "default"))

