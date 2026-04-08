"""LIBERO-specific config parsing helpers."""
from __future__ import annotations

from omegaconf import DictConfig

from .schema import resolve_libero_image_keys


def resolve_libero_cfg_image_keys(cfg: DictConfig) -> tuple[str, ...]:
    image_keys_cfg = cfg.residual.get("image_keys", None)
    source = image_keys_cfg if image_keys_cfg is not None else cfg.sac.image_keys
    return resolve_libero_image_keys(str(k) for k in source)
