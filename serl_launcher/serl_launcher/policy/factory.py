"""Policy backend factory helpers used by residual runtime entrypoints."""
from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from serl_launcher.policy.base import PolicyClient
from serl_launcher.policy.base import PolicyPrefetcher
from serl_launcher.policy.openpi.client import OpenPIPolicyClient
from serl_launcher.policy.openpi.prefetch import AsyncOpenPIPolicyPrefetcher


def resolve_policy_backend_type(cfg: DictConfig) -> str:
    policy_cfg = cfg.get("policy", None)
    if policy_cfg is None:
        return "openpi"
    return str(policy_cfg.get("type", "openpi")).strip().lower()


def build_policy_client(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
) -> PolicyClient:
    policy_type = resolve_policy_backend_type(cfg)
    if policy_type == "openpi":
        return OpenPIPolicyClient(
            host=str(cfg.openpi.host),
            port=int(cfg.openpi.port),
            logger=logger,
        )
    raise ValueError(f"Unsupported policy backend type: {policy_type!r}")


def build_policy_prefetcher(
    cfg: DictConfig,
    *,
    logger: logging.Logger,
) -> PolicyPrefetcher:
    policy_type = resolve_policy_backend_type(cfg)
    if policy_type == "openpi":
        return AsyncOpenPIPolicyPrefetcher(
            host=str(cfg.openpi.host),
            port=int(cfg.openpi.port),
            logger=logger,
        )
    raise ValueError(f"Unsupported policy backend type: {policy_type!r}")


def build_policy_backend_info(cfg: DictConfig) -> dict[str, Any]:
    return {"type": resolve_policy_backend_type(cfg)}
