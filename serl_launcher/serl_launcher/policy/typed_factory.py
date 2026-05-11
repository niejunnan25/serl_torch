"""Typed-config policy backend helpers for the current residual training flow."""
from __future__ import annotations

import logging
from typing import Any

from serl_launcher.policy.base import PolicyClient
from serl_launcher.policy.base import PolicyPrefetcher


def resolve_policy_backend_type(cfg: Any) -> str:
    policy_type = str(cfg.policy.type).strip().lower()
    return policy_type if policy_type else "openpi"


def resolve_policy_backend_id(cfg: Any) -> str:
    policy_type = resolve_policy_backend_type(cfg)
    policy_id_value = getattr(cfg.policy, "id", None)
    if policy_id_value is None:
        return policy_type
    policy_id = str(policy_id_value).strip()
    return policy_id if policy_id else policy_type


def describe_policy_backend(cfg: Any) -> str:
    policy_type = resolve_policy_backend_type(cfg)
    policy_id = resolve_policy_backend_id(cfg)
    return policy_type if policy_id == policy_type else f"{policy_type}:{policy_id}"


def _resolve_policy_endpoint(cfg: Any) -> tuple[str, int]:
    return str(cfg.policy.host), int(cfg.policy.port)


def build_policy_client(
    cfg: Any,
    *,
    logger: logging.Logger,
) -> PolicyClient:
    policy_type = resolve_policy_backend_type(cfg)
    host, port = _resolve_policy_endpoint(cfg)
    action_dim = int(cfg.env.action_dim)
    if policy_type == "openpi":
        from serl_launcher.policy.openpi.client import OpenPIPolicyClient

        return OpenPIPolicyClient(
            host=host,
            port=port,
            action_dim=action_dim,
            logger=logger,
        )
    if policy_type == "joyra":
        from serl_launcher.policy.joyra.client import JoyRAPolicyClient

        return JoyRAPolicyClient(
            host=host,
            port=port,
            action_dim=action_dim,
            logger=logger,
        )
    raise ValueError(f"Unsupported policy backend type: {policy_type!r}")


def build_policy_prefetcher(
    cfg: Any,
    *,
    logger: logging.Logger,
) -> PolicyPrefetcher:
    policy_type = resolve_policy_backend_type(cfg)
    host, port = _resolve_policy_endpoint(cfg)
    action_dim = int(cfg.env.action_dim)
    if policy_type == "openpi":
        from serl_launcher.policy.openpi.prefetch import AsyncOpenPIPolicyPrefetcher

        return AsyncOpenPIPolicyPrefetcher(
            host=host,
            port=port,
            action_dim=action_dim,
            logger=logger,
        )
    if policy_type == "joyra":
        from serl_launcher.policy.joyra.prefetch import AsyncJoyRAPolicyPrefetcher

        return AsyncJoyRAPolicyPrefetcher(
            host=host,
            port=port,
            action_dim=action_dim,
            logger=logger,
        )
    raise ValueError(f"Unsupported policy backend type: {policy_type!r}")
