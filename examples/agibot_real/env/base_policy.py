"""Example-local base-policy adapter for AgiBot residual training."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from serl_launcher.policy.base import PolicyInferInfo
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.policy.typed_factory import resolve_policy_backend_type

from .policy_input import build_agibot_policy_input

JOYRA_RAW_ACTION_DIM = 18


def _canonicalize_action_chunk(
    *,
    raw_actions: np.ndarray,
    backend_type: str,
    action_dim: int,
    chunk_horizon: int,
) -> np.ndarray:
    raw_actions_array = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions_array.ndim != 2:
        raise ValueError(
            f"{backend_type} policy must return a 2D action chunk, got shape "
            f"{raw_actions_array.shape}"
        )
    if int(raw_actions_array.shape[0]) < int(chunk_horizon):
        raise ValueError(
            f"{backend_type} policy returned only {int(raw_actions_array.shape[0])} actions, "
            f"expected at least chunk_horizon={int(chunk_horizon)}"
        )
    if int(raw_actions_array.shape[1]) < int(action_dim):
        raise ValueError(
            f"{backend_type} policy returned action_dim={int(raw_actions_array.shape[1])}, "
            f"smaller than canonical action_dim={int(action_dim)}"
        )
    if backend_type == "joyra" and int(raw_actions_array.shape[1]) < JOYRA_RAW_ACTION_DIM:
        raise ValueError(
            "JoyRA policy must expose its raw 18D action chunk before AgiBot canonicalization, "
            f"got shape {raw_actions_array.shape}"
        )
    return np.asarray(
        raw_actions_array[: int(chunk_horizon), : int(action_dim)],
        dtype=np.float32,
    )


def canonicalize_agibot_action_chunks(
    *,
    raw_actions: Sequence[np.ndarray] | np.ndarray,
    backend_type: str,
    action_dim: int,
    chunk_horizon: int,
) -> list[np.ndarray]:
    raw_actions_array = np.asarray(raw_actions, dtype=np.float32)
    if raw_actions_array.ndim == 2:
        return [
            _canonicalize_action_chunk(
                raw_actions=raw_actions_array,
                backend_type=backend_type,
                action_dim=action_dim,
                chunk_horizon=chunk_horizon,
            )
        ]
    if raw_actions_array.ndim != 3:
        raise ValueError(
            f"Expected raw batched action chunks to be rank-2 or rank-3, got {raw_actions_array.shape}"
        )
    return [
        _canonicalize_action_chunk(
            raw_actions=raw_actions_array[idx],
            backend_type=backend_type,
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
        )
        for idx in range(int(raw_actions_array.shape[0]))
    ]


@dataclass(slots=True)
class AgiBotBasePolicy:
    """Backend adapter that always exposes canonical AgiBot 14D action chunks.

    当前这层逻辑仍然保留在 example 内部，而不是迁到 shared policy factory。
    原因是 AgiBot 真实机器人目前需要在这里把 OpenPI/JoyRA 的输入输出
    统一到同一个 canonical 14D dual-arm action chunk 约定，属于 example-
    local 语义，不适合在第一轮对齐里直接抽成共享基础设施。
    """

    _client: Any
    _backend_type: str
    _description: str
    _action_dim: int
    _chunk_horizon: int

    @property
    def client(self) -> Any:
        return self._client

    @property
    def backend_type(self) -> str:
        return self._backend_type

    @property
    def action_dim(self) -> int:
        return int(self._action_dim)

    @property
    def chunk_horizon(self) -> int:
        return int(self._chunk_horizon)

    def infer(
        self,
        obs: dict[str, Any],
        *,
        prompt: str,
    ) -> tuple[np.ndarray, PolicyInferInfo]:
        policy_input = build_agibot_policy_input(obs, prompt)
        raw_actions, infer_info = self._client.infer(policy_input)
        canonical_actions = _canonicalize_action_chunk(
            raw_actions=raw_actions,
            backend_type=self._backend_type,
            action_dim=self._action_dim,
            chunk_horizon=self._chunk_horizon,
        )
        info = dict(infer_info)
        info.update(
            {
                "backend_type": self._backend_type,
                "backend": self._description,
                "canonical_action_dim": int(self._action_dim),
                "canonical_chunk_horizon": int(self._chunk_horizon),
                "raw_action_dim": int(np.asarray(raw_actions).shape[-1]),
            }
        )
        return canonical_actions, info

    def describe(self) -> str:
        return self._description

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()


def build_agibot_base_policy(
    cfg: Any,
    *,
    logger: logging.Logger,
    host: str | None = None,
    port: int | None = None,
) -> AgiBotBasePolicy:
    backend_type = resolve_policy_backend_type(cfg)
    description = describe_policy_backend(cfg)
    host = str(cfg.policy.host) if host is None else str(host)
    port = int(cfg.policy.port) if port is None else int(port)
    action_dim = int(cfg.env.action_dim)
    chunk_horizon = int(cfg.residual.chunk_horizon)

    if backend_type == "openpi":
        from serl_launcher.policy.openpi.client import OpenPIPolicyClient

        client = OpenPIPolicyClient(
            host=host,
            port=port,
            action_dim=action_dim,
            logger=logger,
        )
    elif backend_type == "joyra":
        from serl_launcher.policy.joyra.client import JoyRAPolicyClient

        client = JoyRAPolicyClient(
            host=host,
            port=port,
            action_dim=JOYRA_RAW_ACTION_DIM,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported AgiBot base policy backend: {backend_type!r}")

    return AgiBotBasePolicy(
        _client=client,
        _backend_type=backend_type,
        _description=description,
        _action_dim=action_dim,
        _chunk_horizon=chunk_horizon,
    )


__all__ = [
    "AgiBotBasePolicy",
    "build_agibot_base_policy",
    "canonicalize_agibot_action_chunks",
]
