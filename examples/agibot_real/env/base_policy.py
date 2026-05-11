"""Example-local base-policy adapter for AgiBot residual training."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from serl_launcher.policy.base import PolicyInferInfo
from serl_launcher.policy.typed_factory import describe_policy_backend
from serl_launcher.policy.typed_factory import resolve_policy_backend_type

from .observation import AGIBOT_ARM_STATE_DIM
from .observation import AGIBOT_RIGHT_ARM_STATE_SLICE
from .observation import AGIBOT_STATE_DIM
from .observation import build_agibot_state
from .policy_input import build_agibot_policy_input

JOYRA_RAW_ACTION_DIM = 18
POLICY_ACTION_LAYOUT_FULL = "full"
POLICY_ACTION_LAYOUT_RIGHT_ARM = "right_arm"


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


def _normalize_action_layout(action_layout: Any) -> str:
    layout = str(action_layout or POLICY_ACTION_LAYOUT_FULL).strip().lower()
    if layout in {"", POLICY_ACTION_LAYOUT_FULL, "dual_arm", "bimanual"}:
        return POLICY_ACTION_LAYOUT_FULL
    if layout in {
        POLICY_ACTION_LAYOUT_RIGHT_ARM,
        "right",
        "right_hand",
        "right_arm_camera_position",
        "right_hand_camera_position",
    }:
        return POLICY_ACTION_LAYOUT_RIGHT_ARM
    raise ValueError(
        "policy.action_layout must be one of "
        f"'{POLICY_ACTION_LAYOUT_FULL}' or '{POLICY_ACTION_LAYOUT_RIGHT_ARM}', "
        f"got {action_layout!r}"
    )


def _infer_action_layout_from_policy_id(policy_id: Any) -> str | None:
    if policy_id is None:
        return None
    token = str(policy_id).strip().lower()
    if "right" in token and ("hand" in token or "arm" in token):
        return POLICY_ACTION_LAYOUT_RIGHT_ARM
    return None


def _resolve_prompts(prompt: str | Sequence[str], count: int) -> list[str]:
    if isinstance(prompt, str):
        return [prompt] * int(count)
    prompts = [str(value) for value in prompt]
    if len(prompts) != int(count):
        raise ValueError(f"Expected {count} prompts for batched inference, got {len(prompts)}")
    return prompts


def _canonicalize_right_arm_action_chunk(
    *,
    raw_actions: np.ndarray,
    obs: dict[str, Any],
    backend_type: str,
    action_dim: int,
    chunk_horizon: int,
) -> np.ndarray:
    if int(action_dim) != AGIBOT_STATE_DIM:
        raise ValueError(
            "Right-arm OpenPI adaptation expects canonical AgiBot action_dim="
            f"{AGIBOT_STATE_DIM}, got {int(action_dim)}"
        )
    right_actions = _canonicalize_action_chunk(
        raw_actions=raw_actions,
        backend_type=backend_type,
        action_dim=AGIBOT_ARM_STATE_DIM,
        chunk_horizon=chunk_horizon,
    )
    current_state = build_agibot_state(obs)
    full_actions = np.repeat(
        current_state.reshape(1, AGIBOT_STATE_DIM),
        repeats=int(chunk_horizon),
        axis=0,
    ).astype(np.float32, copy=False)
    full_actions[:, AGIBOT_RIGHT_ARM_STATE_SLICE] = right_actions
    return np.asarray(full_actions, dtype=np.float32)


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
    _action_layout: str = POLICY_ACTION_LAYOUT_FULL

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

    @property
    def action_layout(self) -> str:
        return str(self._action_layout)

    def _policy_state_slice(self) -> slice | None:
        if self._action_layout == POLICY_ACTION_LAYOUT_RIGHT_ARM:
            return AGIBOT_RIGHT_ARM_STATE_SLICE
        return None

    def _canonicalize_client_actions(
        self,
        raw_actions: np.ndarray,
        *,
        obs: dict[str, Any],
    ) -> np.ndarray:
        if self._action_layout == POLICY_ACTION_LAYOUT_RIGHT_ARM:
            return _canonicalize_right_arm_action_chunk(
                raw_actions=raw_actions,
                obs=obs,
                backend_type=self._backend_type,
                action_dim=self._action_dim,
                chunk_horizon=self._chunk_horizon,
            )
        return _canonicalize_action_chunk(
            raw_actions=raw_actions,
            backend_type=self._backend_type,
            action_dim=self._action_dim,
            chunk_horizon=self._chunk_horizon,
        )

    def infer(
        self,
        obs: dict[str, Any],
        *,
        prompt: str,
    ) -> tuple[np.ndarray, PolicyInferInfo]:
        policy_input = build_agibot_policy_input(
            obs,
            prompt,
            state_slice=self._policy_state_slice(),
        )
        raw_actions, infer_info = self._client.infer(policy_input)
        canonical_actions = self._canonicalize_client_actions(raw_actions, obs=obs)
        info = dict(infer_info)
        info.update(
            {
                "backend_type": self._backend_type,
                "backend": self._description,
                "action_layout": self._action_layout,
                "canonical_action_dim": int(self._action_dim),
                "canonical_chunk_horizon": int(self._chunk_horizon),
                "raw_action_dim": int(np.asarray(raw_actions).shape[-1]),
            }
        )
        return canonical_actions, info

    def infer_many(
        self,
        observations: Sequence[dict[str, Any]],
        *,
        prompt: str | Sequence[str],
    ) -> tuple[list[np.ndarray], PolicyInferInfo]:
        observations_list = list(observations)
        if not observations_list:
            return [], {
                "backend_type": self._backend_type,
                "backend": self._description,
                "action_layout": self._action_layout,
                "batch_size": 0,
            }
        prompts = _resolve_prompts(prompt, len(observations_list))

        client_infer_many = getattr(self._client, "infer_many", None)
        if not callable(client_infer_many):
            chunks: list[np.ndarray] = []
            for obs, item_prompt in zip(observations_list, prompts, strict=True):
                action_chunk, _info = self.infer(obs, prompt=item_prompt)
                chunks.append(action_chunk)
            return chunks, {
                "backend_type": self._backend_type,
                "backend": self._description,
                "action_layout": self._action_layout,
                "canonical_action_dim": int(self._action_dim),
                "canonical_chunk_horizon": int(self._chunk_horizon),
                "batch_size": len(chunks),
                "fallback_serial": True,
            }

        policy_inputs = [
            build_agibot_policy_input(
                obs,
                item_prompt,
                state_slice=self._policy_state_slice(),
            )
            for obs, item_prompt in zip(observations_list, prompts, strict=True)
        ]
        raw_chunks, batch_info = client_infer_many(policy_inputs)
        if isinstance(raw_chunks, np.ndarray):
            if raw_chunks.ndim == 2 and len(observations_list) == 1:
                raw_chunks_list = [raw_chunks]
            elif raw_chunks.ndim == 3:
                raw_chunks_list = [raw_chunks[idx] for idx in range(int(raw_chunks.shape[0]))]
            else:
                raise ValueError(f"Expected batched raw action chunks, got shape {raw_chunks.shape}")
        else:
            raw_chunks_list = list(raw_chunks)
        if len(raw_chunks_list) != len(observations_list):
            raise ValueError(
                "Batched policy returned a different number of chunks than requested: "
                f"{len(raw_chunks_list)} != {len(observations_list)}"
            )

        canonical_chunks = [
            self._canonicalize_client_actions(raw_chunk, obs=obs)
            for raw_chunk, obs in zip(raw_chunks_list, observations_list, strict=True)
        ]
        info = dict(batch_info or {})
        info.update(
            {
                "backend_type": self._backend_type,
                "backend": self._description,
                "action_layout": self._action_layout,
                "canonical_action_dim": int(self._action_dim),
                "canonical_chunk_horizon": int(self._chunk_horizon),
                "raw_action_dim": int(np.asarray(raw_chunks_list[0]).shape[-1]),
                "batch_size": len(canonical_chunks),
            }
        )
        return canonical_chunks, info

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
    configured_layout = getattr(cfg.policy, "action_layout", POLICY_ACTION_LAYOUT_FULL)
    action_layout = _normalize_action_layout(configured_layout)
    if action_layout == POLICY_ACTION_LAYOUT_FULL:
        inferred_layout = _infer_action_layout_from_policy_id(
            getattr(cfg.policy, "id", None)
        )
        if backend_type == "openpi" and inferred_layout is not None:
            action_layout = inferred_layout
    if backend_type != "openpi" and action_layout != POLICY_ACTION_LAYOUT_FULL:
        raise ValueError(
            f"policy.action_layout={action_layout!r} is only supported for OpenPI"
        )

    if backend_type == "openpi":
        from serl_launcher.policy.openpi.client import OpenPIPolicyClient

        client = OpenPIPolicyClient(
            host=host,
            port=port,
            action_dim=(
                AGIBOT_ARM_STATE_DIM
                if action_layout == POLICY_ACTION_LAYOUT_RIGHT_ARM
                else action_dim
            ),
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
        _action_layout=action_layout,
    )


__all__ = [
    "AgiBotBasePolicy",
    "POLICY_ACTION_LAYOUT_FULL",
    "POLICY_ACTION_LAYOUT_RIGHT_ARM",
    "build_agibot_base_policy",
    "canonicalize_agibot_action_chunks",
]
