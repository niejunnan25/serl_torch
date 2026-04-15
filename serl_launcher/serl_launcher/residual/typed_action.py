"""Typed-config residual action helpers for the current residual training flow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from serl_launcher.residual.action import ResidualActionTransform
from serl_launcher.residual.action_spec import build_residual_limits
from serl_launcher.residual.action_spec import resolve_control_indices


@dataclass(frozen=True)
class ResidualActionSpec:
    full_action_dim: int
    control_indices: np.ndarray
    residual_limits: np.ndarray
    alpha: float
    clip_gripper: bool
    chunk_horizon: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "full_action_dim", int(self.full_action_dim))
        object.__setattr__(
            self,
            "control_indices",
            np.asarray(self.control_indices, dtype=np.int64).reshape(-1),
        )
        object.__setattr__(
            self,
            "residual_limits",
            np.asarray(self.residual_limits, dtype=np.float32).reshape(-1),
        )
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "clip_gripper", bool(self.clip_gripper))
        object.__setattr__(self, "chunk_horizon", int(self.chunk_horizon))

        if self.full_action_dim <= 0:
            raise ValueError(
                f"full_action_dim must be positive, got {self.full_action_dim}"
            )
        if self.chunk_horizon <= 0:
            raise ValueError(
                f"chunk_horizon must be positive, got {self.chunk_horizon}"
            )
        if self.control_indices.size <= 0:
            raise ValueError("control_indices must not be empty")
        if self.control_indices.size != self.residual_limits.size:
            raise ValueError(
                "control_indices and residual_limits size mismatch: "
                f"{self.control_indices.size} != {self.residual_limits.size}"
            )
        if np.any(self.control_indices < 0) or np.any(
            self.control_indices >= self.full_action_dim
        ):
            raise ValueError(
                "control_indices out of range for full_action_dim: "
                f"indices={self.control_indices.tolist()} full_action_dim={self.full_action_dim}"
            )
        if (not np.isfinite(self.alpha)) or self.alpha < 0.0:
            raise ValueError(f"alpha must be finite and >= 0.0, got {self.alpha!r}")

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        action_dim: int,
    ) -> "ResidualActionSpec":
        control_indices = resolve_control_indices(
            full_action_dim=int(action_dim),
            action_mask=cfg.residual.action_mask,
        )
        residual_limits = build_residual_limits(
            control_indices,
            full_action_dim=int(action_dim),
            action_limits=cfg.residual.action_limits,
        )
        return cls(
            full_action_dim=action_dim,
            control_indices=control_indices,
            residual_limits=residual_limits,
            alpha=cfg.residual.alpha,
            clip_gripper=cfg.residual.clip_gripper,
            chunk_horizon=cfg.residual.chunk_horizon,
        )

    @property
    def policy_action_dim(self) -> int:
        return int(self.control_indices.shape[0])

    @property
    def chunk_policy_action_dim(self) -> int:
        return int(self.chunk_horizon * self.policy_action_dim)

    @property
    def chunk_critic_action_dim(self) -> int:
        return int(self.chunk_horizon * self.full_action_dim)

    def build_action_transform(self) -> ResidualActionTransform:
        return ResidualActionTransform(
            control_indices=self.control_indices,
            limits=self.residual_limits,
            full_action_dim=self.full_action_dim,
            chunk_horizon=1,
            chunk_step_enabled=False,
            clip_gripper=self.clip_gripper,
        )

    def build_chunk_action_transform(self) -> ResidualActionTransform:
        return ResidualActionTransform(
            control_indices=self.control_indices,
            limits=self.residual_limits,
            full_action_dim=self.full_action_dim,
            chunk_horizon=self.chunk_horizon,
            chunk_step_enabled=True,
            clip_gripper=self.clip_gripper,
        )

    def compose(
        self,
        *,
        base_action: np.ndarray,
        residual_action: np.ndarray,
    ) -> np.ndarray:
        _delta, final_action = self.build_action_transform().compose(
            base_action=base_action,
            residual_action=residual_action,
            alpha=self.alpha,
        )
        return np.asarray(final_action, dtype=np.float32)

    def compose_chunk(
        self,
        *,
        base_action_chunk: np.ndarray,
        residual_action: np.ndarray,
    ) -> np.ndarray:
        transform = self.build_chunk_action_transform()
        base_chunk_arr = np.asarray(base_action_chunk, dtype=np.float32)
        if base_chunk_arr.ndim == 1:
            expected_dim = int(self.chunk_critic_action_dim)
            if int(base_chunk_arr.size) != expected_dim:
                raise ValueError(
                    "Unexpected flat base_action_chunk size: "
                    f"{base_chunk_arr.shape} vs ({expected_dim},)"
                )
            base_chunk_arr = base_chunk_arr.reshape(
                int(self.chunk_horizon),
                int(self.full_action_dim),
            )

        residual_arr = np.asarray(residual_action, dtype=np.float32)
        if residual_arr.ndim == 1:
            expected_dim = int(self.chunk_policy_action_dim)
            if int(residual_arr.size) != expected_dim:
                raise ValueError(
                    "Unexpected flat residual_action size: "
                    f"{residual_arr.shape} vs ({expected_dim},)"
                )
            residual_arr = residual_arr.reshape(
                int(self.chunk_horizon),
                int(self.policy_action_dim),
            )

        _delta_chunk, final_chunk = transform.compose_chunk(
            base_chunk=base_chunk_arr,
            residual_chunk=residual_arr,
            alpha=self.alpha,
        )
        return np.asarray(final_chunk, dtype=np.float32)
