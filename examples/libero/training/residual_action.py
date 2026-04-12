from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig


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

        if int(self.full_action_dim) <= 0:
            raise ValueError(
                f"full_action_dim must be positive, got {self.full_action_dim}"
            )
        if int(self.chunk_horizon) <= 0:
            raise ValueError(
                f"chunk_horizon must be positive, got {self.chunk_horizon}"
            )
        if int(self.control_indices.size) <= 0:
            raise ValueError("control_indices must not be empty")
        if int(self.control_indices.size) != int(self.residual_limits.size):
            raise ValueError(
                "control_indices and residual_limits size mismatch: "
                f"{self.control_indices.size} != {self.residual_limits.size}"
            )
        if np.any(self.control_indices < 0) or np.any(
            self.control_indices >= int(self.full_action_dim)
        ):
            raise ValueError(
                "control_indices out of range for full_action_dim: "
                f"indices={self.control_indices.tolist()} full_action_dim={self.full_action_dim}"
            )
        if (not np.isfinite(float(self.alpha))) or float(self.alpha) < 0.0:
            raise ValueError(
                f"alpha must be finite and >= 0.0, got {self.alpha!r}"
            )

    @classmethod
    def from_cfg(
        cls,
        cfg: DictConfig,
        *,
        action_dim: int,
    ) -> "ResidualActionSpec":
        action_mask_cfg = cfg.residual.get("action_mask", None)
        if action_mask_cfg is None:
            control_indices = np.arange(int(action_dim), dtype=np.int64)
        else:
            control_indices = np.flatnonzero(
                np.asarray([bool(v) for v in action_mask_cfg], dtype=bool)
            ).astype(np.int64)
        residual_limits = np.asarray(
            cfg.residual.action_limits,
            dtype=np.float32,
        ).reshape(-1)[control_indices]
        return cls(
            full_action_dim=int(action_dim),
            control_indices=control_indices,
            residual_limits=residual_limits,
            alpha=float(cfg.residual.alpha),
            clip_gripper=bool(cfg.residual.get("clip_gripper", True)),
            chunk_horizon=int(cfg.residual.get("chunk_horizon", 1)),
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

    def build_action_transform(self) -> dict[str, object]:
        return {
            "type": "residual_combined",
            "control_indices": [int(v) for v in self.control_indices],
            "limits": [float(v) for v in self.residual_limits],
            "full_action_dim": int(self.full_action_dim),
            "chunk_horizon": 1,
            "chunk_step_enabled": False,
            "clip_gripper": bool(self.clip_gripper),
            "base_action_key": "base_action",
            "base_action_chunk_key": "base_action_chunk",
            "scale_key": "alpha",
        }

    def build_chunk_action_transform(self) -> dict[str, object]:
        return {
            "type": "residual_combined",
            "control_indices": [int(v) for v in self.control_indices],
            "limits": [float(v) for v in self.residual_limits],
            "full_action_dim": int(self.full_action_dim),
            "chunk_horizon": int(self.chunk_horizon),
            "chunk_step_enabled": True,
            "clip_gripper": bool(self.clip_gripper),
            "base_action_key": "base_action",
            "base_action_chunk_key": "base_action_chunk",
            "scale_key": "alpha",
        }

    def compose(
        self,
        *,
        base_action: np.ndarray,
        residual_action: np.ndarray,
    ) -> np.ndarray:
        base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
        residual_action_arr = np.asarray(residual_action, dtype=np.float32).reshape(-1)
        if int(base_action_arr.size) != int(self.full_action_dim):
            raise ValueError(
                "Unexpected base_action size: "
                f"{base_action_arr.shape} vs ({int(self.full_action_dim)},)"
            )
        if int(residual_action_arr.size) != int(self.policy_action_dim):
            raise ValueError(
                "Unexpected residual_action size: "
                f"{residual_action_arr.shape} vs ({int(self.policy_action_dim)},)"
            )
        clipped_residual = np.clip(residual_action_arr, -1.0, 1.0)
        delta_action = np.zeros((self.full_action_dim,), dtype=np.float32)
        delta_action[self.control_indices] = (
            clipped_residual * float(self.alpha) * self.residual_limits
        )
        final_action = base_action_arr + delta_action
        if self.clip_gripper and final_action.size > 0:
            final_action[-1] = np.clip(final_action[-1], -1.0, 1.0)
        return np.asarray(final_action, dtype=np.float32)

    def compose_chunk(
        self,
        *,
        base_action_chunk: np.ndarray,
        residual_action: np.ndarray,
    ) -> np.ndarray:
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
        if base_chunk_arr.ndim != 2 or base_chunk_arr.shape != (
            int(self.chunk_horizon),
            int(self.full_action_dim),
        ):
            raise ValueError(
                "Unexpected base_action_chunk shape: "
                f"{base_chunk_arr.shape} vs ({int(self.chunk_horizon)}, {int(self.full_action_dim)})"
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
        if residual_arr.ndim != 2 or residual_arr.shape != (
            int(self.chunk_horizon),
            int(self.policy_action_dim),
        ):
            raise ValueError(
                "Unexpected residual_action chunk shape: "
                f"{residual_arr.shape} vs ({int(self.chunk_horizon)}, {int(self.policy_action_dim)})"
            )

        clipped_residual = np.clip(residual_arr, -1.0, 1.0)
        delta_chunk = np.zeros(
            (int(self.chunk_horizon), int(self.full_action_dim)),
            dtype=np.float32,
        )
        delta_chunk[:, self.control_indices] = (
            clipped_residual * float(self.alpha) * self.residual_limits.reshape(1, -1)
        )
        final_chunk = base_chunk_arr + delta_chunk
        if self.clip_gripper and final_chunk.shape[-1] > 0:
            final_chunk[:, -1] = np.clip(final_chunk[:, -1], -1.0, 1.0)
        return np.asarray(final_chunk, dtype=np.float32)
