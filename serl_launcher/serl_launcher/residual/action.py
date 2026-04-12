"""Residual action helpers shared across environments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch


def _validate_residual_alpha(alpha: float) -> float:
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value):
        raise ValueError(f"Residual alpha must be finite, got {alpha!r}")
    if alpha_value < 0.0:
        raise ValueError(f"Residual alpha must be >= 0, got {alpha_value}")
    return alpha_value


def select_action_chunk_window(
    action_chunk: np.ndarray,
    horizon: int,
    *,
    action_dim: Optional[int] = None,
) -> np.ndarray:
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2:
        raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")
    expected_action_dim = (
        int(action_dim) if action_dim is not None else int(chunk.shape[1])
    )
    if expected_action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {expected_action_dim}")
    if chunk.shape[1] != expected_action_dim:
        raise ValueError(
            f"Unexpected action chunk shape: {chunk.shape}, expected second dim {expected_action_dim}"
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if chunk.shape[0] == 0:
        raise ValueError("OpenPI returned empty action chunk")
    # Base-policy chunk length is part of the runtime contract. A short chunk is a
    # backend/data bug and should fail loudly instead of being silently padded.
    if chunk.shape[0] < horizon:
        raise ValueError(
            "Action chunk shorter than required horizon: "
            f"got {int(chunk.shape[0])}, expected at least {int(horizon)}"
        )
    return chunk[:horizon]


def as_numpy_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.shape[0] != int(action_dim):
        raise ValueError(
            f"Residual action dim mismatch: got {action_arr.shape[0]} expected {int(action_dim)}"
        )
    return action_arr


def reshape_flat_action_to_chunk(
    action: np.ndarray, *, action_dim: int, chunk_horizon: int
) -> np.ndarray:
    """Validate a flat policy output and reshape it into a 2-D action chunk."""
    flat = as_numpy_action(action, int(action_dim) * int(chunk_horizon))
    return flat.reshape(int(chunk_horizon), int(action_dim))


def _clip_gripper_last_dim_torch(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-1] <= 0:
        return actions
    clipped_last = torch.clamp(actions[..., -1:], -1.0, 1.0)
    if actions.shape[-1] == 1:
        return clipped_last
    return torch.cat([actions[..., :-1], clipped_last], dim=-1)


def _compose_step_numpy(
    *,
    base_action: np.ndarray,
    residual_action: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    alpha: float,
    clip_gripper: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    base_action_arr = np.asarray(base_action, dtype=np.float32)
    residual_action_arr = np.asarray(residual_action, dtype=np.float32)
    clipped = np.clip(residual_action_arr, -1.0, 1.0)
    residual_scale = _validate_residual_alpha(alpha)
    bounded = np.clip(clipped * residual_scale, -residual_scale, residual_scale)
    applied_delta = bounded * np.asarray(limits, dtype=np.float32)

    delta_full = np.zeros_like(base_action_arr, dtype=np.float32)
    delta_full[np.asarray(indices, dtype=np.int64)] = applied_delta

    final_action = base_action_arr + delta_full
    if clip_gripper and final_action.shape[0] > 0:
        final_action[-1] = np.clip(final_action[-1], -1.0, 1.0)
    return delta_full, final_action


def _compose_chunk_numpy(
    *,
    base_chunk: np.ndarray,
    residual_chunk: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    alpha: float,
    clip_gripper: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    base_chunk_arr = np.asarray(base_chunk, dtype=np.float32)
    residual_chunk_arr = np.asarray(residual_chunk, dtype=np.float32)

    if base_chunk_arr.ndim != 2:
        raise ValueError(
            f"Unexpected base_chunk shape: {base_chunk_arr.shape}, expected 2-D chunk"
        )
    if (
        residual_chunk_arr.ndim != 2
        or residual_chunk_arr.shape[0] != base_chunk_arr.shape[0]
    ):
        raise ValueError(
            "Residual chunk must be 2D and share the same horizon as base_chunk: "
            f"{residual_chunk_arr.shape} vs {base_chunk_arr.shape}"
        )

    delta_chunk = np.zeros_like(base_chunk_arr, dtype=np.float32)
    final_chunk = np.zeros_like(base_chunk_arr, dtype=np.float32)
    for step_idx in range(base_chunk_arr.shape[0]):
        delta_step, final_step = _compose_step_numpy(
            base_action=base_chunk_arr[step_idx],
            residual_action=residual_chunk_arr[step_idx],
            indices=indices,
            limits=limits,
            alpha=alpha,
            clip_gripper=clip_gripper,
        )
        delta_chunk[step_idx] = delta_step
        final_chunk[step_idx] = final_step
    return delta_chunk, final_chunk


@dataclass(frozen=True)
class ResidualActionTransform:
    control_indices: np.ndarray
    limits: np.ndarray
    full_action_dim: int
    chunk_horizon: int = 1
    chunk_step_enabled: bool = False
    clip_gripper: bool = True
    base_action_key: str = "base_action"
    base_action_chunk_key: str = "base_action_chunk"
    scale_key: str = "alpha"

    def __post_init__(self) -> None:
        control_indices = np.asarray(self.control_indices, dtype=np.int64).reshape(-1)
        limits = np.asarray(self.limits, dtype=np.float32).reshape(-1)
        if int(self.full_action_dim) <= 0:
            raise ValueError(
                f"full_action_dim must be positive, got {self.full_action_dim}"
            )
        if int(self.chunk_horizon) <= 0:
            raise ValueError(
                f"chunk_horizon must be positive, got {self.chunk_horizon}"
            )
        if control_indices.size == 0:
            raise ValueError("control_indices must not be empty")
        if limits.shape[0] != control_indices.shape[0]:
            raise ValueError(
                "limits must match control_indices length: "
                f"{limits.shape[0]} != {control_indices.shape[0]}"
            )
        if np.any(control_indices < 0) or np.any(
            control_indices >= int(self.full_action_dim)
        ):
            raise ValueError(
                "control_indices out of range for full_action_dim="
                f"{int(self.full_action_dim)}: {control_indices.tolist()}"
            )
        object.__setattr__(self, "control_indices", control_indices)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "full_action_dim", int(self.full_action_dim))
        object.__setattr__(self, "chunk_horizon", int(self.chunk_horizon))
        object.__setattr__(self, "chunk_step_enabled", bool(self.chunk_step_enabled))
        object.__setattr__(self, "clip_gripper", bool(self.clip_gripper))
        object.__setattr__(self, "base_action_key", str(self.base_action_key))
        object.__setattr__(
            self, "base_action_chunk_key", str(self.base_action_chunk_key)
        )
        object.__setattr__(self, "scale_key", str(self.scale_key))

    @property
    def residual_action_dim(self) -> int:
        return int(self.control_indices.shape[0])

    @property
    def policy_action_dim(self) -> int:
        if self.chunk_step_enabled:
            return int(self.chunk_horizon * self.residual_action_dim)
        return int(self.residual_action_dim)

    @property
    def critic_action_dim(self) -> int:
        if self.chunk_step_enabled:
            return int(self.chunk_horizon * self.full_action_dim)
        return int(self.full_action_dim)

    def compose(
        self,
        *,
        base_action: np.ndarray,
        residual_action: np.ndarray,
        alpha: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.chunk_step_enabled:
            raise ValueError("compose() is only valid for step residual transforms")
        return _compose_step_numpy(
            base_action=base_action,
            residual_action=residual_action,
            indices=self.control_indices,
            limits=self.limits,
            alpha=alpha,
            clip_gripper=self.clip_gripper,
        )

    def compose_chunk(
        self,
        *,
        base_chunk: np.ndarray,
        residual_chunk: np.ndarray,
        alpha: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _compose_chunk_numpy(
            base_chunk=base_chunk,
            residual_chunk=residual_chunk,
            indices=self.control_indices,
            limits=self.limits,
            alpha=alpha,
            clip_gripper=self.clip_gripper,
        )

    def compose_step_torch(
        self,
        *,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if self.chunk_step_enabled:
            raise ValueError(
                "compose_step_torch() is only valid for step residual transforms"
            )
        if base_action.ndim != 2:
            raise ValueError(
                f"Unexpected base action rank for step transform: {base_action.shape}"
            )
        control_indices = torch.as_tensor(
            self.control_indices,
            device=residual_action.device,
            dtype=torch.long,
        )
        limits = torch.as_tensor(
            self.limits,
            device=residual_action.device,
            dtype=residual_action.dtype,
        )
        scale = alpha.to(device=residual_action.device, dtype=residual_action.dtype)
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1)
        clipped = torch.clamp(residual_action, -1.0, 1.0)

        if clipped.ndim == 2:
            delta = clipped * scale * limits.view(1, -1)
            final_action = base_action.to(dtype=residual_action.dtype).clone()
            final_action[:, control_indices] = final_action[:, control_indices] + delta
        elif clipped.ndim == 3:
            delta = clipped * scale.unsqueeze(1) * limits.view(1, 1, -1)
            final_action = (
                base_action.to(dtype=residual_action.dtype)
                .unsqueeze(1)
                .expand(-1, clipped.shape[1], -1)
                .clone()
            )
            final_action[:, :, control_indices] = (
                final_action[:, :, control_indices] + delta
            )
        else:
            raise ValueError(
                f"Unsupported policy action rank for step transform: {clipped.shape}"
            )

        if self.clip_gripper and final_action.shape[-1] > 0:
            final_action = _clip_gripper_last_dim_torch(final_action)
        return final_action

    def compose_chunk_torch(
        self,
        *,
        base_chunk: torch.Tensor,
        residual_action: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if base_chunk.ndim != 3:
            raise ValueError(f"Unexpected base action chunk shape: {base_chunk.shape}")
        if (
            base_chunk.shape[1] != int(self.chunk_horizon)
            or base_chunk.shape[2] != int(self.full_action_dim)
        ):
            raise ValueError(
                "Unexpected base action chunk dims: "
                f"{tuple(base_chunk.shape)} vs (*, {int(self.chunk_horizon)}, {int(self.full_action_dim)})"
            )

        control_indices = torch.as_tensor(
            self.control_indices,
            device=residual_action.device,
            dtype=torch.long,
        )
        limits = torch.as_tensor(
            self.limits,
            device=residual_action.device,
            dtype=residual_action.dtype,
        )
        scale = alpha.to(device=residual_action.device, dtype=residual_action.dtype)
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1)
        clipped = torch.clamp(residual_action, -1.0, 1.0)

        if clipped.ndim == 2:
            residual_chunk = clipped.reshape(
                -1, int(self.chunk_horizon), int(self.residual_action_dim)
            )
            delta = residual_chunk * scale.unsqueeze(1) * limits.view(1, 1, -1)
            final_chunk = base_chunk.to(dtype=residual_action.dtype).clone()
            final_chunk[:, :, control_indices] = (
                final_chunk[:, :, control_indices] + delta
            )
            if self.clip_gripper and final_chunk.shape[-1] > 0:
                final_chunk = _clip_gripper_last_dim_torch(final_chunk)
            return final_chunk.reshape(-1, int(self.critic_action_dim))

        if clipped.ndim == 3:
            residual_chunk = clipped.reshape(
                -1,
                clipped.shape[1],
                int(self.chunk_horizon),
                int(self.residual_action_dim),
            )
            delta = (
                residual_chunk
                * scale.unsqueeze(1).unsqueeze(1)
                * limits.view(1, 1, 1, -1)
            )
            final_chunk = (
                base_chunk.to(dtype=residual_action.dtype)
                .unsqueeze(1)
                .expand(-1, clipped.shape[1], -1, -1)
                .clone()
            )
            final_chunk[:, :, :, control_indices] = (
                final_chunk[:, :, :, control_indices] + delta
            )
            if self.clip_gripper and final_chunk.shape[-1] > 0:
                final_chunk = _clip_gripper_last_dim_torch(final_chunk)
            return final_chunk.reshape(-1, clipped.shape[1], int(self.critic_action_dim))

        raise ValueError(
            f"Unsupported policy action rank for chunk transform: {clipped.shape}"
        )

    def project_critic_action_mask_to_policy_space(
        self,
        action_mask: torch.Tensor,
        *,
        policy_action_dim: int,
    ) -> torch.Tensor:
        control_indices = torch.as_tensor(
            self.control_indices,
            device=action_mask.device,
            dtype=torch.long,
        )
        if self.chunk_step_enabled:
            expected_critic_dim = int(self.critic_action_dim)
            if int(action_mask.shape[-1]) != expected_critic_dim:
                raise ValueError(
                    "Unexpected chunk critic action_mask dim: "
                    f"{action_mask.shape[-1]} != {expected_critic_dim}"
                )
            projected = action_mask.reshape(
                *action_mask.shape[:-1], int(self.chunk_horizon), int(self.full_action_dim)
            )
            projected = projected.index_select(dim=-1, index=control_indices)
            projected = projected.reshape(*projected.shape[:-2], -1)
        else:
            expected_critic_dim = int(self.full_action_dim)
            if int(action_mask.shape[-1]) != expected_critic_dim:
                raise ValueError(
                    "Unexpected step critic action_mask dim: "
                    f"{action_mask.shape[-1]} != {expected_critic_dim}"
                )
            projected = action_mask.index_select(dim=-1, index=control_indices)

        if int(projected.shape[-1]) != int(policy_action_dim):
            raise ValueError(
                "Projected policy action_mask dim mismatch: "
                f"{projected.shape[-1]} != {policy_action_dim}"
            )
        return projected

    @classmethod
    def from_legacy_config(cls, config: Mapping[str, Any]) -> "ResidualActionTransform":
        transform_type = str(config.get("type", "residual_combined")).strip().lower()
        if transform_type != "residual_combined":
            raise ValueError(f"Unsupported action_transform.type: {transform_type!r}")
        return cls(
            control_indices=np.asarray(config["control_indices"], dtype=np.int64),
            limits=np.asarray(config["limits"], dtype=np.float32),
            full_action_dim=int(config["full_action_dim"]),
            chunk_horizon=int(config.get("chunk_horizon", 1)),
            chunk_step_enabled=bool(config.get("chunk_step_enabled", False)),
            clip_gripper=bool(config.get("clip_gripper", True)),
            base_action_key=str(config.get("base_action_key", "base_action")),
            base_action_chunk_key=str(
                config.get("base_action_chunk_key", "base_action_chunk")
            ),
            scale_key=str(config.get("scale_key", "alpha")),
        )


def coerce_residual_action_transform(
    value: Any,
) -> Optional[ResidualActionTransform]:
    if value is None:
        return None
    if isinstance(value, ResidualActionTransform):
        return value
    if isinstance(value, Mapping):
        return ResidualActionTransform.from_legacy_config(value)
    raise TypeError(
        "action_transform must be None, ResidualActionTransform, or a compatible "
        f"mapping; got {type(value)}"
    )


def compose_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    alpha: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return _compose_step_numpy(
        base_action=base_action,
        residual_action=residual_action,
        indices=np.asarray(indices, dtype=np.int64),
        limits=np.asarray(limits, dtype=np.float32),
        alpha=alpha,
        clip_gripper=clip_gripper,
    )


def compose_residual_action_chunk(
    *,
    base_chunk: np.ndarray,
    residual_chunk: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    alpha: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return _compose_chunk_numpy(
        base_chunk=base_chunk,
        residual_chunk=residual_chunk,
        indices=np.asarray(indices, dtype=np.int64),
        limits=np.asarray(limits, dtype=np.float32),
        alpha=alpha,
        clip_gripper=clip_gripper,
    )
