"""Generic residual observation assembly helpers."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _resolve_residual_scale(*, alpha: Optional[float], default: float = 1.0) -> float:
    scale = default if alpha is None else alpha
    scale_value = float(scale)
    if not np.isfinite(scale_value):
        raise ValueError(f"Residual alpha must be finite, got {scale!r}")
    if scale_value < 0.0:
        raise ValueError(f"Residual alpha must be >= 0, got {scale_value}")
    return scale_value


def _build_fused_residual_state(
    *,
    state_core: np.ndarray,
    base_action: np.ndarray,
    base_action_chunk: Optional[np.ndarray],
    alpha: Optional[float] = None,
) -> np.ndarray:
    residual_scale = _resolve_residual_scale(alpha=alpha, default=1.0)
    base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)

    fused_parts = [
        np.asarray(state_core, dtype=np.float32).reshape(-1),
        base_action_arr,
    ]

    if base_action_chunk is not None:
        fused_parts.append(np.asarray(base_action_chunk, dtype=np.float32).reshape(-1))

    fused_parts.append(np.asarray([float(residual_scale)], dtype=np.float32))
    return np.concatenate(fused_parts, axis=-1).astype(np.float32)


def build_residual_step_obs_from_core(
    core: Dict[str, np.ndarray],
    *,
    base_action: np.ndarray,
    base_action_chunk: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    stack_horizon: int = 1,
) -> Dict[str, np.ndarray]:
    if int(stack_horizon) != 1:
        raise ValueError(
            f"Only stack_horizon=1 is currently supported, got {int(stack_horizon)}"
        )
    if "state_core" not in core:
        raise KeyError("core must include 'state_core'")

    residual_scale = _resolve_residual_scale(alpha=alpha, default=1.0)
    state_core_arr = np.asarray(core["state_core"], dtype=np.float32).reshape(-1)
    base_action_arr = np.asarray(base_action, dtype=np.float32).reshape(-1)
    base_action_chunk_arr = (
        np.asarray(base_action_chunk, dtype=np.float32)
        if base_action_chunk is not None
        else None
    )

    policy_state = _build_fused_residual_state(
        state_core=state_core_arr,
        base_action=base_action_arr,
        base_action_chunk=base_action_chunk_arr,
        alpha=float(residual_scale),
    )

    obs_out: Dict[str, np.ndarray] = {
        "state": np.expand_dims(policy_state, axis=0).astype(np.float32),
        "base_action": np.expand_dims(base_action_arr, axis=0).astype(np.float32),
        "alpha": np.asarray([[float(residual_scale)]], dtype=np.float32),
    }
    for key, value in core.items():
        if key == "state_core":
            continue
        obs_out[key] = np.expand_dims(np.asarray(value).copy(), axis=0)
    if base_action_chunk_arr is not None:
        obs_out["base_action_chunk"] = np.expand_dims(
            base_action_chunk_arr.astype(np.float32), axis=0
        )
    return obs_out
