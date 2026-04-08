"""Helpers for validating residual alpha values."""
from __future__ import annotations

import math
from typing import Any


def validate_alpha(
    value: Any,
    *,
    name: str = "alpha",
    allow_zero: bool = True,
) -> float:
    if value is None:
        raise ValueError(f"{name} must be set")
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(alpha):
        raise ValueError(f"{name} must be finite, got {alpha!r}")
    if allow_zero:
        if alpha < 0.0:
            raise ValueError(f"{name} must be >= 0.0, got {alpha}")
    elif alpha <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {alpha}")
    return alpha


def require_residual_alpha(residual_cfg: Any, *, path: str = "residual.alpha") -> float:
    if residual_cfg is None:
        raise ValueError(f"{path} must be explicitly set")
    try:
        alpha_raw = residual_cfg.get("alpha", None)
    except AttributeError as exc:
        raise TypeError(
            f"{path} container must support .get(...), got {type(residual_cfg)}"
        ) from exc
    if alpha_raw is None:
        raise ValueError(f"{path} must be explicitly set")
    return validate_alpha(alpha_raw, name=path, allow_zero=True)
