from __future__ import annotations

"""Runtime statistics for residual actions."""

from typing import Any

import numpy as np


class ResidualActionStatsAccumulator:
    """Accumulate per-episode residual-action statistics."""

    def __init__(self, *, saturation_threshold: float = 0.95) -> None:
        self._saturation_threshold = float(saturation_threshold)
        self._residual_sum = 0.0
        self._residual_abs_sum = 0.0
        self._residual_square_sum = 0.0
        self._residual_max_abs = 0.0
        self._residual_values = 0
        self._residual_saturated_values = 0
        self._delta_abs_sum = 0.0
        self._delta_max_abs = 0.0
        self._delta_values = 0

    def add(
        self,
        *,
        residual_action: Any,
        base_action: Any | None = None,
        final_action: Any | None = None,
    ) -> None:
        residual = np.asarray(residual_action, dtype=np.float32).reshape(-1)
        if residual.size > 0:
            residual_abs = np.abs(residual)
            self._residual_sum += float(np.sum(residual))
            self._residual_abs_sum += float(np.sum(residual_abs))
            self._residual_square_sum += float(np.sum(np.square(residual)))
            self._residual_max_abs = max(
                self._residual_max_abs,
                float(np.max(residual_abs)),
            )
            self._residual_values += int(residual.size)
            self._residual_saturated_values += int(
                np.count_nonzero(residual_abs >= self._saturation_threshold)
            )

        if base_action is None or final_action is None:
            return
        base = np.asarray(base_action, dtype=np.float32)
        final = np.asarray(final_action, dtype=np.float32)
        if base.shape != final.shape:
            return
        delta_abs = np.abs(final - base)
        if delta_abs.size <= 0:
            return
        self._delta_abs_sum += float(np.sum(delta_abs))
        self._delta_max_abs = max(self._delta_max_abs, float(np.max(delta_abs)))
        self._delta_values += int(delta_abs.size)

    def summary(self) -> dict[str, float | int]:
        if self._residual_values <= 0:
            return {}
        residual_values = max(int(self._residual_values), 1)
        residual_mean = self._residual_sum / float(residual_values)
        residual_mean_abs = self._residual_abs_sum / float(residual_values)
        residual_mean_square = self._residual_square_sum / float(residual_values)
        residual_variance = max(
            0.0,
            residual_mean_square - float(residual_mean) ** 2,
        )
        payload: dict[str, float | int] = {
            "mean_abs": float(residual_mean_abs),
            "max_abs": float(self._residual_max_abs),
            "std": float(residual_variance**0.5),
            "saturation_rate": float(self._residual_saturated_values)
            / float(residual_values),
            "value_count": int(self._residual_values),
        }
        if self._delta_values > 0:
            delta_values = max(int(self._delta_values), 1)
            payload["action_delta_mean_abs"] = float(
                self._delta_abs_sum / float(delta_values)
            )
            payload["action_delta_max_abs"] = float(self._delta_max_abs)
        return payload
