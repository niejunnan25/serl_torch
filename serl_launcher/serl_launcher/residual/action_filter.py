from __future__ import annotations

"""Residual action filtering helpers shared across rollout environments."""

from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass(slots=True)
class ResidualDeltaActionFilter:
    enabled: bool
    alpha: float
    max_delta: float | None = None
    warmup_steps: int = 0
    reset_each_episode: bool = True
    _previous_delta: np.ndarray | None = field(init=False, default=None, repr=False)
    _total_steps: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.alpha = float(self.alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        self.max_delta = None if self.max_delta is None else float(self.max_delta)
        if self.max_delta is not None and self.max_delta <= 0.0:
            raise ValueError(f"max_delta must be positive, got {self.max_delta}")
        self.warmup_steps = int(self.warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be nonnegative, got {self.warmup_steps}"
            )
        self.reset_each_episode = bool(self.reset_each_episode)
        self._previous_delta: np.ndarray | None = None
        self._total_steps = 0

    @property
    def total_steps(self) -> int:
        return int(self._total_steps)

    def reset_episode(self) -> None:
        if self.reset_each_episode:
            self._previous_delta = None

    def filter_action_chunk(
        self,
        *,
        base_action_chunk: np.ndarray,
        final_action_chunk: np.ndarray,
    ) -> np.ndarray:
        final_actions = np.asarray(final_action_chunk, dtype=np.float32)
        if not self.enabled:
            return final_actions

        base_actions = np.asarray(base_action_chunk, dtype=np.float32)
        if base_actions.shape != final_actions.shape:
            if int(base_actions.size) != int(final_actions.size):
                raise ValueError(
                    "base and final action chunk shape mismatch: "
                    f"base={base_actions.shape} final={final_actions.shape}"
                )
            base_actions = base_actions.reshape(final_actions.shape)

        residual_delta_chunk = final_actions - base_actions
        filtered_delta_chunk = self.filter_residual_delta_chunk(residual_delta_chunk)
        return np.asarray(base_actions + filtered_delta_chunk, dtype=np.float32)

    def filter_residual_delta_chunk(
        self,
        residual_delta_chunk: np.ndarray,
    ) -> np.ndarray:
        if not self.enabled:
            return np.asarray(residual_delta_chunk, dtype=np.float32)

        filtered = np.array(residual_delta_chunk, dtype=np.float32, copy=True)
        if filtered.ndim != 2:
            return filtered

        for index in range(filtered.shape[0]):
            current_delta = filtered[index]

            if self._previous_delta is None:
                self._previous_delta = np.array(
                    current_delta,
                    dtype=np.float32,
                    copy=True,
                )
            elif self._total_steps >= self.warmup_steps:
                smoothed = (
                    self.alpha * current_delta
                    + (1.0 - self.alpha) * self._previous_delta
                )
                if self.max_delta is not None:
                    delta = smoothed - self._previous_delta
                    smoothed = self._previous_delta + np.clip(
                        delta,
                        -self.max_delta,
                        self.max_delta,
                    )
                self._previous_delta = np.asarray(smoothed, dtype=np.float32)
            else:
                self._previous_delta = np.array(
                    current_delta,
                    dtype=np.float32,
                    copy=True,
                )

            filtered[index] = self._previous_delta
            self._total_steps += 1

        return filtered
