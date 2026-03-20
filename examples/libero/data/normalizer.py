"""Generic state/action normalization utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np


class StateActionNormalizer:
    _EPS = 1e-6

    def __init__(self, stats_path: str | Path):
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.state_mean = np.asarray(raw["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(raw["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(raw["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(raw["action_std"], dtype=np.float32)

        if self.state_mean.shape != self.state_std.shape:
            raise ValueError("state_mean/state_std shape mismatch")
        if self.action_mean.shape != self.action_std.shape:
            raise ValueError("action_mean/action_std shape mismatch")

        self.fused_mean = np.concatenate([self.state_mean, self.action_mean], axis=-1)
        self.fused_std = np.concatenate([self.state_std, self.action_std], axis=-1)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (np.asarray(state, dtype=np.float32) - self.state_mean) / (self.state_std + self._EPS)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (np.asarray(action, dtype=np.float32) - self.action_mean) / (self.action_std + self._EPS)

    def normalize_fused(self, fused_state: np.ndarray) -> np.ndarray:
        return (np.asarray(fused_state, dtype=np.float32) - self.fused_mean) / (self.fused_std + self._EPS)


def load_normalizer(task_key: str, stats_dir: str | Path | None = None) -> Optional[StateActionNormalizer]:
    if stats_dir is None:
        stats_dir = Path(__file__).resolve().parent / "stats"
    stats_dir = Path(stats_dir)
    stats_path = stats_dir / f"{task_key}.json"
    if not stats_path.exists():
        logging.getLogger(__name__).warning(
            "Normalization stats file not found: %s — running without normalization",
            stats_path,
        )
        return None
    return StateActionNormalizer(stats_path)

