"""State / Action 归一化：从 JSON 统计文件加载 mean/std，提供 normalize / denormalize。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from utils.constants import ALOHA_ACTION_DIM


class StateActionNormalizer:
    """
    从 JSON 统计文件中加载 state/action 的 mean/std，
    提供 normalize / denormalize 功能，用于残差策略的输入归一化。

    JSON 文件格式::

        {
            "state_mean": [14 floats],
            "state_std":  [14 floats],
            "action_mean": [14 floats],
            "action_std":  [14 floats]
        }
    """

    _EPS = 1e-6  # 防止除零

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

        if self.state_mean.shape != (ALOHA_ACTION_DIM,):
            raise ValueError(
                f"state_mean shape mismatch: {self.state_mean.shape}, expected ({ALOHA_ACTION_DIM},)"
            )
        if self.action_mean.shape != (ALOHA_ACTION_DIM,):
            raise ValueError(
                f"action_mean shape mismatch: {self.action_mean.shape}, expected ({ALOHA_ACTION_DIM},)"
            )

        # 预先拼接好 fused_state 的 mean/std (28 维 = 14 state + 14 action)
        self.fused_mean = np.concatenate([self.state_mean, self.action_mean], axis=-1)
        self.fused_std = np.concatenate([self.state_std, self.action_std], axis=-1)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """归一化 state (14 维)。"""
        return (np.asarray(state, dtype=np.float32) - self.state_mean) / (self.state_std + self._EPS)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """归一化 action (14 维)。"""
        return (np.asarray(action, dtype=np.float32) - self.action_mean) / (self.action_std + self._EPS)

    def normalize_fused(self, fused_state: np.ndarray) -> np.ndarray:
        """归一化已拼接的 fused_state (28 维 = state || action)。"""
        return (np.asarray(fused_state, dtype=np.float32) - self.fused_mean) / (self.fused_std + self._EPS)

    def denormalize_state(self, normalized: np.ndarray) -> np.ndarray:
        return np.asarray(normalized, dtype=np.float32) * (self.state_std + self._EPS) + self.state_mean

    def denormalize_action(self, normalized: np.ndarray) -> np.ndarray:
        return np.asarray(normalized, dtype=np.float32) * (self.action_std + self._EPS) + self.action_mean


def load_normalizer(
    task_name: str,
    stats_dir: str | Path | None = None,
) -> Optional[StateActionNormalizer]:
    """
    根据任务名加载归一化器。若统计文件不存在则返回 None 并打印警告。

    stats_dir 默认为 ``data/stats/``，统计文件名为 ``{task_name}.json``。
    """
    if stats_dir is None:
        stats_dir = Path(__file__).resolve().parent / "stats"
    stats_dir = Path(stats_dir)
    stats_path = stats_dir / f"{task_name}.json"
    if not stats_path.exists():
        logging.getLogger(__name__).warning(
            "Normalization stats file not found: %s — running without normalization", stats_path
        )
        return None
    return StateActionNormalizer(stats_path)
