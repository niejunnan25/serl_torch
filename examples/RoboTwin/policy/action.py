"""动作相关工具：control indices 解析、residual limits、动作组合、chunk 窗口。"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from utils.constants import ALOHA_ACTION_DIM


# ---------------------------------------------------------------------------
# Action chunk 窗口
# ---------------------------------------------------------------------------


def select_action_chunk_window(action_chunk: np.ndarray, horizon: int) -> np.ndarray:
    """
    从 OpenPI 返回的动作块中截取或填充为指定步数 horizon 的窗口。

    - 若动作块步数 >= horizon：取前 horizon 步返回。
    - 若动作块步数 < horizon：用最后一步重复填充到 horizon 步。
    """
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ALOHA_ACTION_DIM:
        raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if chunk.shape[0] == 0:
        raise ValueError("OpenPI returned empty action chunk")

    if chunk.shape[0] >= horizon:
        return chunk[:horizon]

    pad_count = horizon - chunk.shape[0]
    tail = np.repeat(chunk[-1:, :], pad_count, axis=0)
    return np.concatenate([chunk, tail], axis=0)


# ---------------------------------------------------------------------------
# Control indices 解析
# ---------------------------------------------------------------------------


def controlled_action_indices(control_gripper: bool) -> np.ndarray:
    """
    返回受控动作在 14 维 ALOHA 动作中的下标。
    control_gripper=True 时返回 0..13（双臂+双夹爪）；False 时排除夹爪维度 6 和 13。
    """
    if control_gripper:
        return np.arange(ALOHA_ACTION_DIM, dtype=np.int64)
    return np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)


def default_control_indices_for_dim(action_dim: int) -> np.ndarray:
    """
    根据 residual 输出维度给出默认受控下标：

    - 14: 双臂+双夹爪
    - 12: 双臂（不含夹爪）
    - 7 : 左臂+左夹爪
    - 6 : 左臂（不含夹爪）
    """
    dim = int(action_dim)
    if dim == 14:
        return np.arange(ALOHA_ACTION_DIM, dtype=np.int64)
    if dim == 12:
        return np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    if dim == 7:
        return np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    if dim == 6:
        return np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    raise ValueError(
        "Unsupported residual.action_dim. "
        "Please set residual.action_indices explicitly for custom layouts."
    )


def _normalize_control_indices(indices: List[int]) -> np.ndarray:
    arr = np.asarray([int(v) for v in indices], dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("residual.action_indices cannot be empty")
    if np.any(arr < 0) or np.any(arr >= ALOHA_ACTION_DIM):
        raise ValueError(
            f"residual.action_indices must be in [0, {ALOHA_ACTION_DIM - 1}], got {arr.tolist()}"
        )
    if np.unique(arr).size != arr.size:
        raise ValueError(f"residual.action_indices has duplicates: {arr.tolist()}")
    return arr


def resolve_control_indices(
    *,
    action_dim: Optional[int] = None,
    action_indices: Optional[List[int]] = None,
    control_gripper: Optional[bool] = None,
) -> np.ndarray:
    """
    解析受控动作下标（优先级从高到低）：

    1. ``residual.action_indices``（最明确）
    2. ``residual.action_dim``（按默认映射）
    3. ``residual.control_gripper``（兼容旧配置）
    """
    if action_indices is not None:
        resolved = _normalize_control_indices(action_indices)
        if action_dim is not None and resolved.size != int(action_dim):
            raise ValueError(
                "residual.action_dim does not match residual.action_indices length: "
                f"{int(action_dim)} vs {int(resolved.size)}"
            )
        return resolved

    if action_dim is not None:
        return default_control_indices_for_dim(int(action_dim))

    # 兼容旧配置
    if control_gripper is None:
        control_gripper = True
    return controlled_action_indices(bool(control_gripper))


# ---------------------------------------------------------------------------
# Residual limits
# ---------------------------------------------------------------------------


def build_residual_limits(indices: np.ndarray, arm_limit: float, gripper_limit: float) -> np.ndarray:
    """
    按 ALOHA 动作顺序（左臂 6 维 + 左夹爪 1 维 + 右臂 6 维 + 右夹爪 1 维）构建每维残差上限，
    再按 indices 选取。
    """
    full_limits = np.asarray(
        [arm_limit] * 6 + [gripper_limit] + [arm_limit] * 6 + [gripper_limit],
        dtype=np.float32,
    )
    return full_limits[indices]


# ---------------------------------------------------------------------------
# 单步 / Chunk 动作组合
# ---------------------------------------------------------------------------


def as_numpy_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    """将策略输出动作转成一维 numpy，并校验维度。"""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != action_dim:
        raise ValueError(f"Residual action dim mismatch: got {action.shape[0]} expected {action_dim}")
    return action


def compose_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    indices: np.ndarray,
    limits: np.ndarray,
    residual_scale: float,
    xi: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    单步动作组合：将基动作与残差动作合成最终动作。

    残差先 clip 到 [-1,1]，再按 xi 映射到 [-xi,xi]，
    最后乘以 limits 和 residual_scale 得到 delta。返回 ``(delta_full, final_action)``。
    """
    base_action = np.asarray(base_action, dtype=np.float32)
    residual_action = np.asarray(residual_action, dtype=np.float32)

    clipped = np.clip(residual_action, -1.0, 1.0)
    xi = float(max(0.0, xi))
    bounded = np.clip(clipped * xi, -xi, xi)
    applied_delta = bounded * limits * float(residual_scale)

    delta_full = np.zeros_like(base_action, dtype=np.float32)
    delta_full[indices] = applied_delta

    final_action = base_action + delta_full
    if clip_gripper:
        final_action[6] = np.clip(final_action[6], 0.0, 1.0)
        final_action[13] = np.clip(final_action[13], 0.0, 1.0)

    return delta_full, final_action


def split_residual_chunk_action(
    residual_chunk_action: np.ndarray,
    horizon: int,
    per_step_action_dim: int,
) -> np.ndarray:
    """将展平的残差 chunk 动作拆成 ``(horizon, per_step_action_dim)``。"""
    residual = np.asarray(residual_chunk_action, dtype=np.float32).reshape(-1)
    expected_dim = int(horizon) * int(per_step_action_dim)
    if residual.shape[0] != expected_dim:
        raise ValueError(
            f"Residual chunk dim mismatch: got {residual.shape[0]} expected {expected_dim}"
        )
    return residual.reshape(int(horizon), int(per_step_action_dim))


def compose_residual_chunk_action(
    base_chunk: np.ndarray,
    residual_chunk_action: np.ndarray,
    horizon: int,
    indices: np.ndarray,
    limits: np.ndarray,
    residual_scale: float,
    xi: float = 1.0,
    clip_gripper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """整段 chunk 的动作组合：base_chunk 与残差 chunk 逐步合成。"""
    base_chunk = np.asarray(base_chunk, dtype=np.float32)
    if base_chunk.shape != (int(horizon), ALOHA_ACTION_DIM):
        raise ValueError(
            f"Base chunk shape mismatch: got {base_chunk.shape} expected {(int(horizon), ALOHA_ACTION_DIM)}"
        )

    residual_per_step = split_residual_chunk_action(
        residual_chunk_action=residual_chunk_action,
        horizon=int(horizon),
        per_step_action_dim=int(len(indices)),
    )

    clipped = np.clip(residual_per_step, -1.0, 1.0)
    xi = float(max(0.0, xi))
    bounded = np.clip(clipped * xi, -xi, xi)
    applied_delta = bounded * limits.reshape(1, -1) * float(residual_scale)

    delta_full = np.zeros_like(base_chunk, dtype=np.float32)
    delta_full[:, indices] = applied_delta

    final_chunk = base_chunk + delta_full
    if clip_gripper:
        final_chunk[:, 6] = np.clip(final_chunk[:, 6], 0.0, 1.0)
        final_chunk[:, 13] = np.clip(final_chunk[:, 13], 0.0, 1.0)

    return delta_full, final_chunk
