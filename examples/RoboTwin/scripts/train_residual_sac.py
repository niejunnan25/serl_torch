from __future__ import annotations

"""
RoboTwin 残差策略训练脚本（单任务，步级残差，DrQ-SAC）。

核心训练流程：
1. OpenPI 先给出 base action chunk（默认长度 10）；
2. 残差策略每执行一步都推理一次（每次输出 action_dim 维残差，可配置）；
3. 在同一个 base chunk 内逐步执行：`a_final_t = a_base_t + a_res_t`；
4. 以“单步 residual 决策”为一个 transition 写入 replay；
5. 使用 DrQ-SAC（SAC + 图像随机裁剪增强）进行更新。
6. 与评估对齐：若某 seed 连专家预检都失败，则跳过该 seed，不计入训练 episode。

在线数据流（与 env 交互）：
obs_raw -> OpenPI infer_chunk -> base_chunk(H,14)
      -> build_residual_step_obs(含图像 + state(14 + 14))
      -> residual policy.sample_actions -> residual_step_action(action_dim,)
      -> compose_residual_action(base_t + delta_t) -> final_action(14,)
      -> env.step(final_action) -> next_obs_raw,reward,done
      -> 组装单步 transition 写入 online replay
      -> 达到训练条件后，从 online/offline 混采 batch 更新 DrQ-SAC

离线数据流（可选）：
offline payload -> transition 解析
  A. 已是 residual 格式：仅做 shape/dtype/key 规范化后入 offline replay
  B. 专家动作格式：用同一 OpenPI 接口回推 base_chunk，
     用 (a_expert - a_base)/(limits*scale) 反解 residual，再入 offline replay
"""

import json
import logging
import os
import pickle
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

    # Keep legacy `gym.*` imports working when only Gymnasium is installed.
    sys.modules["gym"] = gym
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
del _PROJECT_ROOT

from utils import JsonlLogger, ensure_serl_launcher_importable
from utils.config_utils import (
    set_global_seeds,
    resolve_image_keys,
    resolve_control_indices_from_cfg,
    build_drq_agent,
    sample_probing_steps,
)
from data import StateActionNormalizer, load_normalizer
from env_wrappers import (
    RemoteRoboTwinTaskEnv,
    RoboTwinTaskEnv,
    load_task_args,
    resolve_robo_root,
    setup_robotwin_pythonpath,
)
from policy import (
    OpenPIChunkClient,
    as_numpy_action,
    build_residual_limits,
    build_residual_step_obs,
    compose_residual_action,
    select_action_chunk_window,
)

ensure_serl_launcher_importable()

from torch.utils.tensorboard import SummaryWriter

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.utils.checkpoint_utils import save_agent_checkpoint
from serl_launcher.utils.train_utils import concat_batches


def _obs_space_from_sample(sample_obs: Dict[str, np.ndarray]) -> gym.spaces.Dict:
    """由样本观测字典推断 gym Dict 空间，用于初始化 ReplayBuffer。"""
    spaces: Dict[str, gym.spaces.Space] = {}
    for key, value in sample_obs.items():
        arr = np.asarray(value)
        if key == "state":
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=np.float32,
            )
        elif np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            spaces[key] = gym.spaces.Box(
                low=info.min,
                high=info.max,
                shape=arr.shape,
                dtype=arr.dtype,
            )
        else:
            spaces[key] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=arr.shape,
                dtype=np.float32,
            )
    return gym.spaces.Dict(spaces)


def _clone_obs_dict(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """深拷贝观测字典（以 numpy copy 为准），避免后续原地修改污染 replay 数据。"""
    return {key: np.asarray(value).copy() for key, value in obs_dict.items()}


def _zero_obs_like(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """构造与观测结构一致的全零观测（用于 done 时 next_observation 占位）。"""
    return {key: np.zeros_like(value) for key, value in obs_dict.items()}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """将标量/数组安全转成 float。"""
    if value is None:
        return float(default)
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])
    except Exception:  # noqa: BLE001
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    """将标量/数组安全转成 bool。"""
    if value is None:
        return bool(default)
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return bool(default)
        return bool(arr.reshape(-1)[0])
    except Exception:  # noqa: BLE001
        return bool(default)


def _reduce_reward(value: Any, default: float = 0.0) -> float:
    """将 reward 规约为标量。若是向量/数组，使用求和而不是取第一个元素。"""
    if value is None:
        return float(default)
    try:
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return float(default)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.sum())
    except Exception:  # noqa: BLE001
        return float(default)


def _resolve_offline_paths(dataset_paths: Any, base_dir: Path) -> List[Path]:
    """解析配置中的离线路径（支持单文件、目录自动发现 *.pkl、glob 通配符）。"""
    import glob as _glob

    if dataset_paths is None:
        return []

    raw_items: List[str]
    if isinstance(dataset_paths, (str, Path)):
        raw_items = [str(dataset_paths)]
    elif isinstance(dataset_paths, Iterable):
        raw_items = [str(item) for item in dataset_paths]
    else:
        raw_items = [str(dataset_paths)]

    paths: List[Path] = []
    for item in raw_items:
        if not item or item.lower() == "null":
            continue
        p = Path(item).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        if p.is_dir():
            discovered = sorted(p.glob("*.pkl"))
            paths.extend(discovered)
        elif any(c in str(p) for c in ("*", "?", "[")):
            expanded = sorted(_glob.glob(str(p)))
            paths.extend(Path(ep) for ep in expanded)
        else:
            paths.append(p)
    return paths


def _to_action_sequence(action_like: Any, full_action_dim: int) -> np.ndarray:
    """将输入动作统一转换为 (T, full_action_dim) 的序列（不做 horizon 填充）。"""
    arr = np.asarray(action_like, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] == full_action_dim:
            arr = arr.reshape(1, full_action_dim)
        elif arr.shape[0] % full_action_dim == 0:
            arr = arr.reshape(-1, full_action_dim)
        else:
            raise ValueError(f"Cannot reshape action to chunk: shape={arr.shape}")
    elif arr.ndim == 2 and arr.shape[1] == full_action_dim:
        pass
    else:
        raise ValueError(f"Unsupported action chunk shape: {arr.shape}")
    return arr


def _normalize_obs_dict_for_buffer(
    obs_dict: Dict[str, Any],
    sample_obs_template: Dict[str, np.ndarray],
) -> Optional[Dict[str, np.ndarray]]:
    """将观测字典按样本模板做 key/shape/dtype 对齐，不匹配则返回 None。"""
    if not isinstance(obs_dict, dict):
        return None
    if set(obs_dict.keys()) != set(sample_obs_template.keys()):
        return None

    normalized: Dict[str, np.ndarray] = {}
    for key, template_value in sample_obs_template.items():
        template_arr = np.asarray(template_value)
        arr = np.asarray(obs_dict[key])
        if arr.shape != template_arr.shape:
            return None
        if arr.dtype != template_arr.dtype:
            arr = arr.astype(template_arr.dtype, copy=False)
        else:
            arr = arr.copy()
        normalized[key] = arr
    return normalized


def _looks_like_transition_dict(item: Dict[str, Any]) -> bool:
    """粗略判断一个 dict 是否像 transition。"""
    keys = set(item.keys())
    if {"observations", "actions", "next_observations"}.issubset(keys):
        return True
    obs_keys = {
        "obs",
        "observation",
        "obs_raw",
        "observation_raw",
        "raw_obs",
        "raw_observation",
        "observations_raw",
        "observations",
        "next_obs",
        "next_observation",
        "next_obs_raw",
        "next_observation_raw",
        "next_raw_obs",
        "next_raw_observation",
        "next_observations_raw",
        "next_observations",
    }
    action_keys = {
        "actions",
        "action",
        "action_chunk",
        "expert_action_chunk",
        "expert_chunk",
        "expert_actions",
        "residual_action",
        "residual_actions",
    }
    return bool(keys & obs_keys) and bool(keys & action_keys)


def _collect_transitions_from_payload(payload: Any) -> List[Dict[str, Any]]:
    """从 pkl 载入对象中提取 transition 列表。"""
    transitions: List[Dict[str, Any]] = []
    visited_ids: set[int] = set()

    def _visit(obj: Any) -> None:
        obj_id = id(obj)
        if obj_id in visited_ids:
            return
        visited_ids.add(obj_id)

        if isinstance(obj, dict):
            if _looks_like_transition_dict(obj):
                transitions.append(obj)
                return

            preferred_keys = ("transitions", "samples", "data", "records", "items", "episodes", "trajectories")
            expanded = False
            for key in preferred_keys:
                value = obj.get(key)
                if isinstance(value, (dict, list, tuple)):
                    _visit(value)
                    expanded = True

            if not expanded:
                for value in obj.values():
                    if isinstance(value, (dict, list, tuple)):
                        _visit(value)
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                _visit(item)

    _visit(payload)
    return transitions


def _extract_raw_obs_from_transition(transition: Dict[str, Any], *, next_obs: bool) -> Optional[Dict[str, Any]]:
    """从 transition 中提取 RoboTwin 原始观测（含 joint_action/observation）。"""
    candidate_keys = (
        (
            "next_obs",
            "next_observation",
            "next_obs_raw",
            "next_observation_raw",
            "next_raw_obs",
            "next_raw_observation",
            "next_observations_raw",
            "next_observations",
        )
        if next_obs
        else (
            "obs",
            "observation",
            "obs_raw",
            "observation_raw",
            "raw_obs",
            "raw_observation",
            "observations_raw",
            "observations",
        )
    )

    for key in candidate_keys:
        value = transition.get(key)
        if isinstance(value, dict) and "joint_action" in value and "observation" in value:
            return value
    return None


def _extract_chunk_step_index(
    transition: Dict[str, Any],
    horizon: int,
) -> int:
    """从 transition 中提取 chunk 内步号；无则默认 0。"""
    for key in ("chunk_step", "step_in_chunk", "chunk_index", "step_idx"):
        if key not in transition:
            continue
        try:
            idx = int(np.asarray(transition[key]).reshape(-1)[0])
            return max(0, min(int(horizon) - 1, idx))
        except Exception:  # noqa: BLE001
            continue
    return 0


def _has_chunk_step_key(transition: Dict[str, Any]) -> bool:
    """离线样本是否显式提供了 chunk 内步号。"""
    return any(key in transition for key in ("chunk_step", "step_in_chunk", "chunk_index", "step_idx"))


def _extract_action_sequence_by_keys(
    transition: Dict[str, Any],
    *,
    keys: Tuple[str, ...],
    full_action_dim: int,
) -> Optional[np.ndarray]:
    """按给定 key 顺序提取动作序列 (T, full_action_dim)。"""
    for key in keys:
        if key not in transition:
            continue
        try:
            return _to_action_sequence(transition[key], full_action_dim)
        except Exception:  # noqa: BLE001
            continue
    return None


def _prepare_preconverted_transition(
    transition: Dict[str, Any],
    *,
    sample_obs_template: Dict[str, np.ndarray],
    action_dim: int,
    full_action_dim: int,
    control_indices: np.ndarray,
    chunk_horizon: int,
    accept_plain_preconverted: bool,
    clip_residual_to_unit: bool,
) -> Optional[Dict[str, Any]]:
    """
    尝试将“已是 residual 格式”的 transition 规范化后直接写入。

    目标是把来源不一致的数据统一成 replay 所需字段：
    observations/actions/next_observations/rewards/masks/dones，
    并确保：
    1. obs key 与 shape 与在线数据一致；
    2. action 是当前步 residual（action_dim 维）；
    3. reward/mask/done 都是标量语义。
    """
    required = {"observations", "next_observations"}
    if not required.issubset(transition.keys()):
        return None

    explicit_residual = bool(
        transition.get("is_residual", False)
        or transition.get("action_is_residual", False)
        or str(transition.get("action_type", "")).lower() == "residual"
        or ("residual_action" in transition)
        or ("residual_actions" in transition)
        or ("a_res" in transition)
    )
    if (not explicit_residual) and (not accept_plain_preconverted):
        return None

    obs = _normalize_obs_dict_for_buffer(transition["observations"], sample_obs_template)
    next_obs = _normalize_obs_dict_for_buffer(transition["next_observations"], sample_obs_template)
    if obs is None or next_obs is None:
        return None

    step_idx = _extract_chunk_step_index(transition, chunk_horizon)
    has_step_key = _has_chunk_step_key(transition)
    action_source = transition.get("residual_action", transition.get("residual_actions", transition.get("a_res")))
    if action_source is None:
        action_source = transition.get("actions", None)
    if action_source is None:
        return None

    try:
        arr = np.asarray(action_source, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] in (action_dim, full_action_dim):
                seq = arr.reshape(1, arr.shape[0])
            elif arr.shape[0] % full_action_dim == 0:
                seq = arr.reshape(-1, full_action_dim)
            elif arr.shape[0] % action_dim == 0:
                seq = arr.reshape(-1, action_dim)
            else:
                return None
        elif arr.ndim == 2:
            if arr.shape[1] in (action_dim, full_action_dim):
                seq = arr
            else:
                return None
        else:
            return None

        if seq.shape[0] > 1 and not has_step_key:
            return None
        row = seq[min(step_idx, seq.shape[0] - 1)]
        if clip_residual_to_unit:
            row = np.clip(row, -1.0, 1.0)
        if row.shape[0] == full_action_dim and action_dim != full_action_dim:
            row = row[control_indices]
        if row.shape[0] != action_dim:
            return None
    except Exception:  # noqa: BLE001
        return None

    done = _safe_bool(transition.get("dones", transition.get("done", False)), default=False)
    mask = _safe_float(transition.get("masks", 0.0 if done else 1.0), default=0.0 if done else 1.0)
    reward = _reduce_reward(transition.get("rewards", transition.get("reward", 0.0)), default=0.0)

    return {
        "observations": obs,
        "actions": np.asarray(row, dtype=np.float32),
        "next_observations": next_obs,
        "rewards": np.float32(reward),
        "masks": np.float32(mask),
        "dones": bool(done),
    }


def _convert_expert_transition_to_residual(
    transition: Dict[str, Any],
    *,
    sample_obs_template: Dict[str, np.ndarray],
    action_dim: int,
    full_action_dim: int,
    chunk_horizon: int,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    residual_xi: float,
    expert_reference_scale: float,
    clip_residual_to_unit: bool,
    require_next_obs: bool,
    prompt: str,
    openpi_client: OpenPIChunkClient,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    normalizer: Optional[StateActionNormalizer] = None,
    debug_logger: Optional[logging.Logger] = None,
    debug_log_counter: Optional[List[int]] = None,
) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    将专家 transition 转成残差 transition。
    关键公式：a_res_raw = (a_expert - a_base) / (limits * xi * expert_reference_scale)。
    """
    def _log_fail(reason: str) -> None:
        if debug_logger is not None and debug_log_counter is not None and debug_log_counter[0] < 1:
            debug_logger.warning("offline expert->residual convert failed (reason): %s", reason)
            debug_log_counter[0] += 1

    obs_raw = _extract_raw_obs_from_transition(transition, next_obs=False)
    if obs_raw is None:
        _log_fail("obs_raw is None (transition missing observations/next_observations)")
        return None
    step_idx = _extract_chunk_step_index(transition, chunk_horizon)
    has_step_key = _has_chunk_step_key(transition)

    base_seq = _extract_action_sequence_by_keys(
        transition,
        keys=("base_chunk", "base_action_chunk", "base_actions"),
        full_action_dim=full_action_dim,
    )
    if base_seq is None:
        openpi_chunk, _ = openpi_client.infer_chunk(obs_raw, prompt)
        base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
    else:
        base_chunk = select_action_chunk_window(base_seq, horizon=chunk_horizon)

    expert_seq = _extract_action_sequence_by_keys(
        transition,
        keys=("expert_action_chunk", "expert_chunk", "expert_actions", "action_chunk", "actions", "action"),
        full_action_dim=full_action_dim,
    )
    if expert_seq is None:
        _log_fail("expert_seq is None (transition missing expert_action_chunk/action)")
        return None
    if expert_seq.shape[0] > 1 and not has_step_key:
        _log_fail("expert_seq has multiple rows but no chunk_step key")
        return None
    expert_action = expert_seq[min(step_idx, expert_seq.shape[0] - 1)]
    base_action = base_chunk[step_idx]

    obs_input_raw = build_residual_step_obs(
        obs_raw,
        base_action,
        image_keys=image_keys,
        stack_horizon=stack_horizon,
        normalizer=normalizer,
    )
    obs_input = _normalize_obs_dict_for_buffer(obs_input_raw, sample_obs_template)
    if obs_input is None:
        raw_keys = set(obs_input_raw.keys()) if isinstance(obs_input_raw, dict) else set()
        tpl_keys = set(sample_obs_template.keys())
        raw_shapes = {k: np.asarray(obs_input_raw[k]).shape for k in raw_keys} if isinstance(obs_input_raw, dict) else {}
        tpl_shapes = {k: np.asarray(sample_obs_template[k]).shape for k in tpl_keys}
        _log_fail(
            "obs_input normalize failed (key or shape mismatch). "
            f"raw_keys={raw_keys} tpl_keys={tpl_keys} raw_shapes={raw_shapes} tpl_shapes={tpl_shapes}"
        )
        return None

    # 将"专家最终动作"映射到"残差原始动作"域：
    # residual_raw = (a_expert - a_base) / (limits * xi * scale)
    # 这样离线数据与在线 residual policy 输出空间保持一致（在线执行前会先乘 xi）。
    scale = max(float(expert_reference_scale), 1e-6)
    xi = max(float(residual_xi), 1e-6)
    denom = residual_limits * xi * scale
    raw_residual = (expert_action[control_indices] - base_action[control_indices]) / denom

    clipped_values = int(np.count_nonzero((raw_residual < -1.0) | (raw_residual > 1.0)))
    if clip_residual_to_unit:
        raw_residual = np.clip(raw_residual, -1.0, 1.0)

    residual_step_action = raw_residual.reshape(-1).astype(np.float32)
    if residual_step_action.shape[0] != action_dim:
        _log_fail(f"residual_step_action shape {residual_step_action.shape} != action_dim {action_dim}")
        return None

    done = _safe_bool(
        transition.get("dones", transition.get("done", transition.get("terminated", False))),
        default=False,
    )
    reward = _reduce_reward(
        transition.get("rewards", transition.get("reward", transition.get("success", 0.0))),
        default=0.0,
    )

    # next_obs 的构建要保证“时序对齐”：
    # - 若还在同一个 VLA chunk 内，next 仍配同一 base_chunk；
    # - 若当前步是 chunk 尾步，则为 next_obs 重新请求下一段 base_chunk。
    next_obs_raw = _extract_raw_obs_from_transition(transition, next_obs=True)
    if done:
        next_obs_input = _zero_obs_like(obs_input)
        mask = 0.0
    elif next_obs_raw is None:
        if require_next_obs:
            _log_fail("next_obs_raw is None but require_next_obs=True")
            return None
        next_obs_input = _zero_obs_like(obs_input)
        mask = 0.0
        done = True
    else:
        if step_idx < (chunk_horizon - 1):
            next_base_action = base_chunk[step_idx + 1]
        else:
            next_base_seq = _extract_action_sequence_by_keys(
                transition,
                keys=("next_base_chunk", "next_base_action_chunk", "next_base_actions"),
                full_action_dim=full_action_dim,
            )
            if next_base_seq is None:
                next_openpi_chunk, _ = openpi_client.infer_chunk(next_obs_raw, prompt)
                next_base_chunk = select_action_chunk_window(next_openpi_chunk, horizon=chunk_horizon)
            else:
                next_base_chunk = select_action_chunk_window(next_base_seq, horizon=chunk_horizon)
            next_base_action = next_base_chunk[0]

        next_obs_input_raw = build_residual_step_obs(
            next_obs_raw,
            next_base_action,
            image_keys=image_keys,
            stack_horizon=stack_horizon,
            normalizer=normalizer,
        )
        next_obs_input = _normalize_obs_dict_for_buffer(next_obs_input_raw, sample_obs_template)
        if next_obs_input is None:
            _log_fail("next_obs_input normalize failed (shape mismatch with sample_obs_template)")
            return None
        mask = 1.0

    mask_value = transition.get("masks", None)
    if mask_value is not None:
        mask = _safe_float(mask_value, default=mask)

    return (
        {
            "observations": obs_input,
            "actions": residual_step_action,
            "next_observations": _clone_obs_dict(next_obs_input),
            "rewards": np.float32(reward),
            "masks": np.float32(mask),
            "dones": bool(done),
        },
        clipped_values,
    )


def _load_offline_residual_buffer(
    cfg: DictConfig,
    *,
    sample_obs_template: Dict[str, np.ndarray],
    offline_buffer: ReplayBuffer,
    action_dim: int,
    full_action_dim: int,
    chunk_horizon: int,
    control_indices: np.ndarray,
    residual_limits: np.ndarray,
    residual_xi: float,
    openpi_client: OpenPIChunkClient,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    logger: logging.Logger,
    normalizer: Optional[StateActionNormalizer] = None,
) -> Dict[str, int]:
    """加载离线数据并写入离线 buffer。"""
    stats = {
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "clipped_values": 0,
        "errors": 0,
    }
    max_error_logs = 20
    logged_errors = 0
    debug_log_counter: List[int] = [0]

    offline_paths = _resolve_offline_paths(cfg.offline.dataset_paths, Path.cwd())
    stats["files_total"] = len(offline_paths)
    logger.info("offline dataset_paths resolved: %d pkl files found", len(offline_paths))
    if not offline_paths:
        logger.warning("offline.enabled=true but offline.dataset_paths is empty")
        return stats

    max_transitions = int(cfg.offline.max_transitions) if cfg.offline.max_transitions is not None else None

    # 预统计总 transition 数，用于进度条
    total_transitions = 0
    for path in offline_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                total_transitions += len(_collect_transitions_from_payload(payload))
            except Exception:  # noqa: S110
                pass
    if max_transitions is not None:
        total_transitions = min(total_transitions, max_transitions)

    pbar = tqdm(
        total=total_transitions,
        desc="Converting offline to residual",
        unit="trans",
        dynamic_ncols=True,
    )

    for path in offline_paths:
        if max_transitions is not None and stats["inserted"] >= max_transitions:
            break
        if not path.exists():
            stats["files_missing"] += 1
            logger.warning("offline dataset not found: %s", path)
            continue

        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            stats["skipped"] += 1
            logger.warning("failed to load offline dataset %s: %s", path, exc)
            continue

        transitions = _collect_transitions_from_payload(payload)
        if not transitions:
            logger.warning("no transitions found in offline dataset: %s", path)
            continue
        stats["files_loaded"] += 1

        for transition_idx, transition in enumerate(transitions):
            if max_transitions is not None and stats["inserted"] >= max_transitions:
                break
            stats["candidates"] += 1
            pbar.update(1)
            pbar.set_postfix(inserted=stats["inserted"], skipped=stats["skipped"], refresh=False)

            if not isinstance(transition, dict):
                stats["skipped"] += 1
                continue

            try:
                prepared = _prepare_preconverted_transition(
                    transition,
                    sample_obs_template=sample_obs_template,
                    action_dim=action_dim,
                    full_action_dim=full_action_dim,
                    control_indices=control_indices,
                    chunk_horizon=chunk_horizon,
                    accept_plain_preconverted=bool(cfg.offline.accept_plain_preconverted),
                    clip_residual_to_unit=bool(cfg.offline.clip_residual_to_unit),
                )
            except Exception as exc:  # noqa: BLE001
                stats["skipped"] += 1
                stats["errors"] += 1
                if logged_errors < max_error_logs:
                    logger.warning(
                        "offline preconverted parse failed file=%s transition=%s: %s",
                        path,
                        transition_idx,
                        exc,
                    )
                    logged_errors += 1
                continue
            if prepared is not None:
                offline_buffer.insert(prepared)
                stats["inserted"] += 1
                continue

            try:
                converted = _convert_expert_transition_to_residual(
                    transition,
                    sample_obs_template=sample_obs_template,
                    action_dim=action_dim,
                    full_action_dim=full_action_dim,
                    chunk_horizon=chunk_horizon,
                    control_indices=control_indices,
                    residual_limits=residual_limits,
                    residual_xi=residual_xi,
                    expert_reference_scale=float(cfg.offline.expert_reference_scale),
                    clip_residual_to_unit=bool(cfg.offline.clip_residual_to_unit),
                    require_next_obs=bool(cfg.offline.require_next_obs),
                    prompt=str(cfg.task.prompt),
                    openpi_client=openpi_client,
                    image_keys=image_keys,
                    stack_horizon=stack_horizon,
                    normalizer=normalizer,
                    debug_logger=logger,
                    debug_log_counter=debug_log_counter,
                )
            except Exception as exc:  # noqa: BLE001
                stats["skipped"] += 1
                stats["errors"] += 1
                if logged_errors < max_error_logs:
                    logger.warning(
                        "offline expert->residual convert failed file=%s transition=%s: %s",
                        path,
                        transition_idx,
                        exc,
                    )
                    logged_errors += 1
                continue
            if converted is None:
                stats["skipped"] += 1
                continue

            payload_dict, clipped_values = converted
            stats["clipped_values"] += int(clipped_values)
            offline_buffer.insert(payload_dict)
            stats["inserted"] += 1

    pbar.close()
    return stats


def _sample_mixed_batch(
    online_buffer: ReplayBuffer,
    offline_buffer: Optional[ReplayBuffer],
    *,
    batch_size: int,
    offline_ratio: float,
    symmetric_replay: bool = False,
) -> Tuple[Dict[str, Any], int, int]:
    """
    按 offline_ratio 混采 batch，返回 (batch, online_bs, offline_bs)。

    注意：这里沿 batch 维拼接（axis=0），保证最终 batch 结构仍是
    与 agent.update_high_utd 兼容的标准 replay batch 格式。
    """
    if offline_buffer is None or len(offline_buffer) == 0 or ((not symmetric_replay) and offline_ratio <= 0.0):
        return online_buffer.sample(batch_size=batch_size), int(batch_size), 0

    if symmetric_replay:
        offline_bs = int(batch_size // 2)
        online_bs = int(batch_size - offline_bs)
    else:
        offline_bs = int(round(batch_size * offline_ratio))
        offline_bs = max(0, min(batch_size, offline_bs))
        online_bs = int(batch_size - offline_bs)

    if offline_bs == 0:
        return online_buffer.sample(batch_size=batch_size), int(batch_size), 0
    if online_bs == 0:
        return offline_buffer.sample(batch_size=batch_size), 0, int(batch_size)

    online_batch = online_buffer.sample(batch_size=online_bs)
    offline_batch = offline_buffer.sample(batch_size=offline_bs)
    mixed_batch = concat_batches(offline_batch, online_batch, axis=0)
    return mixed_batch, int(online_bs), int(offline_bs)


def _scheduled_scalar(
    *,
    target_value: float,
    global_policy_step: int,
    scheduler_cfg: Optional[DictConfig],
    min_key: str,
    default_min: float,
    name: str,
) -> float:
    """
    通用标量调度器（linear/cosine）。
    - scheduler 未启用：直接返回 target_value；
    - 启用后：从 min_value 退火到 target_value。
    """
    target_value = float(target_value)
    if scheduler_cfg is None or (not bool(scheduler_cfg.get("enabled", False))):
        return target_value

    min_value = float(scheduler_cfg.get(min_key, default_min))
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    anneal_steps = int(scheduler_cfg.get("anneal_steps", 1))
    sched_type = str(scheduler_cfg.get("type", "linear")).lower()

    if global_policy_step < warmup_steps:
        progress = 0.0
    else:
        progress = float(global_policy_step - warmup_steps) / float(max(1, anneal_steps))
    progress = float(np.clip(progress, 0.0, 1.0))

    if sched_type == "linear":
        factor = progress
    elif sched_type == "cosine":
        factor = 0.5 * (1.0 - float(np.cos(np.pi * progress)))
    else:
        raise ValueError(f"Unsupported {name}.type: {sched_type}")

    value = min_value + (target_value - min_value) * factor
    if target_value >= min_value:
        value = min(value, target_value)
    else:
        value = max(value, target_value)
    return float(value)


def _scheduled_residual_scale(
    cfg: DictConfig,
    *,
    phase_scale: float,
    global_policy_step: int,
) -> float:
    """
    计算当前步 residual scale（兼容历史配置）。
    """
    phase_scale = float(phase_scale)
    if phase_scale <= 0.0:
        return 0.0
    scheduler_cfg = cfg.training.get("residual_scale_scheduler", None)
    return _scheduled_scalar(
        target_value=phase_scale,
        global_policy_step=global_policy_step,
        scheduler_cfg=scheduler_cfg,
        min_key="min_scale",
        default_min=0.0,
        name="residual_scale_scheduler",
    )


def _scheduled_xi(
    cfg: DictConfig,
    *,
    base_xi: float,
    global_policy_step: int,
) -> float:
    """
    计算当前步 xi（论文语义：xi 由 scheduler 调参）。
    """
    base_xi = float(max(0.0, base_xi))
    scheduler_cfg = cfg.training.get("xi_scheduler", None)
    xi = _scheduled_scalar(
        target_value=base_xi,
        global_policy_step=global_policy_step,
        scheduler_cfg=scheduler_cfg,
        min_key="min_xi",
        default_min=base_xi,
        name="xi_scheduler",
    )
    return float(max(0.0, xi))


def _bootstrap_offline_with_base_success(
    cfg: DictConfig,
    *,
    env: RoboTwinTaskEnv,
    openpi_client: OpenPIChunkClient,
    offline_buffer: ReplayBuffer,
    sample_obs_template: Dict[str, np.ndarray],
    action_dim: int,
    image_keys: Tuple[str, ...],
    stack_horizon: int,
    chunk_horizon: int,
    logger: logging.Logger,
    normalizer: Optional[StateActionNormalizer] = None,
) -> Dict[str, int]:
    """
    用 base policy 自动收集成功轨迹，写入 offline buffer（residual action 全零）。
    这是 Stage-1 里“成功 base rollout 离线初始化”的实现。
    """
    stats = {
        "enabled": 0,
        "attempts": 0,
        "episodes_collected": 0,
        "success_episodes": 0,
        "inserted": 0,
        "seed_start": 0,
        "seed_next": 0,
    }

    bootstrap_cfg = cfg.offline.get("bootstrap_base", None)
    if bootstrap_cfg is None or (not bool(bootstrap_cfg.get("enabled", False))):
        return stats

    stats["enabled"] = 1
    target_success_episodes = int(bootstrap_cfg.get("success_episodes", 0))
    if target_success_episodes <= 0:
        logger.warning("offline.bootstrap_base.enabled=true but success_episodes<=0, skip bootstrap")
        return stats

    max_seed_attempts = int(bootstrap_cfg.get("max_seed_attempts", max(1000, target_success_episodes * 100)))
    seed_base_cfg = bootstrap_cfg.get("seed_base", None)
    if seed_base_cfg is None:
        seed_cursor = int(cfg.task.seed_base) + 1_000_000
    else:
        seed_cursor = int(seed_base_cfg)
    stats["seed_start"] = int(seed_cursor)

    max_ep_steps_override = bootstrap_cfg.get("max_env_steps_per_episode", None)
    only_success = bool(bootstrap_cfg.get("only_success", True))

    bootstrap_env = RoboTwinTaskEnv(
        task_name=env.task_name,
        task_args=dict(env.task_args),
        prompt=str(env.prompt),
        max_setup_retries=int(env.max_setup_retries),
        instruction_type=env.instruction_type,
        logger=logger,
    )
    try:
        while stats["attempts"] < max_seed_attempts and stats["success_episodes"] < target_success_episodes:
            seed = int(seed_cursor)
            seed_cursor += 1
            stats["attempts"] += 1

            episode_info = None
            if bool(cfg.training.expert_check):
                passed, episode_info = bootstrap_env.expert_precheck(seed=seed, episode_id=-1)
                if not passed:
                    continue

            try:
                obs_raw = bootstrap_env.reset(seed=seed, episode_id=-1, episode_info=episode_info)
            except Exception as exc:  # noqa: BLE001
                logger.warning("bootstrap reset failed seed=%s: %s", seed, exc)
                continue

            episode_transitions: List[Dict[str, Any]] = []
            success = False
            episode_steps = 0
            max_episode_steps = int(bootstrap_env.step_limit)
            if max_ep_steps_override is not None:
                max_episode_steps = min(max_episode_steps, int(max_ep_steps_override))

            while episode_steps < max_episode_steps:
                openpi_chunk, _ = openpi_client.infer_chunk(obs_raw, bootstrap_env.current_instruction)
                base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
                next_obs_raw = obs_raw
                decision_done = False

                for chunk_step in range(chunk_horizon):
                    if episode_steps >= max_episode_steps:
                        decision_done = True
                        break

                    obs_input = build_residual_step_obs(
                        next_obs_raw,
                        base_chunk[chunk_step],
                        image_keys=image_keys,
                        stack_horizon=stack_horizon,
                        normalizer=normalizer,
                    )
                    obs_input = _normalize_obs_dict_for_buffer(obs_input, sample_obs_template)
                    if obs_input is None:
                        decision_done = True
                        break

                    final_action = base_chunk[chunk_step]
                    next_obs_raw, reward, env_done, _, info = bootstrap_env.step(final_action)
                    episode_steps += 1
                    success = bool(info["success"])
                    timeout = bool(episode_steps >= max_episode_steps)
                    done = bool(env_done or timeout)

                    if done:
                        next_obs_input = _zero_obs_like(obs_input)
                        mask = 0.0
                    elif chunk_step < (chunk_horizon - 1):
                        next_obs_input_raw = build_residual_step_obs(
                            next_obs_raw,
                            base_chunk[chunk_step + 1],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            normalizer=normalizer,
                        )
                        next_obs_input = _normalize_obs_dict_for_buffer(next_obs_input_raw, sample_obs_template)
                        if next_obs_input is None:
                            next_obs_input = _zero_obs_like(obs_input)
                            mask = 0.0
                            done = True
                        else:
                            mask = 1.0
                    else:
                        next_openpi_chunk, _ = openpi_client.infer_chunk(next_obs_raw, bootstrap_env.current_instruction)
                        next_base_chunk = select_action_chunk_window(next_openpi_chunk, horizon=chunk_horizon)
                        next_obs_input_raw = build_residual_step_obs(
                            next_obs_raw,
                            next_base_chunk[0],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            normalizer=normalizer,
                        )
                        next_obs_input = _normalize_obs_dict_for_buffer(next_obs_input_raw, sample_obs_template)
                        if next_obs_input is None:
                            next_obs_input = _zero_obs_like(obs_input)
                            mask = 0.0
                            done = True
                        else:
                            mask = 1.0

                    episode_transitions.append(
                        {
                            "observations": _clone_obs_dict(obs_input),
                            "actions": np.zeros((action_dim,), dtype=np.float32),
                            "next_observations": _clone_obs_dict(next_obs_input),
                            "rewards": np.float32(reward),
                            "masks": np.float32(mask),
                            "dones": bool(done),
                        }
                    )

                    if done:
                        decision_done = True
                        break

                obs_raw = next_obs_raw
                if decision_done:
                    break

            should_keep = bool(success or (not only_success))
            if should_keep:
                for transition in episode_transitions:
                    offline_buffer.insert(transition)
                stats["inserted"] += int(len(episode_transitions))
                stats["episodes_collected"] += 1
                stats["success_episodes"] += int(success)
    finally:
        try:
            bootstrap_env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass

    stats["seed_next"] = int(seed_cursor)
    return stats


def _pretrain_critic_with_calql(
    cfg: DictConfig,
    *,
    agent: DrQAgent,
    offline_buffer: Optional[ReplayBuffer],
    logger: logging.Logger,
    tb_writer: Optional["SummaryWriter"] = None,
) -> Dict[str, Any]:
    """用离线 buffer 做 Cal-QL 风格 critic 预训练。"""
    calql_cfg = cfg.training.get("calql_pretrain", None)
    if calql_cfg is None or (not bool(calql_cfg.get("enabled", False))):
        return {"enabled": 0, "steps": 0}

    warm_steps = int(calql_cfg.get("steps", 0))
    warm_batch_size = int(calql_cfg.get("batch_size", cfg.replay.batch_size))
    calql_alpha = float(calql_cfg.get("alpha", 0.0))
    calql_n_actions = int(calql_cfg.get("n_actions", cfg.sac.get("cql_n_actions", 10)))
    calql_temperature = float(calql_cfg.get("temperature", cfg.sac.get("cql_temperature", 1.0)))
    if warm_steps <= 0 or calql_alpha <= 0.0 or offline_buffer is None or len(offline_buffer) == 0:
        return {
            "enabled": 0,
            "steps": 0,
            "requested_steps": int(warm_steps),
            "offline_buffer_size": int(len(offline_buffer) if offline_buffer is not None else 0),
        }

    info_last: Dict[str, Any] = {}
    pbar = tqdm(
        range(warm_steps),
        desc="Cal-QL critic pretrain",
        unit="step",
        dynamic_ncols=True,
    )
    for step in pbar:
        batch = offline_buffer.sample(batch_size=warm_batch_size)
        agent, info_last = agent.update_critics_calql(
            batch,
            calql_alpha=calql_alpha,
            calql_n_actions=calql_n_actions,
            calql_temperature=calql_temperature,
        )
        if step % 50 == 0 or step == warm_steps - 1:
            loss_str = f"loss={info_last.get('critic_loss', 0):.3f}"
            if "predicted_qs" in info_last:
                loss_str += f" Q={info_last['predicted_qs']:.2f}"
            pbar.set_postfix_str(loss_str)
        if tb_writer is not None:
            for tb_key, info_key in (
                ("calql_pretrain/critic_loss", "critic_loss"),
                ("calql_pretrain/critic_td_loss", "critic_td_loss"),
                ("calql_pretrain/critic_cql_penalty", "critic_cql_penalty"),
                ("calql_pretrain/predicted_qs", "predicted_qs"),
                ("calql_pretrain/target_qs", "target_qs"),
            ):
                if info_key in info_last:
                    tb_writer.add_scalar(tb_key, float(info_last[info_key]), step)
    logger.info(
        (
            "Cal-QL critic pretrain done: steps=%s batch_size=%s offline_buffer=%s "
            "alpha=%.4f n_actions=%s temp=%.4f"
        ),
        warm_steps,
        warm_batch_size,
        len(offline_buffer),
        calql_alpha,
        calql_n_actions,
        calql_temperature,
    )
    return {
        "enabled": 1,
        "steps": int(warm_steps),
        "batch_size": int(warm_batch_size),
        "alpha": float(calql_alpha),
        "n_actions": int(calql_n_actions),
        "temperature": float(calql_temperature),
        "last_info": info_last,
    }


class _AsyncLearner:
    """
    进程内异步采样-学习协调器：
    - actor 在主线程与环境交互；
    - learner 在线程中持续从 replay 采样并更新；
    - 每隔 update_frequency 次 learner 更新，将参数同步给 actor。
    """

    def __init__(
        self,
        *,
        learner_agent: DrQAgent,
        actor_agent: DrQAgent,
        online_buffer: ReplayBuffer,
        offline_buffer: Optional[ReplayBuffer],
        batch_size: int,
        offline_ratio: float,
        symmetric_replay: bool,
        training_starts: int,
        utd_ratio: int,
        update_frequency: int,
        idle_sleep_sec: float,
    ) -> None:
        self.learner_agent = learner_agent
        self.actor_agent = actor_agent
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.batch_size = int(batch_size)
        self.offline_ratio = float(offline_ratio)
        self.symmetric_replay = bool(symmetric_replay)
        self.training_starts = int(training_starts)
        self.utd_ratio = int(utd_ratio)
        self.update_frequency = max(1, int(update_frequency))
        self.idle_sleep_sec = float(max(1e-4, idle_sleep_sec))

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.replay_lock = threading.Lock()
        self.actor_lock = threading.Lock()
        self.learner_lock = threading.Lock()

        self.update_steps = 0
        self.last_update_info: Dict[str, Any] = {}

    def _sync_actor(self, params: Dict[str, Any], target_params: Dict[str, Any]) -> None:
        with self.actor_lock:
            self.actor_agent.state.params = params
            self.actor_agent.state.target_params = target_params

    def sync_now(self) -> None:
        with self.learner_lock:
            params = self.learner_agent.state.params
            target_params = self.learner_agent.state.target_params
        self._sync_actor(params, target_params)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="robotwin-async-learner")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        # 停止前做一次最终同步，保证 actor 参数是最新版本。
        self.sync_now()

    def sample_actor_action(self, obs_input: Dict[str, np.ndarray], action_dim: int) -> np.ndarray:
        with self.actor_lock:
            sampled = self.actor_agent.sample_actions(obs_input, deterministic=False)
        return as_numpy_action(sampled, action_dim)

    def save_checkpoint(self, checkpoint_dir: str, *, step: int, keep: int) -> None:
        with self.learner_lock:
            save_agent_checkpoint(checkpoint_dir, self.learner_agent, step=step, keep=keep)

    def get_last_update_info(self) -> Dict[str, Any]:
        with self.learner_lock:
            return dict(self.last_update_info)

    def get_update_steps(self) -> int:
        with self.learner_lock:
            return int(self.update_steps)

    def _sample_batch(self) -> Optional[Tuple[Dict[str, Any], int, int]]:
        with self.replay_lock:
            if len(self.online_buffer) < self.training_starts:
                return None
            batch, online_bs, offline_bs = _sample_mixed_batch(
                self.online_buffer,
                self.offline_buffer,
                batch_size=self.batch_size,
                offline_ratio=self.offline_ratio,
                symmetric_replay=self.symmetric_replay,
            )
        return batch, online_bs, offline_bs

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sampled = self._sample_batch()
            if sampled is None:
                time.sleep(self.idle_sleep_sec)
                continue
            batch, online_bs, offline_bs = sampled

            params_to_sync: Optional[Dict[str, Any]] = None
            target_params_to_sync: Optional[Dict[str, Any]] = None
            with self.learner_lock:
                self.learner_agent, info = self.learner_agent.update_high_utd(
                    batch,
                    utd_ratio=self.utd_ratio,
                )
                info["online_batch_size"] = int(online_bs)
                info["offline_batch_size"] = int(offline_bs)
                self.last_update_info = info
                self.update_steps += 1
                if self.update_steps % self.update_frequency == 0:
                    params_to_sync = self.learner_agent.state.params
                    target_params_to_sync = self.learner_agent.state.target_params

            if params_to_sync is not None and target_params_to_sync is not None:
                self._sync_actor(params_to_sync, target_params_to_sync)


@hydra.main(version_base=None, config_path="../conf", config_name="train_residual_sac")
def main(cfg: DictConfig) -> None:
    """
    训练入口：负责环境交互、replay 写入、策略更新、日志与 checkpoint。

    训练主循环层级：
    phase -> episode -> chunk(来自 OpenPI) -> step(残差每步推理/执行/入库)。
    """
    # Hydra 运行目录：所有日志和模型默认写到这里。
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("train_residual_sac")

    # 固定随机性，减少重复实验波动。
    set_global_seeds(int(cfg.seed))

    env_backend = str(cfg.get("env", {}).get("backend", "local")).lower()
    if env_backend not in {"local", "remote"}:
        raise ValueError(f"env.backend must be 'local' or 'remote', got {env_backend}")

    robo_root = resolve_robo_root(cfg.robo_root) if env_backend == "local" else None
    # local 模式下环境在当前进程内实例化，需要 RoboTwin 路径与 cwd。
    if env_backend == "local":
        setup_robotwin_pythonpath(robo_root)
        os.chdir(robo_root)
        logger.info("RoboTwin root: %s", robo_root)
    else:
        logger.info("Use remote env backend; skip local RoboTwin import/chdir")
    logger.info("Hydra run dir: %s", run_dir)
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    logger.info("Config:\n%s", resolved_yaml)
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as _f:
        _f.write(resolved_yaml)

    if env_backend == "local":
        assert robo_root is not None
        task_args = load_task_args(robo_root, str(cfg.task.name), str(cfg.task.task_config))
    else:
        # remote 模式下 task_args 由 robotwin2 环境中的 server 解析，这里只传最小信息。
        task_args = {"task_config": str(cfg.task.task_config)}

    instruction_type = str(cfg.task.get("instruction_type", "seen"))
    # 任务环境封装：对底层 setup_demo / take_action / get_obs 做了统一接口。
    if env_backend == "local":
        env = RoboTwinTaskEnv(
            task_name=str(cfg.task.name),
            task_args=task_args,
            prompt=str(cfg.task.prompt),
            max_setup_retries=int(cfg.task.setup_retries),
            instruction_type=instruction_type,
            logger=logger,
        )
    else:
        remote_cfg = cfg.get("env", {}).get("remote", {})
        env = RemoteRoboTwinTaskEnv(
            host=str(remote_cfg.get("host", "127.0.0.1")),
            port=int(remote_cfg.get("port", 9100)),
            timeout_sec=float(remote_cfg.get("timeout_sec", 120.0)),
            robo_root=remote_cfg.get("robo_root", cfg.robo_root),
            task_name=str(cfg.task.name),
            task_args=task_args,
            prompt=str(cfg.task.prompt),
            max_setup_retries=int(cfg.task.setup_retries),
            instruction_type=instruction_type,
            logger=logger,
        )

    # ---------- 归一化器（可选）----------
    norm_cfg = cfg.get("normalization", None)
    normalizer: StateActionNormalizer | None = None
    if norm_cfg is not None and bool(norm_cfg.get("enabled", False)):
        stats_dir = norm_cfg.get("stats_dir", None)
        normalizer = load_normalizer(str(cfg.task.name), stats_dir=stats_dir)
        if normalizer is not None:
            logger.info("State/action normalizer loaded for task=%s", cfg.task.name)
    else:
        logger.info("Normalization disabled (normalization.enabled not set)")

    # OpenPI 客户端：提供 base policy chunk 推理。
    openpi_client = OpenPIChunkClient(
        host=str(cfg.openpi.host),
        port=int(cfg.openpi.port),
        logger=logger,
    )

    # 图像键与堆叠维度：当前实现只支持 stack_horizon=1（单帧）。
    image_keys = resolve_image_keys(cfg)
    stack_horizon = int(cfg.sac.obs_stack_horizon)
    if stack_horizon != 1:
        raise ValueError("Only obs_stack_horizon=1 is currently supported")

    control_indices = resolve_control_indices_from_cfg(cfg)

    residual_limits = build_residual_limits(
        control_indices,
        arm_limit=float(cfg.residual.arm_delta_limit),
        gripper_limit=float(cfg.residual.gripper_delta_limit),
    )
    residual_xi = float(cfg.residual.get("xi", 1.0))
    if residual_xi <= 0.0:
        raise ValueError(f"residual.xi must be positive, got {residual_xi}")

    # VLA chunk 长度由配置决定；残差每步输出一次（维度由 control_indices 决定）。
    chunk_horizon = int(cfg.residual.chunk_horizon)
    if chunk_horizon <= 0:
        raise ValueError(f"residual.chunk_horizon must be positive, got {chunk_horizon}")
    full_action_dim = int(cfg.robot_action_dim)
    per_step_action_dim = int(len(control_indices))
    action_dim = int(per_step_action_dim)
    logger.info(
        "Residual config: image_keys=%s action_dim=%s full_action_dim=%s action_indices=%s chunk_horizon=%s xi=%.4f",
        list(image_keys),
        action_dim,
        full_action_dim,
        control_indices.tolist(),
        chunk_horizon,
        residual_xi,
    )
    offline_enabled = bool(cfg.offline.enabled)
    offline_ratio = float(cfg.offline.ratio)
    if not (0.0 <= offline_ratio <= 1.0):
        raise ValueError(f"offline.ratio must be in [0,1], got {offline_ratio}")
    symmetric_replay = bool(cfg.offline.get("symmetric_replay", False))
    async_cfg = cfg.training.get("async", None)
    async_enabled = bool(async_cfg.get("enabled", False)) if async_cfg is not None else False
    async_update_frequency = int(async_cfg.get("update_frequency", 1)) if async_cfg is not None else 1
    async_idle_sleep_sec = float(async_cfg.get("idle_sleep_sec", 0.002)) if async_cfg is not None else 0.002
    if async_enabled and any((not bool(phase.get("train", True))) for phase in cfg.training.phases):
        logger.warning(
            "Detected non-train phase in training.phases; disable async mode to preserve phase semantics."
        )
        async_enabled = False
    logger.info(
        "Async collection-learning: enabled=%s update_frequency=%s idle_sleep_sec=%.4f",
        async_enabled,
        async_update_frequency,
        async_idle_sleep_sec,
    )
    warmup_base_episodes = int(cfg.training.get("warmup_base_episodes", 0))
    warmup_base_steps = int(cfg.training.get("warmup_base_steps", 0))
    max_online_env_steps = int(cfg.training.get("max_online_env_steps", 0))

    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(action_dim,),
        dtype=np.float32,
    )

    # 懒初始化：拿到首个 obs_input 后再构建 agent / replay（避免手写 shape）。
    # 这样可直接复用真实输入 shape，避免手工定义观测空间与网络输入不一致。
    agent: DrQAgent | None = None  # actor-side agent（同步模式下等同 learner）
    learner_agent: DrQAgent | None = None
    async_learner: _AsyncLearner | None = None
    replay_buffer: ReplayBuffer | None = None
    offline_buffer: ReplayBuffer | None = None
    offline_stats: Dict[str, int] = {
        "files_total": 0,
        "files_loaded": 0,
        "files_missing": 0,
        "candidates": 0,
        "inserted": 0,
        "skipped": 0,
        "clipped_values": 0,
        "errors": 0,
    }
    bootstrap_stats: Dict[str, Any] = {}
    warmstart_info: Dict[str, Any] = {"enabled": 0, "steps": 0}
    critic_warmstarted = False

    checkpoint_dir = Path(str(cfg.training.checkpoint_dir))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step_logger = JsonlLogger(run_dir / str(cfg.logging.step_log_file))
    episode_logger = JsonlLogger(run_dir / str(cfg.logging.episode_log_file))

    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    tb_step_period = int(cfg.logging.get("tb_step_period", 100))
    logger.info("TensorBoard log dir: %s (step period=%d)", run_dir / "tb", tb_step_period)

    global_env_step = 0
    global_policy_step = 0
    episode_id = 0
    total_success = 0
    recent_successes: deque[int] = deque(maxlen=20)
    skipped_seeds = 0
    seed_cursor = int(cfg.task.seed_base)
    stopped_by_env_budget = False

    try:
        # 分阶段训练（baseline / small_residual / full_residual）。
        for phase in cfg.training.phases:
            if max_online_env_steps > 0 and global_env_step >= max_online_env_steps:
                stopped_by_env_budget = True
                break
            phase_name = str(phase.name)
            phase_episodes = int(phase.episodes)
            phase_train = bool(phase.get("train", True))
            residual_scale = float(phase.residual_scale)
            phase_seed_attempts = 0
            max_seed_attempts_cfg = phase.get("max_seed_attempts", cfg.training.get("max_seed_attempts_per_phase", None))
            if max_seed_attempts_cfg is None:
                max_seed_attempts = max(1000, int(phase_episodes) * 100)
            else:
                max_seed_attempts = int(max_seed_attempts_cfg)

            logger.info(
                "Start phase=%s episodes=%s train=%s residual_scale=%.4f max_seed_attempts=%s",
                phase_name,
                phase_episodes,
                phase_train,
                residual_scale,
                max_seed_attempts,
            )

            phase_episode_count = 0
            while phase_episode_count < phase_episodes:
                if max_online_env_steps > 0 and global_env_step >= max_online_env_steps:
                    stopped_by_env_budget = True
                    break
                phase_seed_attempts += 1
                if phase_seed_attempts > max_seed_attempts:
                    raise RuntimeError(
                        "Exceeded max seed attempts in training phase. "
                        f"phase={phase_name}, attempts={phase_seed_attempts}, "
                        f"completed_phase_episodes={phase_episode_count}, skipped_seeds={skipped_seeds}"
                    )
                seed = int(seed_cursor)
                seed_cursor += 1

                episode_info = None
                if bool(cfg.training.expert_check):
                    passed, episode_info = env.expert_precheck(seed=seed, episode_id=episode_id)
                    if not passed:
                        skipped_seeds += 1
                        logger.warning("skip seed=%s in phase=%s: expert precheck failed", seed, phase_name)
                        continue

                obs_raw = env.reset(seed=seed, episode_id=episode_id, episode_info=episode_info)

                # 若本轮已提前算出 next_base_chunk，就缓存到下一轮复用，减少一次 OpenPI 调用。
                # 该缓存只跨“chunk 决策边界”生效，不跨 episode。
                cached_base_chunk: np.ndarray | None = None
                cached_infer_info: Dict[str, float | None] | None = None

                episode_success = False
                episode_return = 0.0
                episode_steps = 0

                max_episode_steps = int(env.step_limit)
                if cfg.training.max_env_steps_per_episode is not None:
                    max_episode_steps = min(max_episode_steps, int(cfg.training.max_env_steps_per_episode))

                # Stage-2 probing: 先用 base policy 随机 rollout 若干步做初始状态分布对齐，
                # 这些 probing 步仅用于“状态初始化”，不写入 residual replay。
                probing_steps_target = sample_probing_steps(cfg.training, episode_horizon=max_episode_steps)
                if probing_steps_target > 0:
                    probing_remaining = int(min(probing_steps_target, max_episode_steps - episode_steps))
                    while probing_remaining > 0 and episode_steps < max_episode_steps:
                        probe_chunk, probe_info = openpi_client.infer_chunk(obs_raw, env.current_instruction)
                        probe_base_chunk = select_action_chunk_window(probe_chunk, horizon=chunk_horizon)
                        probe_done = False
                        for probe_step in range(chunk_horizon):
                            if probing_remaining <= 0 or episode_steps >= max_episode_steps:
                                break
                            base_action = probe_base_chunk[probe_step]
                            next_obs_raw, reward, env_done, _, info = env.step(base_action)
                            episode_steps += 1
                            global_env_step += 1
                            probing_remaining -= 1
                            episode_return += float(reward)
                            episode_success = bool(info["success"])
                            budget_exhausted = bool(
                                max_online_env_steps > 0 and global_env_step >= max_online_env_steps
                            )

                            timeout = bool(episode_steps >= max_episode_steps)
                            done = bool(env_done or timeout or budget_exhausted)
                            step_logger.write(
                                {
                                    "global_env_step": int(global_env_step),
                                    "global_policy_step": int(global_policy_step),
                                    "episode_id": episode_id,
                                    "phase": phase_name,
                                    "episode_step": episode_steps,
                                    "seed": int(env.last_seed if env.last_seed is not None else seed),
                                    "is_probing": True,
                                    "replan_point": bool(probe_step == 0),
                                    "chunk_step": int(probe_step),
                                    "chunk_horizon": int(chunk_horizon),
                                    "infer_e2e_ms": probe_info.get("e2e_ms") if probe_step == 0 else None,
                                    "infer_policy_ms": probe_info.get("policy_ms") if probe_step == 0 else None,
                                    "infer_server_ms": probe_info.get("server_ms") if probe_step == 0 else None,
                                    "a_base": base_action.tolist(),
                                    "a_res": [0.0] * full_action_dim,
                                    "a_final": base_action.tolist(),
                                    "residual_scale": 0.0,
                                    "reward": float(reward),
                                    "done": bool(done),
                                    "success": bool(episode_success),
                                }
                            )
                            obs_raw = next_obs_raw
                            if done:
                                probe_done = True
                                break
                        if probe_done:
                            break
                    if (
                        episode_steps >= max_episode_steps
                        or episode_success
                        or (max_online_env_steps > 0 and global_env_step >= max_online_env_steps)
                    ):
                        episode_done = True
                    else:
                        episode_done = False
                else:
                    episode_done = False

                while (episode_steps < max_episode_steps) and (not episode_done):
                    # 1) 取 base chunk：优先复用缓存，否则调用 OpenPI 现推。
                    #    base_chunk 形状为 (chunk_horizon, 14)。
                    if cached_base_chunk is None:
                        openpi_chunk, infer_info = openpi_client.infer_chunk(obs_raw, env.current_instruction)
                        base_chunk = select_action_chunk_window(openpi_chunk, horizon=chunk_horizon)
                    else:
                        base_chunk = cached_base_chunk
                        infer_info = cached_infer_info or {
                            "e2e_ms": None,
                            "policy_ms": None,
                            "server_ms": None,
                        }
                        cached_base_chunk = None
                        cached_infer_info = None

                    next_obs_raw = obs_raw

                    # 2) 在同一 base chunk 内逐步执行：每步都推理一次残差动作。
                    #    即：VLA 每 chunk_horizon 步推一次；residual 每 1 步推一次。
                    for chunk_step in range(chunk_horizon):
                        if episode_steps >= max_episode_steps:
                            episode_done = True
                            break

                        # 将当前时刻观测 next_obs_raw 与当前步 base_action 组合成策略输入：
                        # obs_input = {多路图像, state=[joint(14), base_action(14)]}
                        obs_input = build_residual_step_obs(
                            next_obs_raw,
                            base_chunk[chunk_step],
                            image_keys=image_keys,
                            stack_horizon=stack_horizon,
                            normalizer=normalizer,
                        )

                        # 首次拿到真实观测形状后，初始化 agent 和 replay。
                        if agent is None:
                            learner_agent = build_drq_agent(
                                cfg,
                                sample_obs=obs_input,
                                action_dim=action_dim,
                                image_keys=image_keys,
                            )
                            replay_buffer = ReplayBuffer(
                                observation_space=_obs_space_from_sample(obs_input),
                                action_space=action_space,
                                capacity=int(cfg.replay.capacity),
                            )
                            if offline_enabled:
                                offline_buffer = ReplayBuffer(
                                    observation_space=_obs_space_from_sample(obs_input),
                                    action_space=action_space,
                                    capacity=int(cfg.offline.capacity),
                                )
                                if env_backend == "local":
                                    bootstrap_stats = _bootstrap_offline_with_base_success(
                                        cfg,
                                        env=env,
                                        openpi_client=openpi_client,
                                        offline_buffer=offline_buffer,
                                        sample_obs_template=obs_input,
                                        action_dim=action_dim,
                                        image_keys=image_keys,
                                        stack_horizon=stack_horizon,
                                        chunk_horizon=chunk_horizon,
                                        logger=logger,
                                        normalizer=normalizer,
                                    )
                                else:
                                    bootstrap_stats = {
                                        "enabled": 0,
                                        "skipped_reason": "remote_env_backend",
                                    }
                                    if bool(cfg.offline.get("bootstrap_base", {}).get("enabled", False)):
                                        logger.warning(
                                            "offline.bootstrap_base is enabled but skipped in remote env backend. "
                                            "Use local backend for bootstrap, or run a dedicated bootstrap pipeline."
                                        )
                                if bootstrap_stats.get("enabled", 0):
                                    logger.info(
                                        (
                                            "offline bootstrap done: success_episodes=%s/%s attempts=%s "
                                            "inserted=%s seed_range=[%s,%s)"
                                        ),
                                        bootstrap_stats.get("success_episodes", 0),
                                        int(cfg.offline.get("bootstrap_base", {}).get("success_episodes", 0)),
                                        bootstrap_stats.get("attempts", 0),
                                        bootstrap_stats.get("inserted", 0),
                                        bootstrap_stats.get("seed_start", 0),
                                        bootstrap_stats.get("seed_next", 0),
                                    )
                                offline_stats = _load_offline_residual_buffer(
                                    cfg,
                                    sample_obs_template=obs_input,
                                    offline_buffer=offline_buffer,
                                    action_dim=action_dim,
                                    full_action_dim=full_action_dim,
                                    chunk_horizon=chunk_horizon,
                                    control_indices=control_indices,
                                    residual_limits=residual_limits,
                                    residual_xi=residual_xi,
                                    openpi_client=openpi_client,
                                    image_keys=image_keys,
                                    stack_horizon=stack_horizon,
                                    logger=logger,
                                    normalizer=normalizer,
                                )
                                logger.info(
                                    (
                                        "offline buffer loaded: size=%s files_loaded=%s/%s "
                                        "candidates=%s inserted=%s skipped=%s clipped_values=%s errors=%s"
                                    ),
                                    len(offline_buffer),
                                    offline_stats["files_loaded"],
                                    offline_stats["files_total"],
                                    offline_stats["candidates"],
                                    offline_stats["inserted"],
                                    offline_stats["skipped"],
                                    offline_stats["clipped_values"],
                                    offline_stats["errors"],
                                )
                            if (not critic_warmstarted) and offline_enabled:
                                warmstart_info = _pretrain_critic_with_calql(
                                    cfg,
                                    agent=learner_agent,
                                    offline_buffer=offline_buffer,
                                    logger=logger,
                                    tb_writer=tb_writer,
                                )
                                critic_warmstarted = True

                            if async_enabled:
                                # 异步模式：actor 与 learner 使用不同实例，按频率同步参数。
                                agent = build_drq_agent(
                                    cfg,
                                    sample_obs=obs_input,
                                    action_dim=action_dim,
                                    image_keys=image_keys,
                                )
                                agent.state.params = learner_agent.state.params
                                agent.state.target_params = learner_agent.state.target_params
                                async_learner = _AsyncLearner(
                                    learner_agent=learner_agent,
                                    actor_agent=agent,
                                    online_buffer=replay_buffer,
                                    offline_buffer=offline_buffer if offline_enabled else None,
                                    batch_size=int(cfg.replay.batch_size),
                                    offline_ratio=offline_ratio,
                                    symmetric_replay=symmetric_replay,
                                    training_starts=int(cfg.training.training_starts),
                                    utd_ratio=int(cfg.sac.utd_ratio),
                                    update_frequency=async_update_frequency,
                                    idle_sleep_sec=async_idle_sleep_sec,
                                )
                                async_learner.start()
                            else:
                                # 同步模式：采样与学习共用一个 agent。
                                agent = learner_agent

                        assert agent is not None
                        assert learner_agent is not None
                        assert replay_buffer is not None

                        residual_scale_step = _scheduled_residual_scale(
                            cfg,
                            phase_scale=residual_scale,
                            global_policy_step=global_policy_step,
                        )
                        xi_step = _scheduled_xi(
                            cfg,
                            base_xi=residual_xi,
                            global_policy_step=global_policy_step,
                        )

                        # 3) 决定当前步 residual 动作来源：
                        #    - warmup_base_episodes / warmup_base_steps 阶段：强制 base-only（残差全零）
                        #    - residual_scale<=0: 强制全零（基线等价）
                        #    - warmup 或非训练阶段: 随机探索
                        #    - 其余: 策略采样
                        in_warmup_episode = bool(episode_id < warmup_base_episodes)
                        in_warmup_step = bool(warmup_base_steps > 0 and global_policy_step < warmup_base_steps)
                        if phase_train and (in_warmup_episode or in_warmup_step):
                            residual_step_action = np.zeros((action_dim,), dtype=np.float32)
                        elif residual_scale_step <= 0.0:
                            residual_step_action = np.zeros((action_dim,), dtype=np.float32)
                        elif (not phase_train) or (global_policy_step < int(cfg.training.random_steps)):
                            residual_step_action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
                            residual_step_action *= float(cfg.training.random_action_scale)
                        else:
                            if async_learner is not None:
                                residual_step_action = async_learner.sample_actor_action(obs_input, action_dim)
                            else:
                                sampled = agent.sample_actions(obs_input, deterministic=False)
                                residual_step_action = as_numpy_action(sampled, action_dim)

                        # 4) 与当前步 base action 融合成 final action 并执行。
                        #    这里会做 residual clip/limit/scale，然后加到 base_action 上。
                        delta_action, final_action = compose_residual_action(
                            base_action=base_chunk[chunk_step],
                            residual_action=residual_step_action,
                            indices=control_indices,
                            limits=residual_limits,
                            residual_scale=residual_scale_step,
                            xi=xi_step,
                            clip_gripper=bool(cfg.residual.clip_gripper),
                        )

                        # 5) 环境推进一步，得到用于下一步决策的 next_obs_raw。
                        next_obs_raw, reward, env_done, _, info = env.step(final_action)
                        episode_steps += 1
                        global_env_step += 1
                        episode_return += float(reward)
                        episode_success = bool(info["success"])
                        budget_exhausted = bool(
                            max_online_env_steps > 0 and global_env_step >= max_online_env_steps
                        )

                        timeout = bool(episode_steps >= max_episode_steps)
                        done = bool(env_done or timeout or budget_exhausted)

                        step_logger.write(
                            {
                                "global_env_step": int(global_env_step),
                                "global_policy_step": int(global_policy_step),
                                "episode_id": episode_id,
                                "phase": phase_name,
                                "episode_step": episode_steps,
                                "seed": int(env.last_seed if env.last_seed is not None else seed),
                                "is_probing": False,
                                "replan_point": bool(chunk_step == 0),
                                "chunk_step": int(chunk_step),
                                "chunk_horizon": int(chunk_horizon),
                                "infer_e2e_ms": infer_info.get("e2e_ms") if chunk_step == 0 else None,
                                "infer_policy_ms": infer_info.get("policy_ms") if chunk_step == 0 else None,
                                "infer_server_ms": infer_info.get("server_ms") if chunk_step == 0 else None,
                                "a_base": base_chunk[chunk_step].tolist(),
                                "a_res": delta_action.tolist(),
                                "a_final": final_action.tolist(),
                                "residual_scale": float(residual_scale_step),
                                "xi": float(xi_step),
                                "reward": float(reward),
                                "done": bool(done),
                                "success": bool(episode_success),
                            }
                        )

                        if global_env_step % tb_step_period == 0:
                            tb_writer.add_scalar("step/reward", float(reward), global_env_step)
                            tb_writer.add_scalar("step/residual_scale", float(residual_scale_step), global_env_step)
                            tb_writer.add_scalar("step/xi", float(xi_step), global_env_step)
                            tb_writer.add_scalar(
                                "step/residual_action_magnitude",
                                float(np.linalg.norm(delta_action)),
                                global_env_step,
                            )
                            if chunk_step == 0 and infer_info.get("e2e_ms") is not None:
                                tb_writer.add_scalar("step/infer_e2e_ms", float(infer_info["e2e_ms"]), global_env_step)
                            if chunk_step == 0 and infer_info.get("policy_ms") is not None:
                                tb_writer.add_scalar("step/infer_policy_ms", float(infer_info["policy_ms"]), global_env_step)

                        # 6) 构建 next_obs_input（用于 replay 的 next_observations）：
                        #    - done: 用零占位，mask=0
                        #    - 同 chunk 内: 复用当前 base_chunk，mask=1
                        #    - chunk 末尾且未结束: 预取下一段 base_chunk，mask=1
                        if done:
                            next_obs_input = _zero_obs_like(obs_input)
                            mask = 0.0
                        elif chunk_step < (chunk_horizon - 1):
                            next_obs_input = build_residual_step_obs(
                                next_obs_raw,
                                base_chunk[chunk_step + 1],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                normalizer=normalizer,
                            )
                            mask = 1.0
                        else:
                            next_openpi_chunk, next_infer_info = openpi_client.infer_chunk(
                                next_obs_raw,
                                env.current_instruction,
                            )
                            next_base_chunk = select_action_chunk_window(next_openpi_chunk, horizon=chunk_horizon)
                            next_obs_input = build_residual_step_obs(
                                next_obs_raw,
                                next_base_chunk[0],
                                image_keys=image_keys,
                                stack_horizon=stack_horizon,
                                normalizer=normalizer,
                            )
                            cached_base_chunk = next_base_chunk
                            cached_infer_info = next_infer_info
                            mask = 1.0

                        # 7) 以“单步 residual 决策”为单位写入 replay。
                        #    单条 transition 即 (s_t, a_res_t, r_t, s_{t+1}, done_t)。
                        transition_payload = {
                            "observations": _clone_obs_dict(obs_input),
                            "actions": residual_step_action.astype(np.float32),
                            "next_observations": _clone_obs_dict(next_obs_input),
                            "rewards": np.float32(reward),
                            "masks": np.float32(mask),
                            "dones": bool(done),
                        }
                        if async_learner is not None:
                            with async_learner.replay_lock:
                                replay_buffer.insert(transition_payload)
                        else:
                            replay_buffer.insert(transition_payload)

                        # 8) 训练更新：满足起训步数后按 update_every 和 updates_per_step 执行。
                        #    每次更新可按比例混合 online/offline batch。
                        update_info: Dict[str, Any] = {}
                        if async_learner is None:
                            if (
                                phase_train
                                and len(replay_buffer) >= int(cfg.training.training_starts)
                                and global_policy_step % int(cfg.training.update_every) == 0
                            ):
                                for _ in range(int(cfg.training.updates_per_step)):
                                    batch, online_bs, offline_bs = _sample_mixed_batch(
                                        replay_buffer,
                                        offline_buffer if offline_enabled else None,
                                        batch_size=int(cfg.replay.batch_size),
                                        offline_ratio=offline_ratio,
                                        symmetric_replay=symmetric_replay,
                                    )
                                    # DrQ 更新内部会先做图像随机裁剪增强，再执行 SAC 的 critic/actor/temperature 更新。
                                    learner_agent, update_info = learner_agent.update_high_utd(
                                        batch,
                                        utd_ratio=int(cfg.sac.utd_ratio),
                                    )
                                    update_info["online_batch_size"] = int(online_bs)
                                    update_info["offline_batch_size"] = int(offline_bs)
                                agent = learner_agent
                        else:
                            update_info = async_learner.get_last_update_info()

                        if update_info and global_env_step % tb_step_period == 0:
                            for tb_key, info_key in (
                                ("critic/loss", "critic_loss"),
                                ("critic/td_loss", "critic_td_loss"),
                                ("critic/cql_penalty", "critic_cql_penalty"),
                                ("critic/predicted_qs", "predicted_qs"),
                                ("critic/target_qs", "target_qs"),
                                ("actor/loss", "actor_loss"),
                                ("actor/entropy", "entropy"),
                                ("actor/temperature", "temperature"),
                                ("actor/temperature_loss", "temperature_loss"),
                            ):
                                if info_key in update_info:
                                    tb_writer.add_scalar(tb_key, float(update_info[info_key]), global_env_step)

                        global_policy_step += 1

                        # 周期性保存 checkpoint。
                        if (
                            phase_train
                            and int(cfg.training.checkpoint_period) > 0
                            and global_policy_step % int(cfg.training.checkpoint_period) == 0
                        ):
                            if async_learner is not None:
                                async_learner.save_checkpoint(
                                    str(checkpoint_dir),
                                    step=global_policy_step,
                                    keep=int(cfg.training.keep_checkpoints),
                                )
                            else:
                                save_agent_checkpoint(
                                    str(checkpoint_dir),
                                    learner_agent,
                                    step=global_policy_step,
                                    keep=int(cfg.training.keep_checkpoints),
                                )

                        if done:
                            episode_done = True
                            break

                    obs_raw = next_obs_raw
                    if episode_done:
                        break

                # episode 结束：累计成功率并记录 episode 级日志。
                total_success += int(episode_success)
                recent_successes.append(int(episode_success))
                running_success_rate = float(total_success) / float(episode_id + 1)
                recent_success_rate = float(sum(recent_successes)) / float(len(recent_successes))

                episode_logger.write(
                    {
                        "episode_id": episode_id,
                        "phase": phase_name,
                        "seed": int(env.last_seed if env.last_seed is not None else seed),
                        "success": bool(episode_success),
                        "episode_steps": int(episode_steps),
                        "episode_return": float(episode_return),
                        "global_env_step": int(global_env_step),
                        "global_policy_step": int(global_policy_step),
                        "running_success_rate": running_success_rate,
                    }
                )

                tb_writer.add_scalar("episode/success", int(episode_success), global_env_step)
                tb_writer.add_scalar("episode/return", float(episode_return), global_env_step)
                tb_writer.add_scalar("episode/length", int(episode_steps), global_env_step)
                tb_writer.add_scalar("episode/running_success_rate", running_success_rate, global_env_step)
                tb_writer.add_scalar("episode/recent_success_rate_20", recent_success_rate, global_env_step)
                tb_writer.add_scalar("system/online_buffer_size", int(len(replay_buffer)) if replay_buffer else 0, global_env_step)
                if offline_buffer is not None:
                    tb_writer.add_scalar("system/offline_buffer_size", int(len(offline_buffer)), global_env_step)
                tb_writer.add_scalar("system/global_policy_step", int(global_policy_step), global_env_step)
                if async_learner is not None:
                    tb_writer.add_scalar("system/learner_update_steps", int(async_learner.get_update_steps()), global_env_step)

                logger.info(
                    "episode=%s phase=%s success=%s steps=%s return=%.2f success_rate=%.3f",
                    episode_id,
                    phase_name,
                    episode_success,
                    episode_steps,
                    episode_return,
                    running_success_rate,
                )
                episode_id += 1
                phase_episode_count += 1
                if max_online_env_steps > 0 and global_env_step >= max_online_env_steps:
                    stopped_by_env_budget = True
                    break

            if stopped_by_env_budget:
                logger.info(
                    "stop training: reached max_online_env_steps=%s (global_env_step=%s)",
                    max_online_env_steps,
                    global_env_step,
                )
                break

        if async_learner is not None:
            async_learner.stop()

        # 训练结束汇总：写 summary.json。
        summary = {
            "env_backend": env_backend,
            "episodes": int(episode_id),
            "global_env_steps": int(global_env_step),
            "global_policy_steps": int(global_policy_step),
            "total_success": int(total_success),
            "success_rate": float(total_success / max(1, episode_id)),
            "checkpoint_dir": str(checkpoint_dir),
            "chunk_horizon": int(chunk_horizon),
            "residual_action_dim": int(action_dim),
            "expert_check": bool(cfg.training.expert_check),
            "skipped_seeds": int(skipped_seeds),
            "seed_start": int(cfg.task.seed_base),
            "seed_next": int(seed_cursor),
            "seed_attempts": int(seed_cursor - int(cfg.task.seed_base)),
            "offline_enabled": bool(offline_enabled),
            "offline_ratio": float(offline_ratio),
            "offline_symmetric_replay": bool(symmetric_replay),
            "offline_buffer_size": int(len(offline_buffer) if offline_buffer is not None else 0),
            "offline_stats": offline_stats,
            "bootstrap_stats": bootstrap_stats,
            "residual_xi": float(residual_xi),
            "xi_scheduler_enabled": bool(
                cfg.training.get("xi_scheduler", {}).get("enabled", False)
                if cfg.training.get("xi_scheduler", None) is not None
                else False
            ),
            "warmup_base_episodes": int(warmup_base_episodes),
            "warmup_base_steps": int(warmup_base_steps),
            "max_online_env_steps": int(max_online_env_steps),
            "stopped_by_env_budget": bool(stopped_by_env_budget),
            "async_enabled": bool(async_enabled),
            "async_update_frequency": int(async_update_frequency),
            "learner_update_steps": int(async_learner.get_update_steps() if async_learner is not None else 0),
            "probing_alpha": (
                float(cfg.training.get("probing_alpha"))
                if cfg.training.get("probing_alpha", None) is not None
                else None
            ),
            "critic_pretrain": warmstart_info,
        }

        with open(run_dir / str(cfg.logging.summary_file), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("training done: %s", summary)

    finally:
        if async_learner is not None:
            async_learner.stop()
        try:
            env.close(clear_cache=False)
        except Exception:  # noqa: BLE001
            pass
        step_logger.close()
        episode_logger.close()
        tb_writer.close()


if __name__ == "__main__":
    main()
