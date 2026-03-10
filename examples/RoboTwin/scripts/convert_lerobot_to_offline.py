#!/usr/bin/env python3
"""
将 LeRobot v2.x parquet 数据集转换为离线训练可用的 pkl 文件。

数据流：
  LeRobot parquet ──▶ 逐 episode 读取 ──▶ 按 chunk_horizon 分组
      ──▶ (可选) 调用 OpenPI 预算 base_chunk
      ──▶ 组装 per-step transition ──▶ pickle 保存

输出 transition 格式与 train_residual_sac.py 中的
``_convert_expert_transition_to_residual`` 兼容：
  - observations / next_observations : RoboTwin raw obs (图像存为 JPEG bytes)
  - expert_action_chunk : (chunk_horizon, 14)
  - base_chunk          : (chunk_horizon, 14)  # 仅在启用 OpenPI 时
  - chunk_step          : int
  - rewards / dones / success

用法示例：
  # 不调用 OpenPI（训练时由训练代码在线推理 base）
  python scripts/convert_lerobot_to_offline.py \
      --dataset_dir /path/to/single_task_clean_place_a2b_left \
      --output_dir  data/offline/place_a2b_left

  # 预算 base_chunk（推荐，训练时更快）
  python scripts/convert_lerobot_to_offline.py \
      --dataset_dir /path/to/single_task_clean_place_a2b_left \
      --output_dir  data/offline/place_a2b_left \
      --openpi_host localhost --openpi_port 9000
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import ALOHA_ACTION_DIM

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("convert_lerobot")


# ---------------------------------------------------------------------------
# LeRobot 解析
# ---------------------------------------------------------------------------

def _read_episodes_meta(dataset_dir: Path) -> List[Dict[str, Any]]:
    """读取 meta/episodes.jsonl，返回每 episode 的元信息列表。"""
    meta_path = dataset_dir / "meta" / "episodes.jsonl"
    episodes: List[Dict[str, Any]] = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def _parquet_path(dataset_dir: Path, episode_index: int) -> Path:
    """LeRobot v2.x 的 parquet 路径约定。"""
    chunk_idx = episode_index // 1000
    return dataset_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG bytes → (H, W, 3) uint8 numpy array。"""
    return np.asarray(Image.open(io.BytesIO(jpeg_bytes)), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 帧 → RoboTwin raw obs
# ---------------------------------------------------------------------------

def _frame_to_obs_raw(row: Dict[str, Any], *, jpeg: bool = True) -> Dict[str, Any]:
    """
    将 LeRobot parquet 的一行转为 RoboTwin raw obs 格式。

    Parameters
    ----------
    row : dict
        pyarrow row 转为 Python dict 后的数据。
    jpeg : bool
        True  → 图像保留 JPEG bytes（存储紧凑，需 _decode_camera_rgb 解码）。
        False → 图像解码为 numpy（OpenPI 推理需要）。
    """
    state = np.asarray(row["observation.state"], dtype=np.float32)

    cam_keys = [
        ("observation.images.cam_high", "head_camera"),
        ("observation.images.cam_left_wrist", "left_camera"),
        ("observation.images.cam_right_wrist", "right_camera"),
    ]

    cameras: Dict[str, Dict[str, Any]] = {}
    for parquet_key, cam_name in cam_keys:
        img_bytes: bytes = row[parquet_key]["bytes"]
        if jpeg:
            cameras[cam_name] = {"rgb_jpeg": img_bytes}
        else:
            cameras[cam_name] = {"rgb": _decode_jpeg(img_bytes)}

    return {
        "joint_action": {"vector": state},
        "observation": cameras,
    }


# ---------------------------------------------------------------------------
# Episode 转换
# ---------------------------------------------------------------------------

def _convert_episode(
    dataset_dir: Path,
    episode_meta: Dict[str, Any],
    *,
    chunk_horizon: int,
    openpi_client: Any | None,
    prompt: str,
) -> List[Dict[str, Any]]:
    """将一个 episode 的 parquet 转为 transition 列表。"""
    ep_idx = int(episode_meta["episode_index"])
    parquet_path = _parquet_path(dataset_dir, ep_idx)
    table = pq.read_table(parquet_path)
    num_frames = len(table)
    columns = table.column_names

    rows = [
        {col: table.column(col)[i].as_py() for col in columns}
        for i in range(num_frames)
    ]

    expert_actions = np.array(
        [np.asarray(r["action"], dtype=np.float32) for r in rows],
        dtype=np.float32,
    )  # (num_frames, 14)

    transitions: List[Dict[str, Any]] = []
    chunk_start = 0

    while chunk_start < num_frames:
        chunk_end = min(chunk_start + chunk_horizon, num_frames)
        chunk_len = chunk_end - chunk_start

        # 专家动作 chunk（pad 到 chunk_horizon）
        expert_chunk = expert_actions[chunk_start:chunk_end]
        if chunk_len < chunk_horizon:
            pad = np.repeat(expert_chunk[-1:], chunk_horizon - chunk_len, axis=0)
            expert_chunk = np.concatenate([expert_chunk, pad], axis=0)

        # 预算 base_chunk（可选）
        base_chunk: np.ndarray | None = None
        if openpi_client is not None:
            from policy import select_action_chunk_window

            obs_decoded = _frame_to_obs_raw(rows[chunk_start], jpeg=False)
            openpi_out, _ = openpi_client.infer_chunk(obs_decoded, prompt)
            base_chunk = select_action_chunk_window(openpi_out, horizon=chunk_horizon)

        for step in range(chunk_len):
            frame_idx = chunk_start + step
            is_last = frame_idx == num_frames - 1

            obs_raw = _frame_to_obs_raw(rows[frame_idx], jpeg=True)

            if is_last:
                next_obs_raw = _frame_to_obs_raw(rows[frame_idx], jpeg=True)
                done = True
                reward = 1.0
            else:
                next_obs_raw = _frame_to_obs_raw(rows[frame_idx + 1], jpeg=True)
                done = False
                reward = 0.0

            transition: Dict[str, Any] = {
                "observations": obs_raw,
                "next_observations": next_obs_raw,
                "expert_action_chunk": expert_chunk.copy(),
                "chunk_step": step,
                "rewards": float(reward),
                "dones": bool(done),
                "success": bool(is_last),
            }
            if base_chunk is not None:
                transition["base_chunk"] = base_chunk.copy()

            transitions.append(transition)

        chunk_start = chunk_end

    return transitions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 LeRobot parquet 数据集转换为残差训练离线 pkl。"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="LeRobot 数据集根目录（含 meta/ 和 data/）",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出目录，每个 episode 保存一个 pkl",
    )
    parser.add_argument("--chunk_horizon", type=int, default=10)
    parser.add_argument(
        "--openpi_host", type=str, default=None,
        help="OpenPI 地址；不设则跳过 base_chunk 预算",
    )
    parser.add_argument("--openpi_port", type=int, default=9000)
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="覆盖 OpenPI prompt（默认使用每 episode 的任务描述）",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _read_episodes_meta(dataset_dir)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    logger.info("Dataset: %s  episodes to convert: %d", dataset_dir, len(episodes))

    openpi_client = None
    if args.openpi_host is not None:
        from policy import OpenPIChunkClient

        openpi_client = OpenPIChunkClient(
            host=args.openpi_host,
            port=args.openpi_port,
            logger=logger,
        )
        logger.info("OpenPI client: %s:%s", args.openpi_host, args.openpi_port)

    all_pkl_paths: List[str] = []
    total_transitions = 0
    t0 = time.time()

    pbar = tqdm(episodes, desc="Converting episodes", unit="ep", dynamic_ncols=True)
    for ep_meta in pbar:
        ep_idx = int(ep_meta["episode_index"])

        if args.prompt is not None:
            prompt = args.prompt
        else:
            tasks = ep_meta.get("tasks", [])
            prompt = tasks[0] if tasks else "none"

        transitions = _convert_episode(
            dataset_dir,
            ep_meta,
            chunk_horizon=args.chunk_horizon,
            openpi_client=openpi_client,
            prompt=prompt,
        )

        pkl_name = f"episode_{ep_idx:06d}.pkl"
        pkl_path = output_dir / pkl_name
        with open(pkl_path, "wb") as f:
            pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)

        all_pkl_paths.append(str(pkl_path))
        total_transitions += len(transitions)
        pbar.set_postfix(
            ep=ep_idx,
            trans=len(transitions),
            total=total_transitions,
            refresh=True,
        )

    elapsed = time.time() - t0

    manifest = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "chunk_horizon": args.chunk_horizon,
        "openpi_precomputed": openpi_client is not None,
        "num_episodes": len(episodes),
        "total_transitions": total_transitions,
        "pkl_files": all_pkl_paths,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Done: %d episodes, %d transitions, %.1fs. Manifest: %s",
        len(episodes),
        total_transitions,
        elapsed,
        manifest_path,
    )


if __name__ == "__main__":
    main()
