#!/usr/bin/env python3
"""
从 LeRobot 格式的专家数据集中计算 state / action 的统计量（mean, std, min, max），
并保存为 JSON 文件，供残差策略训练/评估时做归一化。

用法：
  # 计算单个任务
  python compute_dataset_stats.py --task adjust_bottle

  # 计算所有任务（自动扫描 dataset_root 下的 single_task_clean_* 目录）
  python compute_dataset_stats.py --all

  # 指定自定义路径
  python compute_dataset_stats.py --task adjust_bottle \
      --dataset_root /path/to/lerobot_datasets \
      --output_dir /path/to/stats

输出 JSON 格式：
{
    "task_name": "adjust_bottle",
    "dataset_path": "/path/to/dataset",
    "total_episodes": 50,
    "total_frames": 7072,
    "state_mean": [14 floats],
    "state_std":  [14 floats],
    "state_min":  [14 floats],
    "state_max":  [14 floats],
    "action_mean": [14 floats],
    "action_std":  [14 floats],
    "action_min":  [14 floats],
    "action_max":  [14 floats]
}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# 需要 pandas 和 pyarrow 来读取 parquet
# 激活环境: source /mnt/workspace/envs/conda3/bin/activate robot_njn
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not found. Please activate the correct environment:")
    print("  source /mnt/workspace/envs/conda3/bin/activate robot_njn")
    sys.exit(1)


DEFAULT_DATASET_ROOT = Path("/mnt/workspace/users/niejunnan/lerobot_datasets")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "stats"
DATASET_PREFIX = "single_task_clean_"


def compute_stats_for_task(
    task_name: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> dict:
    """
    读取指定任务的所有 parquet 文件，计算 state / action 的统计量。
    """
    dataset_dir = dataset_root / f"{DATASET_PREFIX}{task_name}"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    # 寻找所有 parquet 文件（可能有多个 chunk）
    parquet_pattern = str(dataset_dir / "data" / "chunk-*" / "*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir / 'data'}")

    all_states = []
    all_actions = []

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "observation.state" not in df.columns or "action" not in df.columns:
            print(f"  WARNING: skipping {pf} — missing observation.state or action columns")
            continue
        states = np.stack(df["observation.state"].values)
        actions = np.stack(df["action"].values)
        all_states.append(states)
        all_actions.append(actions)

    if not all_states:
        raise ValueError(f"No valid data frames found for task {task_name}")

    all_states = np.concatenate(all_states, axis=0).astype(np.float64)
    all_actions = np.concatenate(all_actions, axis=0).astype(np.float64)

    total_frames = all_states.shape[0]
    total_episodes = len(parquet_files)

    # 计算统计量
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)
    state_min = all_states.min(axis=0)
    state_max = all_states.max(axis=0)

    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    action_min = all_actions.min(axis=0)
    action_max = all_actions.max(axis=0)

    # 防止 std 为零（常量维度），用一个小值替代
    eps = 1e-6
    state_std = np.where(state_std < eps, eps, state_std)
    action_std = np.where(action_std < eps, eps, action_std)

    stats = {
        "task_name": task_name,
        "dataset_path": str(dataset_dir),
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "state_dim": int(all_states.shape[1]),
        "action_dim": int(all_actions.shape[1]),
        "state_mean": state_mean.astype(np.float32).tolist(),
        "state_std": state_std.astype(np.float32).tolist(),
        "state_min": state_min.astype(np.float32).tolist(),
        "state_max": state_max.astype(np.float32).tolist(),
        "action_mean": action_mean.astype(np.float32).tolist(),
        "action_std": action_std.astype(np.float32).tolist(),
        "action_min": action_min.astype(np.float32).tolist(),
        "action_max": action_max.astype(np.float32).tolist(),
    }
    return stats


def discover_all_tasks(dataset_root: Path = DEFAULT_DATASET_ROOT) -> list[str]:
    """扫描 dataset_root 下所有 single_task_clean_* 目录，提取任务名。"""
    tasks = []
    for entry in sorted(dataset_root.iterdir()):
        if entry.is_dir() and entry.name.startswith(DATASET_PREFIX):
            task_name = entry.name[len(DATASET_PREFIX):]
            if task_name:
                tasks.append(task_name)
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Compute normalization stats from expert datasets")
    parser.add_argument("--task", type=str, default=None, help="Task name (e.g. adjust_bottle)")
    parser.add_argument("--all", action="store_true", help="Compute stats for all discovered tasks")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Root directory for datasets (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for stats JSON files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        tasks = discover_all_tasks(dataset_root)
        print(f"Discovered {len(tasks)} tasks under {dataset_root}")
    elif args.task:
        tasks = [args.task]
    else:
        parser.error("Please specify --task TASK_NAME or --all")
        return

    success_count = 0
    fail_count = 0

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Computing stats for: {task}")
        try:
            stats = compute_stats_for_task(task, dataset_root)
            output_path = output_dir / f"{task}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved: {output_path}")
            print(f"    episodes={stats['total_episodes']}, frames={stats['total_frames']}")
            print(f"    state_mean={[f'{v:.4f}' for v in stats['state_mean']]}")
            print(f"    state_std ={[f'{v:.4f}' for v in stats['state_std']]}")
            print(f"    action_mean={[f'{v:.4f}' for v in stats['action_mean']]}")
            print(f"    action_std ={[f'{v:.4f}' for v in stats['action_std']]}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Done! Success: {success_count}, Failed: {fail_count}")
    print(f"Stats saved to: {output_dir}")


if __name__ == "__main__":
    main()
