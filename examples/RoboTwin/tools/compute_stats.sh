#!/usr/bin/env bash
set -euo pipefail

# 一键计算数据集归一化统计量（mean/std/min/max）。
#
# 用法：
#   bash tools/compute_stats.sh                                  # 计算所有任务
#   bash tools/compute_stats.sh --task place_a2b_left            # 计算单个任务
#   bash tools/compute_stats.sh --dataset_root /path/to/datasets # 自定义数据集根目录
#   bash tools/compute_stats.sh --output_dir /path/to/stats      # 自定义输出目录
#
# 默认数据集根目录: /mnt/workspace/users/niejunnan/lerobot_datasets
# 默认输出目录:     data/stats/
# 所有额外参数会透传给 compute_dataset_stats.py。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 如果没有传任何参数，默认计算所有任务
if [ $# -eq 0 ]; then
    ARGS="--all"
else
    ARGS="$*"
fi

echo "=========================================="
echo "  Compute Dataset Stats"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Args        : $ARGS"
echo "=========================================="

python scripts/compute_dataset_stats.py $ARGS
