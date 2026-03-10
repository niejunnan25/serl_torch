#!/usr/bin/env bash
set -euo pipefail

# 一键将 LeRobot 数据集转换为离线训练用的 pkl 文件。
#
# 用法：
#   # 基本用法（不调用 OpenPI，训练时在线推理 base）
#   bash tools/convert_offline.sh --task place_a2b_left
#
#   # 预算 base_chunk（推荐，训练时更快）
#   bash tools/convert_offline.sh --task place_a2b_left --openpi
#
#   # 自定义所有路径
#   bash tools/convert_offline.sh \
#       --dataset_dir /path/to/single_task_clean_place_a2b_left \
#       --output_dir data/offline/place_a2b_left \
#       --openpi_host localhost --openpi_port 9000
#
# 快捷参数:
#   --task TASK_NAME    自动推断 dataset_dir 和 output_dir
#   --openpi            启用 OpenPI 预算 base_chunk (host=localhost, port=9000)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_ROOT="/vla/users/niejunnan/assets/robotwin-assets/datasets/single_task/test"
DATASET_PREFIX="single_task_clean_"
OUTPUT_BASE="data/offline"

TASK_NAME=""
DATASET_DIR=""
OUTPUT_DIR=""
USE_OPENPI=false
OPENPI_HOST="localhost"
OPENPI_PORT="9000"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK_NAME="$2"; shift 2 ;;
        --dataset_dir)
            DATASET_DIR="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --openpi)
            USE_OPENPI=true; shift ;;
        --openpi_host)
            OPENPI_HOST="$2"; USE_OPENPI=true; shift 2 ;;
        --openpi_port)
            OPENPI_PORT="$2"; USE_OPENPI=true; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -n "$TASK_NAME" ]; then
    [ -z "$DATASET_DIR" ] && DATASET_DIR="${DATASET_ROOT}/${DATASET_PREFIX}${TASK_NAME}"
    [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="${OUTPUT_BASE}/${TASK_NAME}"
fi

if [ -z "$DATASET_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: 必须指定 --task TASK_NAME 或同时指定 --dataset_dir 和 --output_dir"
    exit 1
fi

CMD=(python scripts/convert_lerobot_to_offline.py
    --dataset_dir "$DATASET_DIR"
    --output_dir "$OUTPUT_DIR"
)

if [ "$USE_OPENPI" = true ]; then
    CMD+=(--openpi_host "$OPENPI_HOST" --openpi_port "$OPENPI_PORT")
fi

CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "=========================================="
echo "  Convert LeRobot → Offline PKL"
echo "=========================================="
echo "  Working dir  : $ROOT_DIR"
echo "  Dataset dir  : $DATASET_DIR"
echo "  Output dir   : $OUTPUT_DIR"
echo "  OpenPI       : $USE_OPENPI"
echo "=========================================="

"${CMD[@]}"
