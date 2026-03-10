#!/usr/bin/env bash
set -euo pipefail

# 启动 OpenPI 服务端（单任务，用于训练/评估时的基策略）。前台运行。
#
# 用法：
#   bash tools/serve_openpi.sh place_a2b_left              # 默认端口 9000
#   bash tools/serve_openpi.sh place_a2b_left --port 9000
#   bash tools/serve_openpi.sh adjust_bottle --port 9001 --gpu_id 1
#
# 参数：
#   TASK_NAME    任务名称（必填）
#   --port N     端口号（可选，默认 9000）
#   --gpu_id N   GPU ID（可选，默认 0）

OPENPI_ROOT="/vla/users/niejunnan/test/openpi"
ASSETS_ROOT="/vla/users/niejunnan/assets/robotwin-assets/finetune-checkpoints"

get_policy_config() {
    echo "pi05_aloha_full_${1}"
}

get_policy_dir() {
    local config_name
    config_name=$(get_policy_config "$1")
    echo "${ASSETS_ROOT}/${config_name}/${config_name}/30000"
}

# 解析参数
TASK_NAME=""
PORT=9000
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        -*)
            echo "错误: 未知参数 $1"
            exit 1
            ;;
        *)
            if [ -z "$TASK_NAME" ]; then
                TASK_NAME="$1"
            else
                echo "错误: 只能指定一个任务名，已指定 '$TASK_NAME'，又遇到 '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$TASK_NAME" ]; then
    echo "用法: bash tools/serve_openpi.sh <TASK_NAME> [--port N] [--gpu_id N]"
    echo ""
    echo "示例:"
    echo "  bash tools/serve_openpi.sh place_a2b_left"
    echo "  bash tools/serve_openpi.sh place_a2b_left --port 9000"
    echo "  bash tools/serve_openpi.sh adjust_bottle --port 9001 --gpu_id 1"
    exit 1
fi

POLICY_CONFIG=$(get_policy_config "$TASK_NAME")
POLICY_DIR=$(get_policy_dir "$TASK_NAME")

if [ ! -d "$POLICY_DIR" ]; then
    echo "错误: 策略目录不存在: $POLICY_DIR"
    exit 1
fi

echo "=========================================="
echo "  OpenPI 服务端"
echo "=========================================="
echo "  任务: $TASK_NAME"
echo "  端口: $PORT"
echo "  GPU:  $GPU_ID"
echo "  配置: $POLICY_CONFIG"
echo "  目录: $POLICY_DIR"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"

# 激活 openpi 环境（Docker 容器内，uv 安装在此环境中）
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate openpi
else
    echo "错误: 未找到 conda，请确保在 Docker 容器内运行并手动激活 openpi 环境"
    exit 1
fi

cd "$OPENPI_ROOT"

uv run scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config="$POLICY_CONFIG" \
    --policy.dir="$POLICY_DIR"
