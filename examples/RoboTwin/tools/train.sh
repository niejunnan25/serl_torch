#!/usr/bin/env bash
set -euo pipefail

# 一键启动残差策略训练。
#
# 用法：
#   bash tools/train.sh                          # 使用默认配置
#   bash tools/train.sh task.name=place_a2b_left  # 覆盖任务名
#   bash tools/train.sh seed=42 training.max_online_env_steps=100000  # 覆盖多个参数
#
# 所有额外参数会透传给 Hydra，可覆盖 conf/train_residual_sac.yaml 中的任意配置。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "  RoboTwin Residual SAC Training"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Config      : conf/train_residual_sac.yaml"
echo "  Extra args  : $*"
echo "=========================================="

# 激活 serl_torch 环境（Docker 容器内）
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate serl_torch
else
    echo "错误: 未找到 conda，请确保在 Docker 容器内运行并手动激活 serl_torch 环境"
    exit 1
fi

python scripts/train_residual_sac.py \
    env.backend=remote \
    env.remote.host=127.0.0.1 \
    env.remote.port=9100 \
    env.remote.robo_root=/vla/users/niejunnan/codebase/RoboTwin \
    offline.bootstrap_base.enabled=false \
    "$@"
