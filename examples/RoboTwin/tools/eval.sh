#!/usr/bin/env bash
set -euo pipefail

# 一键启动残差策略评估。
#
# 用法：
#   bash tools/eval.sh eval.checkpoint_path=/path/to/checkpoint.pt
#   bash tools/eval.sh eval.checkpoint_path=/path/to/checkpoint.pt eval.episodes=50
#   bash tools/eval.sh eval.checkpoint_path=null  # 仅评估 base policy（残差全零）
#
# 所有额外参数会透传给 Hydra，可覆盖 conf/eval_residual_fast.yaml 中的任意配置。
# 默认使用 remote 环境（需先运行 serve_env.sh），连接 localhost:9000 的 OpenPI 服务。
#
# 评估前请确保：
#   1. bash tools/serve_env.sh --port 9100   # 环境服务端
#   2. bash tools/serve_openpi.sh <task> --port 9000  # OpenPI 服务端

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "  RoboTwin Residual SAC Evaluation"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Config      : conf/eval_residual_fast.yaml"
echo "  Extra args  : $*"
echo "=========================================="

python scripts/eval_residual_fast.py \
    env.backend=remote \
    env.remote.host=127.0.0.1 \
    env.remote.port=9200 \
    env.remote.robo_root=/vla/users/niejunnan/codebase/RoboTwin \
    openpi.host=localhost \
    openpi.port=9000 \
    robo_root=/vla/users/niejunnan/codebase/RoboTwin \
    "$@"
