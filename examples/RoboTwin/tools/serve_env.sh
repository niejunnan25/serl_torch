#!/usr/bin/env bash
set -euo pipefail

# 启动 RoboTwin 远程仿真环境服务端。
# 训练时 train.sh 会连接此服务（默认端口 9200）。
#
# 用法：
#   bash tools/serve_env.sh                    # 默认端口 9200
#   bash tools/serve_env.sh --port 9200
#   bash tools/serve_env.sh --port 9100 --skip-render-test
#
# 脚本会自动激活 robotwin2 conda 环境（Docker 容器内）。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT=9200
HOST="127.0.0.1"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "  RoboTwin 远程仿真环境服务端"
echo "=========================================="
echo "  工作目录: $ROOT_DIR"
echo "  地址:    http://${HOST}:${PORT}"
echo "  训练连接: env.remote.port=${PORT}"
echo "=========================================="

# 激活 robotwin2 环境（Docker 容器内）
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate robotwin2
else
    echo "错误: 未找到 conda，请确保在 Docker 容器内运行并手动激活 robotwin2 环境"
    exit 1
fi

python scripts/robotwin_env_server.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"

