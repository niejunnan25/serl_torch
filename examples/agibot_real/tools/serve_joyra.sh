#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOYRA_SERVER_PY="${JOYRA_SERVER_PY:-}"
HOST="${HOST:-0.0.0.0}"
PORT="9000"
GPU_ID="${GPU_ID:-0}"
CKPT_PATH="${JOYRA_CKPT_PATH:-}"
UNNORM_KEY="${JOYRA_UNNORM_KEY:-agibot_genie1}"
NUM_INFERENCE_TIMESTEPS="${JOYRA_NUM_INFERENCE_TIMESTEPS:-10}"
EXTRA_ARGS=()

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-py)
            JOYRA_SERVER_PY="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --unnorm-key)
            UNNORM_KEY="$2"
            shift 2
            ;;
        --num-inference-timesteps)
            NUM_INFERENCE_TIMESTEPS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "  AgiBot JoyRA Server"
echo "=========================================="
echo "  Server py   : $JOYRA_SERVER_PY"
echo "  Host        : $HOST"
echo "  Port        : $PORT"
echo "  GPU         : $GPU_ID"
echo "  Checkpoint  : $CKPT_PATH"
echo "  Unnorm key  : $UNNORM_KEY"
echo "  Infer steps : $NUM_INFERENCE_TIMESTEPS"
echo "=========================================="

codex_activate_conda "${JOYRA_CONDA_PREFIX:-}" "${JOYRA_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${JOYRA_PYTHON_BIN:-${SERL_PYTHON_BIN:-python}}")"

if [[ -z "$JOYRA_SERVER_PY" ]]; then
    echo "ERROR: JOYRA_SERVER_PY is not set."
    echo "Set JOYRA_SERVER_PY=/path/to/server.py or pass --server-py."
    exit 1
fi

JOYRA_SERVER_PY="$(cd "$(dirname "$JOYRA_SERVER_PY")" 2>/dev/null && pwd)/$(basename "$JOYRA_SERVER_PY")"
if [[ ! -f "$JOYRA_SERVER_PY" ]]; then
    echo "ERROR: JoyRA server.py not found: $JOYRA_SERVER_PY"
    exit 1
fi

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: JOYRA_CKPT_PATH is not set."
    echo "Set JOYRA_CKPT_PATH=/path/to/run_dir/checkpoints/steps_xxx.pt or pass --ckpt-path."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec "$PYTHON_BIN" "$JOYRA_SERVER_PY" \
    --host "$HOST" \
    --port "$PORT" \
    --ckpt-path "$CKPT_PATH" \
    --unnorm-key "$UNNORM_KEY" \
    --num-inference-timesteps "$NUM_INFERENCE_TIMESTEPS" \
    "${EXTRA_ARGS[@]}"
