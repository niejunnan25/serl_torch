#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOYRA_ROOT="${JOYRA_ROOT:-}"
JOYRA_SERVER_PY="${JOYRA_SERVER_PY:-}"
HOST="${HOST:-0.0.0.0}"
PORT="9001"
GPU_ID="${GPU_ID:-0}"
CKPT_PATH="${JOYRA_CKPT_PATH:-}"
EXTRA_ARGS=()

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash tools/serve_joyra.sh [--joyra-root DIR] [--ckpt-path FILE] [--host HOST] [--port PORT] [--gpu-id N] [extra JoyRA args...]"
            echo
            echo "Recommended inputs:"
            echo "  JOYRA_ROOT=/path/to/JoyRA"
            echo "  JOYRA_CKPT_PATH=/path/to/checkpoints/steps_xxx.pt"
            echo "  JOYRA_CONDA_ENV=joyra"
            echo
            echo "Default server entrypoint:"
            echo "  \$JOYRA_ROOT/deployment/real_infer/server.py"
            echo
            echo "The wrapper will:"
            echo "  1. activate the JoyRA env"
            echo "  2. cd into JOYRA_ROOT"
            echo "  3. export PYTHONPATH=\$JOYRA_ROOT:\$PYTHONPATH"
            echo "  4. run deployment/real_infer/server.py"
            exit 0
            ;;
        --joyra-root)
            JOYRA_ROOT="$2"
            shift 2
            ;;
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
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --num-inference-timesteps)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

codex_activate_conda "${JOYRA_CONDA_PREFIX:-}" "${JOYRA_CONDA_ENV:-}" "joyra"

PYTHON_BIN="$(codex_python_bin "${JOYRA_PYTHON_BIN:-${SERL_PYTHON_BIN:-python}}")"

if [[ -z "$JOYRA_ROOT" ]]; then
    if [[ -n "$JOYRA_SERVER_PY" ]]; then
        JOYRA_ROOT="$(cd "$(dirname "$JOYRA_SERVER_PY")/../.." 2>/dev/null && pwd)"
    else
        echo "ERROR: JOYRA_ROOT is not set."
        echo "Set JOYRA_ROOT=/path/to/JoyRA or pass --joyra-root."
        exit 1
    fi
fi
if [[ ! -d "$JOYRA_ROOT" ]]; then
    echo "ERROR: JoyRA root directory not found: $JOYRA_ROOT"
    echo "Set JOYRA_ROOT=/path/to/JoyRA or pass --joyra-root."
    exit 1
fi

if [[ -z "$JOYRA_SERVER_PY" ]]; then
    JOYRA_SERVER_PY="$JOYRA_ROOT/deployment/real_infer/server.py"
fi
JOYRA_SERVER_PY="$(cd "$(dirname "$JOYRA_SERVER_PY")" 2>/dev/null && pwd)/$(basename "$JOYRA_SERVER_PY")"
if [[ ! -f "$JOYRA_SERVER_PY" ]]; then
    echo "ERROR: JoyRA server.py not found: $JOYRA_SERVER_PY"
    echo "Expected default entrypoint: $JOYRA_ROOT/deployment/real_infer/server.py"
    echo "If your layout is different, pass --server-py /path/to/server.py."
    exit 1
fi

if [[ -z "$CKPT_PATH" ]]; then
    echo "ERROR: JOYRA_CKPT_PATH is not set."
    echo "Set JOYRA_CKPT_PATH=/path/to/run_dir/checkpoints/steps_xxx.pt or pass --ckpt-path."
    exit 1
fi

echo "=========================================="
echo "  AgiBot JoyRA Server"
echo "=========================================="
echo "  JoyRA root  : $JOYRA_ROOT"
echo "  Server py   : $JOYRA_SERVER_PY"
echo "  Host        : $HOST"
echo "  Port        : $PORT"
echo "  GPU         : $GPU_ID"
echo "  Checkpoint  : $CKPT_PATH"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "  Extra args  : ${EXTRA_ARGS[*]}"
fi
echo "=========================================="

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$JOYRA_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$JOYRA_ROOT"

exec "$PYTHON_BIN" "$JOYRA_SERVER_PY" \
    --host "$HOST" \
    --port "$PORT" \
    --ckpt-path "$CKPT_PATH" \
    "${EXTRA_ARGS[@]}"
