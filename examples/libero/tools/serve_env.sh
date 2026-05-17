#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="127.0.0.1"
PORT="30000"
GPU_ID="${LIBERO_ENV_GPU_ID:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<'EOF'
Usage:
  bash examples/libero/tools/serve_env.sh [--host HOST] [--port PORT] [--gpu-id N]

Options:
  --gpu-id N   GPU id for LIBERO / MuJoCo EGL offscreen rendering.
EOF
            exit 0
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
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$GPU_ID" ]]; then
    if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --gpu-id must be a single non-negative integer, got: $GPU_ID"
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
fi

echo "=========================================="
echo "  LIBERO Remote Env Server"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Address     : http://${HOST}:${PORT}"
echo "  CUDA GPUs   : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  EGL device  : ${MUJOCO_EGL_DEVICE_ID:-<unset>}"
echo "=========================================="

CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
DEFAULT_LIBERO_PREFIX="/vla/users/niejunnan/envs/libero"

if [[ -f "$CONDA_SH" ]]; then
    # Default to the known LIBERO env, but allow callers to override.
    # Example:
    #   LIBERO_CONDA_ENV=myenv bash tools/serve_env.sh
    #   LIBERO_CONDA_PREFIX=/abs/prefix bash tools/serve_env.sh
    source "$CONDA_SH"
    if [[ -n "${LIBERO_CONDA_PREFIX:-}" ]]; then
        conda activate "$LIBERO_CONDA_PREFIX"
    elif [[ -n "${LIBERO_CONDA_ENV:-}" ]]; then
        conda activate "$LIBERO_CONDA_ENV"
    elif [[ -d "$DEFAULT_LIBERO_PREFIX" ]]; then
        conda activate "$DEFAULT_LIBERO_PREFIX"
    fi
fi

PYTHON_BIN="${LIBERO_PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    fi
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[0] >= 3 else 1)
PY
then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    fi
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[0] >= 3 else 1)
PY
then
    echo "ERROR: tools/serve_env.sh requires Python 3; current PYTHON_BIN=${PYTHON_BIN}"
    echo "Hint: set LIBERO_CONDA_ENV or LIBERO_CONDA_PREFIX to a valid LIBERO env."
    exit 1
fi

exec "$PYTHON_BIN" scripts/serve_env.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
