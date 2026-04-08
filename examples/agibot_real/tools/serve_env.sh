#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="127.0.0.1"
PORT="32000"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "  AgiBot Remote Env Server"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Address     : http://${HOST}:${PORT}"
echo "=========================================="

CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    if [[ -n "${AGIBOT_CONDA_PREFIX:-}" ]]; then
        conda activate "$AGIBOT_CONDA_PREFIX"
    elif [[ -n "${AGIBOT_CONDA_ENV:-}" ]]; then
        conda activate "$AGIBOT_CONDA_ENV"
    elif [[ -n "${SERL_CONDA_PREFIX:-}" ]]; then
        conda activate "$SERL_CONDA_PREFIX"
    elif [[ -n "${SERL_CONDA_ENV:-}" ]]; then
        conda activate "$SERL_CONDA_ENV"
    elif [[ -d "/vla/miniconda3/envs/serl_torch" ]]; then
        conda activate serl_torch
    fi
fi

PYTHON_BIN="${AGIBOT_PYTHON_BIN:-${SERL_PYTHON_BIN:-python}}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    fi
fi

exec "$PYTHON_BIN" scripts/services/serve_env.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"

