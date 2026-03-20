#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-/vla/users/niejunnan/codebase/openpi}"
DEFAULT_POLICY_DIR="/vla/users/niejunnan/openpi-assets/checkpoints/pi0_libero"
PORT="30001"
GPU_ID="${GPU_ID:-0}"
USE_CHECKPOINT=true
POLICY_CONFIG="${POLICY_CONFIG:-pi0_libero}"
POLICY_DIR="${POLICY_DIR:-$DEFAULT_POLICY_DIR}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --openpi-root)
            OPENPI_ROOT="$2"
            shift 2
            ;;
        --policy-config)
            POLICY_CONFIG="$2"
            USE_CHECKPOINT=true
            shift 2
            ;;
        --policy-dir)
            POLICY_DIR="$2"
            USE_CHECKPOINT=true
            shift 2
            ;;
        --default-policy)
            USE_CHECKPOINT=false
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "  LIBERO OpenPI Server"
echo "=========================================="
echo "  OpenPI root : $OPENPI_ROOT"
echo "  Port        : $PORT"
echo "  GPU         : $GPU_ID"
echo "  Checkpoint  : $USE_CHECKPOINT"
echo "  Config      : $POLICY_CONFIG"
echo "  Policy dir  : $POLICY_DIR"
echo "=========================================="

CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    if [[ -n "${OPENPI_CONDA_PREFIX:-}" ]]; then
        conda activate "$OPENPI_CONDA_PREFIX"
    elif [[ -n "${OPENPI_CONDA_ENV:-}" ]]; then
        conda activate "$OPENPI_CONDA_ENV"
    elif [[ -d "/vla/users/niejunnan/envs/openpi" ]]; then
        conda activate "/vla/users/niejunnan/envs/openpi"
    elif [[ -d "/vla/miniconda3/envs/openpi" ]]; then
        conda activate openpi
    fi
fi

source /vla/miniconda3/bin/activate base
conda activate openpi-modified

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found in PATH for tools/serve_openpi.sh"
    echo "Hint: set OPENPI_CONDA_ENV or OPENPI_CONDA_PREFIX to an env with uv installed."
    exit 1
fi

# source /vla/miniconda3/bin/activate base
# conda activate openpi-modified

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
cd "$OPENPI_ROOT"

if [[ "$USE_CHECKPOINT" == "true" ]]; then
    if [[ -z "$POLICY_DIR" ]]; then
        echo "ERROR: --policy-dir is required when using checkpoint mode"
        exit 1
    fi
    if [[ ! -d "$POLICY_DIR" ]]; then
        echo "ERROR: checkpoint directory not found: $POLICY_DIR"
        exit 1
    fi
    exec uv run scripts/serve_policy.py \
        --port "$PORT" \
        policy:checkpoint \
        --policy.config="$POLICY_CONFIG" \
        --policy.dir="$POLICY_DIR" \
        "${EXTRA_ARGS[@]}"
else
    exec uv run scripts/serve_policy.py \
        --port "$PORT" \
        --env LIBERO \
        "${EXTRA_ARGS[@]}"
fi
