#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-}"
DEFAULT_POLICY_DIR="${DEFAULT_POLICY_DIR:-}"
PORT="30001"
GPU_ID="${GPU_ID:-0}"
USE_CHECKPOINT=true
POLICY_CONFIG="${POLICY_CONFIG:-pi05_agibot}"
POLICY_DIR="${POLICY_DIR:-$DEFAULT_POLICY_DIR}"
EXTRA_ARGS=()

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

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
echo "  AgiBot OpenPI Server"
echo "=========================================="
echo "  OpenPI root : $OPENPI_ROOT"
echo "  Port        : $PORT"
echo "  GPU         : $GPU_ID"
echo "  Checkpoint  : $USE_CHECKPOINT"
echo "  Config      : $POLICY_CONFIG"
echo "  Policy dir  : $POLICY_DIR"
echo "=========================================="

codex_activate_conda "${OPENPI_CONDA_PREFIX:-}" "${OPENPI_CONDA_ENV:-}" "openpi-modified" "openpi"

if [[ -z "$OPENPI_ROOT" ]]; then
    echo "ERROR: OPENPI_ROOT is not set."
    echo "Set OPENPI_ROOT=relative/path/to/openpi or pass --openpi-root."
    exit 1
fi

OPENPI_ROOT="$(cd "$OPENPI_ROOT" 2>/dev/null && pwd || true)"
if [[ -z "$OPENPI_ROOT" ]]; then
    echo "ERROR: failed to resolve OpenPI root."
    exit 1
fi

if [[ ! -d "$OPENPI_ROOT" ]]; then
    echo "ERROR: OpenPI root not found: $OPENPI_ROOT"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found in PATH for tools/serve_openpi.sh"
    echo "Hint: set OPENPI_CONDA_ENV or OPENPI_CONDA_PREFIX to an env with uv installed."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
cd "$OPENPI_ROOT"

if [[ "$USE_CHECKPOINT" == "true" ]]; then
    if [[ -z "$POLICY_DIR" ]]; then
        echo "ERROR: set POLICY_DIR or pass --policy-dir when using checkpoint mode"
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
        "${EXTRA_ARGS[@]}"
fi
