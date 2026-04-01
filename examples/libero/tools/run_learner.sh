#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONF_DIR="$ROOT_DIR/conf"

CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    if [[ -n "${SERL_CONDA_PREFIX:-}" ]]; then
        conda activate "$SERL_CONDA_PREFIX"
    elif [[ -n "${SERL_CONDA_ENV:-}" ]]; then
        conda activate "$SERL_CONDA_ENV"
    elif [[ -d "/vla/miniconda3/envs/serl_torch" ]]; then
        conda activate serl_torch
    fi
fi

PYTHON_BIN="${SERL_PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_learner.sh <yaml_file_name.yaml|/abs/path/to/config.yaml> --bootstrap /abs/path/to/agentlace_bootstrap.pkl [--gpu_id N] [extra hydra overrides...]

Examples:
  bash tools/run_learner.sh train_residual_sac.yaml --bootstrap /abs/path/to/agentlace_bootstrap.pkl --gpu_id 0
EOF
}

require_arg() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "ERROR: ${flag} requires a value"
        exit 1
    fi
}

CONFIG_ARG=""
BOOTSTRAP_PATH=""
GPU_ID="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --bootstrap)
            require_arg "$1" "${2:-}"
            BOOTSTRAP_PATH="$2"
            shift 2
            ;;
        --bootstrap=*)
            BOOTSTRAP_PATH="${1#*=}"
            shift
            ;;
        --gpu_id|--gpu-id)
            require_arg "$1" "${2:-}"
            GPU_ID="$2"
            shift 2
            ;;
        *)
            if [[ -z "$CONFIG_ARG" ]]; then
                CONFIG_ARG="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

NORMALIZED_EXTRA_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == training.async.*=* ]] || [[ "$arg" == training.async=* ]]; then
        NORMALIZED_EXTRA_ARGS+=("++$arg")
    else
        NORMALIZED_EXTRA_ARGS+=("$arg")
    fi
done

if [[ -z "$CONFIG_ARG" || -z "$BOOTSTRAP_PATH" ]]; then
    usage
    exit 1
fi

if [[ "$CONFIG_ARG" == */* ]]; then
    CONFIG_PATH="$("$PYTHON_BIN" - "$CONFIG_ARG" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
else
    CONFIG_PATH="$DEFAULT_CONF_DIR/$CONFIG_ARG"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: config file not found: $CONFIG_PATH"
    exit 1
fi

BOOTSTRAP_ABS="$("$PYTHON_BIN" - "$BOOTSTRAP_PATH" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
CONFIG_NAME="${CONFIG_BASENAME%.yaml}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$ROOT_DIR"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_learner.py" \
    --config-dir "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    "++training.async.enabled=true" \
    "++training.async.backend=agentlace" \
    "++training.async.agentlace.bootstrap_file=$BOOTSTRAP_ABS" \
    "${NORMALIZED_EXTRA_ARGS[@]}"
