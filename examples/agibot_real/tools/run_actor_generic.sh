#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONF_DIR="$ROOT_DIR/conf"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

require_arg() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "ERROR: ${flag} requires a value"
        exit 1
    fi
}

CONFIG_ARG=""
GPU_ID="0"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash tools/run_actor_generic.sh <yaml|path/to/config.yaml> [--gpu_id N] [extra overrides...]"
            echo
            echo "Config-driven actor wrapper."
            echo "AgiBot real env is local-only."
            echo "This wrapper does not force training.async.* settings."
            echo "Use it when you want scripts/train/run_actor.py to follow the config exactly."
            echo "Default split-process workflow: tools/run_actor.sh or tools/run_actor_agentlace.sh"
            exit 0
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

if [[ -z "$CONFIG_ARG" ]]; then
    echo "ERROR: config path is required"
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

CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
CONFIG_NAME="${CONFIG_BASENAME%.yaml}"

NORMALIZED_EXTRA_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == training.async.*=* ]] || [[ "$arg" == training.async=* ]]; then
        NORMALIZED_EXTRA_ARGS+=("++$arg")
    else
        NORMALIZED_EXTRA_ARGS+=("$arg")
    fi
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$ROOT_DIR"
echo "=========================================="
echo "  AgiBot Generic Actor Wrapper"
echo "=========================================="
echo "  Mode        : config-driven"
echo "  Env mode    : local AgiBot env"
echo "  Async mode  : from config/overrides"
echo "=========================================="
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/train/run_actor.py" \
    --config-dir "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    "${NORMALIZED_EXTRA_ARGS[@]}"
