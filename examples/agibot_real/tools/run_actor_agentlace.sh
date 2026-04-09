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
BOOTSTRAP_PATH=""
GPU_ID="0"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash tools/run_actor_agentlace.sh <yaml|path/to/config.yaml> --bootstrap path/to/bootstrap.pkl [--gpu_id N] [extra overrides...]"
            echo
            echo "Explicit split actor wrapper for external agentlace async training."
            echo "This wrapper forces training.async.enabled=true, training.async.backend=agentlace,"
            echo "and training.async.agentlace.spawn_local_worker=false."
            echo "Default alias: tools/run_actor.sh"
            echo "Config-driven alternative: tools/run_actor_generic.sh"
            echo "The env backend still comes from the config/overrides."
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

if [[ -z "$CONFIG_ARG" || -z "$BOOTSTRAP_PATH" ]]; then
    echo "ERROR: config path and --bootstrap are required"
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
echo "  AgiBot Split Actor Wrapper"
echo "=========================================="
echo "  Mode        : explicit agentlace actor"
echo "  Entry alias : tools/run_actor.sh"
echo "  Env backend : from config/overrides"
echo "  Default     : train config uses env.backend=local"
echo "=========================================="
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/train/run_actor.py" \
    --config-dir "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    "${NORMALIZED_EXTRA_ARGS[@]}" \
    "++training.async.enabled=true" \
    "++training.async.backend=agentlace" \
    "++training.async.agentlace.spawn_local_worker=false" \
    "++training.async.agentlace.bootstrap_file=$BOOTSTRAP_ABS"
