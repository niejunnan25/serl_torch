#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONF_DIR="$ROOT_DIR/conf"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

CONFIG_ARG=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash tools/launch_async_train.sh <yaml|path/to/config.yaml> [hydra overrides...]"
            exit 0
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

HAS_HYDRA_RUN_DIR_OVERRIDE=0
for arg in "${EXTRA_ARGS[@]:-}"; do
    if [[ "$arg" == hydra.run.dir=* ]]; then
        HAS_HYDRA_RUN_DIR_OVERRIDE=1
        break
    fi
done
if [[ "$HAS_HYDRA_RUN_DIR_OVERRIDE" -eq 0 ]]; then
    EXTRA_ARGS+=('hydra.run.dir=${launch.output_root}/${hydra:job.config_name}/${now:%Y-%m-%d_%H-%M-%S}')
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/train/launch_async_train.py" \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    "${EXTRA_ARGS[@]}"
