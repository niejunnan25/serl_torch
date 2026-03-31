#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"
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
    echo "ERROR: run_train.sh requires Python 3; current PYTHON_BIN=${PYTHON_BIN}"
    echo "Hint: set SERL_CONDA_ENV=serl_torch (or SERL_CONDA_PREFIX=/abs/env) before launching."
    exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import hydra  # noqa: F401
PY
then
    echo "ERROR: hydra is not available in PYTHON_BIN=${PYTHON_BIN}"
    echo "Hint: ensure serl training env is activated (SERL_CONDA_ENV=serl_torch)."
    exit 1
fi

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_train.sh <yaml_file_name.yaml|/abs/path/to/config.yaml> [--gpu_id N] [--output-dir DIR] [extra hydra overrides...]

Examples:
  bash tools/run_train.sh train_residual_sac.yaml --gpu_id 0
  bash tools/run_train.sh train_pld_residual_sac.yaml --gpu_id 1
  bash tools/run_train.sh conf/foo.yaml --gpu_id 0 --output-dir outputs/exp0
EOF
}

format_cmd() {
    local formatted=""
    local token=""
    local escaped=""
    for token in "$@"; do
        printf -v escaped "%q" "$token"
        if [[ -n "$formatted" ]]; then
            formatted+=" "
        fi
        formatted+="$escaped"
    done
    printf "%s" "$formatted"
}

require_arg() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "ERROR: ${flag} requires a value"
        exit 1
    fi
}

port_open() {
    local host="$1"
    local port="$2"
    "$PYTHON_BIN" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.3)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
raise SystemExit(0)
PY
}

wait_for_port() {
    local name="$1"
    local host="$2"
    local port="$3"
    local timeout_sec="$4"
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if port_open "$host" "$port"; then
            echo "${name} is ready at ${host}:${port}"
            return 0
        fi
        if (( "$(date +%s)" - start_ts >= timeout_sec )); then
            echo "ERROR: timed out waiting for ${name} at ${host}:${port}"
            return 1
        fi
        sleep 1
    done
}

CONFIG_ARG=""
GPU_ID="0"
OUTPUT_DIR_ARG=""
EXTRA_ARGS=()
ORIGINAL_ARGS=("$@")
LAUNCH_CMD_OVERRIDE="${RUN_TRAIN_LAUNCH_CMD:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --gpu_id|--gpu-id)
            require_arg "$1" "${2:-}"
            GPU_ID="$2"
            shift 2
            ;;
        --output-dir|--output_dir)
            require_arg "$1" "${2:-}"
            OUTPUT_DIR_ARG="$2"
            shift 2
            ;;
        --output-dir=*|--output_dir=*)
            OUTPUT_DIR_ARG="${1#*=}"
            shift
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

CONFIG_DIR="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
CONFIG_NAME="${CONFIG_BASENAME%.yaml}"

OUTPUT_DIR=""
if [[ -n "$OUTPUT_DIR_ARG" ]]; then
    OUTPUT_DIR="$("$PYTHON_BIN" - "$ROOT_DIR" "$OUTPUT_DIR_ARG" <<'PY'
from pathlib import Path
import sys

base = Path(sys.argv[1]).expanduser().resolve()
raw = Path(sys.argv[2]).expanduser()
if raw.is_absolute():
    print(raw.resolve())
else:
    print((base / raw).resolve())
PY
)"
fi

IFS=$'\t' read -r ENV_HOST ENV_PORT OPENPI_HOST OPENPI_PORT ASYNC_EVAL_ENABLED ASYNC_EVAL_ENV_HOST ASYNC_EVAL_ENV_PORT <<<"$("$PYTHON_BIN" - "$CONFIG_DIR" "$CONFIG_NAME" "${EXTRA_ARGS[@]}" <<'PY'
from hydra import compose, initialize_config_dir
import sys

config_dir = sys.argv[1]
config_name = sys.argv[2]
overrides = sys.argv[3:]
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name=config_name, overrides=overrides)

env_host = str(cfg.env.remote.get("host", "127.0.0.1"))
env_port = int(cfg.env.remote.get("port", 30000))
openpi_host = str(cfg.openpi.get("host", "localhost"))
openpi_port = int(cfg.openpi.get("port", 30001))
async_eval_cfg = cfg.training.get("async_eval", {})
async_eval_enabled = bool(async_eval_cfg.get("enabled", False))
async_eval_env_host = str(async_eval_cfg.get("env_host", env_host))
async_eval_env_port = int(async_eval_cfg.get("env_port", 31014))
print(
    f"{env_host}\t{env_port}\t{openpi_host}\t{openpi_port}\t"
    f"{int(async_eval_enabled)}\t{async_eval_env_host}\t{async_eval_env_port}"
)
PY
)"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR=""
HYDRA_RUN_OVERRIDE=""
if [[ -n "$OUTPUT_DIR" ]]; then
    RUN_PARENT_DIR="$OUTPUT_DIR/$CONFIG_NAME"
    mkdir -p "$RUN_PARENT_DIR"
    RUN_DIR="$RUN_PARENT_DIR/$STAMP"
    if [[ -e "$RUN_DIR" ]]; then
        suffix=2
        while [[ -e "${RUN_DIR}_$suffix" ]]; do
            suffix=$((suffix + 1))
        done
        RUN_DIR="${RUN_DIR}_$suffix"
    fi
    mkdir -p "$RUN_DIR"
    SUPPORT_DIR="$RUN_DIR"
    HYDRA_RUN_OVERRIDE="hydra.run.dir=$RUN_DIR"
else
    SUPPORT_ROOT="${RUN_TRAIN_SUPPORT_ROOT:-$ROOT_DIR/outputs/libero/run_train_support}"
    SUPPORT_DIR="$SUPPORT_ROOT/${STAMP}_${CONFIG_NAME}_gpu${GPU_ID}_$$"
    mkdir -p "$SUPPORT_DIR"
fi

ENV_LOG="$SUPPORT_DIR/env_server.log"
OPENPI_LOG="$SUPPORT_DIR/openpi_server.log"
EVAL_ENV_LOG="$SUPPORT_DIR/async_eval_env_server.log"
LAUNCH_LOG="$SUPPORT_DIR/launcher.log"

ENV_PID=""
OPENPI_PID=""
EVAL_ENV_PID=""
EVAL_ENV_REUSED="0"
LAUNCHED_PID=""
CURRENT_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"

launch_in_own_group() {
    local log_file="$1"
    shift

    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" >"$log_file" 2>&1 &
    else
        "$@" >"$log_file" 2>&1 &
    fi
    LAUNCHED_PID=$!
}

stop_service() {
    local pid="$1"
    local pgid=""

    if [[ -z "$pid" ]] || ! kill -0 "$pid" >/dev/null 2>&1; then
        return 0
    fi

    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [[ -n "$pgid" ]] && [[ "$pgid" != "$CURRENT_PGID" ]]; then
        kill -- "-$pgid" >/dev/null 2>&1 || true
    else
        kill "$pid" >/dev/null 2>&1 || true
    fi
    wait "$pid" >/dev/null 2>&1 || true
}

cleanup() {
    local exit_code=$?
    stop_service "$EVAL_ENV_PID"
    stop_service "$OPENPI_PID"
    stop_service "$ENV_PID"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

{
    echo "=========================================="
    echo "  LIBERO run_train.sh"
    echo "=========================================="
    echo "  Root        : $ROOT_DIR"
    if [[ -n "$LAUNCH_CMD_OVERRIDE" ]]; then
        echo "  Launch cmd  : $LAUNCH_CMD_OVERRIDE"
    else
        echo "  Launch cmd  : $(format_cmd bash "$TOOLS_DIR/run_train.sh" "${ORIGINAL_ARGS[@]}")"
    fi
    echo "  Config file : $CONFIG_PATH"
    echo "  Config name : $CONFIG_NAME"
    echo "  GPU         : $GPU_ID"
    echo "  Env server  : ${ENV_HOST}:${ENV_PORT}"
    echo "  OpenPI      : ${OPENPI_HOST}:${OPENPI_PORT}"
    if [[ "$ASYNC_EVAL_ENABLED" == "1" ]]; then
        echo "  Async eval  : enabled"
        echo "  Eval env    : ${ASYNC_EVAL_ENV_HOST}:${ASYNC_EVAL_ENV_PORT}"
    else
        echo "  Async eval  : disabled"
    fi
    if [[ -n "$OUTPUT_DIR" ]]; then
        echo "  Output root : $OUTPUT_DIR"
        echo "  Run dir     : $RUN_DIR"
    fi
    echo "  Support dir : $SUPPORT_DIR"
    echo "  Extra args  : ${EXTRA_ARGS[*]:-<none>}"
    echo "=========================================="
} | tee "$LAUNCH_LOG"

if port_open "$ENV_HOST" "$ENV_PORT"; then
    echo "ERROR: env port already in use: ${ENV_HOST}:${ENV_PORT}" | tee -a "$LAUNCH_LOG"
    exit 1
fi
if port_open "$OPENPI_HOST" "$OPENPI_PORT"; then
    echo "ERROR: OpenPI port already in use: ${OPENPI_HOST}:${OPENPI_PORT}" | tee -a "$LAUNCH_LOG"
    exit 1
fi

echo "Starting env server..." | tee -a "$LAUNCH_LOG"
launch_in_own_group "$ENV_LOG" \
    bash "$TOOLS_DIR/serve_env.sh" --host "$ENV_HOST" --port "$ENV_PORT"
ENV_PID="$LAUNCHED_PID"
wait_for_port "env server" "$ENV_HOST" "$ENV_PORT" 60 | tee -a "$LAUNCH_LOG"

if [[ "$ASYNC_EVAL_ENABLED" == "1" ]]; then
    if [[ "$ASYNC_EVAL_ENV_HOST" == "$ENV_HOST" ]] && [[ "$ASYNC_EVAL_ENV_PORT" == "$ENV_PORT" ]]; then
        echo "WARNING: async eval env matches training env (${ENV_HOST}:${ENV_PORT}); no isolated eval env will be started" | tee -a "$LAUNCH_LOG"
    else
        if port_open "$ASYNC_EVAL_ENV_HOST" "$ASYNC_EVAL_ENV_PORT"; then
            EVAL_ENV_REUSED="1"
            echo "Async eval env server already running, reusing ${ASYNC_EVAL_ENV_HOST}:${ASYNC_EVAL_ENV_PORT}" | tee -a "$LAUNCH_LOG"
        else
            echo "Starting async eval env server..." | tee -a "$LAUNCH_LOG"
            launch_in_own_group "$EVAL_ENV_LOG" \
                bash "$TOOLS_DIR/serve_env.sh" --host "$ASYNC_EVAL_ENV_HOST" --port "$ASYNC_EVAL_ENV_PORT"
            EVAL_ENV_PID="$LAUNCHED_PID"
            wait_for_port "async eval env server" "$ASYNC_EVAL_ENV_HOST" "$ASYNC_EVAL_ENV_PORT" 60 | tee -a "$LAUNCH_LOG"
        fi
    fi
fi

echo "Starting OpenPI server..." | tee -a "$LAUNCH_LOG"
launch_in_own_group "$OPENPI_LOG" \
    bash "$TOOLS_DIR/serve_openpi.sh" --port "$OPENPI_PORT" --gpu-id "$GPU_ID"
OPENPI_PID="$LAUNCHED_PID"
wait_for_port "OpenPI server" "$OPENPI_HOST" "$OPENPI_PORT" 300 | tee -a "$LAUNCH_LOG"

echo "Starting training..." | tee -a "$LAUNCH_LOG"
cd "$ROOT_DIR"
TRAIN_ARGS=(
    --config-path "$CONFIG_DIR"
    --config-name "$CONFIG_NAME"
    env.remote.host="$ENV_HOST"
    env.remote.port="$ENV_PORT"
    openpi.host="$OPENPI_HOST"
    openpi.port="$OPENPI_PORT"
    "${EXTRA_ARGS[@]}"
)
if [[ -n "$HYDRA_RUN_OVERRIDE" ]]; then
    TRAIN_ARGS+=("$HYDRA_RUN_OVERRIDE")
fi
TRAIN_CMD=(
    bash "$TOOLS_DIR/train.sh" "${TRAIN_ARGS[@]}"
)
echo "Training command: CUDA_VISIBLE_DEVICES=$(printf "%q" "$GPU_ID") $(format_cmd "${TRAIN_CMD[@]}")" | tee -a "$LAUNCH_LOG"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
"${TRAIN_CMD[@]}"
