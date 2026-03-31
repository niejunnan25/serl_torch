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
    echo "ERROR: run_collect_online_prefill.sh requires Python 3; current PYTHON_BIN=${PYTHON_BIN}"
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
  bash tools/run_collect_online_prefill.sh <yaml_file_name.yaml|/abs/path/to/config.yaml> [--gpu_id N] [--episodes N] [--output_dir DIR] [extra hydra overrides...]

Examples:
  bash tools/run_collect_online_prefill.sh train_residual_sac.yaml --gpu_id 0 --episodes 100
  bash tools/run_collect_online_prefill.sh /abs/path/to/train.yaml --gpu_id 1 --episodes 100 openpi.port=30011 env.remote.port=30010
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
COLLECT_ARGS=()
HYDRA_OVERRIDES=()

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
        --episodes|--output_dir)
            require_arg "$1" "${2:-}"
            COLLECT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --episodes=*|--output_dir=*)
            COLLECT_ARGS+=("$1")
            shift
            ;;
        *)
            if [[ -z "$CONFIG_ARG" ]]; then
                CONFIG_ARG="$1"
            else
                COLLECT_ARGS+=("$1")
                HYDRA_OVERRIDES+=("$1")
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

IFS=$'\t' read -r ENV_BACKEND ENV_HOST ENV_PORT OPENPI_HOST OPENPI_PORT <<<"$("$PYTHON_BIN" - "$CONFIG_DIR" "$CONFIG_NAME" "${HYDRA_OVERRIDES[@]}" <<'PY'
from hydra import compose, initialize_config_dir
import sys

config_dir = sys.argv[1]
config_name = sys.argv[2]
overrides = sys.argv[3:]
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name=config_name, overrides=overrides)

env_backend = str(cfg.get("env", {}).get("backend", "remote")).lower()
if env_backend == "remote":
    remote_cfg = cfg.get("env", {}).get("remote", {})
    env_host = str(remote_cfg.get("host", "127.0.0.1"))
    env_port = int(remote_cfg.get("port", 30000))
else:
    env_host = ""
    env_port = 0
openpi_host = str(cfg.openpi.get("host", "localhost"))
openpi_port = int(cfg.openpi.get("port", 30001))
print(f"{env_backend}\t{env_host}\t{env_port}\t{openpi_host}\t{openpi_port}")
PY
)"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
SUPPORT_ROOT="${RUN_COLLECT_PREFILL_SUPPORT_ROOT:-$ROOT_DIR/outputs/libero/run_collect_online_prefill_support}"
SUPPORT_DIR="$SUPPORT_ROOT/${STAMP}_${CONFIG_NAME}_gpu${GPU_ID}_$$"
mkdir -p "$SUPPORT_DIR"

ENV_LOG="$SUPPORT_DIR/env_server.log"
OPENPI_LOG="$SUPPORT_DIR/openpi_server.log"
LAUNCH_LOG="$SUPPORT_DIR/launcher.log"

ENV_PID=""
OPENPI_PID=""
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
    stop_service "$OPENPI_PID"
    stop_service "$ENV_PID"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

{
    echo "=========================================="
    echo "  LIBERO run_collect_online_prefill.sh"
    echo "=========================================="
    echo "  Root        : $ROOT_DIR"
    echo "  Config file : $CONFIG_PATH"
    echo "  Config name : $CONFIG_NAME"
    echo "  GPU         : $GPU_ID"
    echo "  Env backend : $ENV_BACKEND"
    if [[ "$ENV_BACKEND" == "remote" ]]; then
        echo "  Env server  : ${ENV_HOST}:${ENV_PORT}"
    else
        echo "  Env server  : <local backend, not required>"
    fi
    echo "  OpenPI      : ${OPENPI_HOST}:${OPENPI_PORT}"
    echo "  Support dir : $SUPPORT_DIR"
    echo "  Collect args: ${COLLECT_ARGS[*]:-<none>}"
    echo "  Overrides   : ${HYDRA_OVERRIDES[*]:-<none>}"
    echo "=========================================="
} | tee "$LAUNCH_LOG"

if [[ "$ENV_BACKEND" == "remote" ]]; then
    if port_open "$ENV_HOST" "$ENV_PORT"; then
        echo "Env server already running, reusing ${ENV_HOST}:${ENV_PORT}" | tee -a "$LAUNCH_LOG"
    else
        echo "Starting env server..." | tee -a "$LAUNCH_LOG"
        launch_in_own_group "$ENV_LOG" \
            bash "$TOOLS_DIR/serve_env.sh" --host "$ENV_HOST" --port "$ENV_PORT"
        ENV_PID="$LAUNCHED_PID"
        wait_for_port "env server" "$ENV_HOST" "$ENV_PORT" 60 | tee -a "$LAUNCH_LOG"
    fi
fi

if port_open "$OPENPI_HOST" "$OPENPI_PORT"; then
    echo "OpenPI server already running, reusing ${OPENPI_HOST}:${OPENPI_PORT}" | tee -a "$LAUNCH_LOG"
else
    echo "Starting OpenPI server..." | tee -a "$LAUNCH_LOG"
    launch_in_own_group "$OPENPI_LOG" \
        bash "$TOOLS_DIR/serve_openpi.sh" --port "$OPENPI_PORT" --gpu-id "$GPU_ID"
    OPENPI_PID="$LAUNCHED_PID"
    wait_for_port "OpenPI server" "$OPENPI_HOST" "$OPENPI_PORT" 300 | tee -a "$LAUNCH_LOG"
fi

echo "Starting online prefill collection..." | tee -a "$LAUNCH_LOG"
RUN_CMD=(
    "$PYTHON_BIN"
    "$ROOT_DIR/scripts/collect_online_prefill.py"
    "$CONFIG_PATH"
)
RUN_CMD+=("${COLLECT_ARGS[@]}")
if [[ "$ENV_BACKEND" == "remote" ]]; then
    RUN_CMD+=(
        "env.remote.host=$ENV_HOST"
        "env.remote.port=$ENV_PORT"
    )
fi
RUN_CMD+=(
    "openpi.host=$OPENPI_HOST"
    "openpi.port=$OPENPI_PORT"
)
"${RUN_CMD[@]}"
