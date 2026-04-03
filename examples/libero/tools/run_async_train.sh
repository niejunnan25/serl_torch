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
    echo "ERROR: run_async_train.sh requires Python 3; current PYTHON_BIN=${PYTHON_BIN}"
    echo "Hint: set SERL_CONDA_ENV=serl_torch or SERL_CONDA_PREFIX=/abs/env"
    exit 1
fi

usage() {
    cat <<'EOF'
Usage:
  bash tools/run_async_train.sh <yaml_file_name.yaml|/abs/path/to/config.yaml> [--gpu_id N] [--actor-gpu-id N] [--learner-gpu-id N] [--openpi-gpu-id N] [--output-dir DIR] [extra hydra overrides...]

Examples:
  bash tools/run_async_train.sh train_residual_sac.yaml --gpu_id 0
  bash tools/run_async_train.sh /abs/path/to/config.yaml --actor-gpu-id 0 --learner-gpu-id 1 --openpi-gpu-id 2
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

check_runtime_requirements() {
    if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import hydra  # noqa: F401
import agentlace  # noqa: F401
try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    import gymnasium  # noqa: F401
PY
    then
        echo "ERROR: run_async_train.sh requires hydra, agentlace, and gym/gymnasium in PYTHON_BIN=${PYTHON_BIN}"
        echo "Hint: ensure the serl_torch environment is activated and dependencies are installed."
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
ACTOR_GPU_ID=""
LEARNER_GPU_ID=""
OPENPI_GPU_ID=""
OUTPUT_DIR_ARG=""
EXTRA_ARGS=()
COMPOSE_OVERRIDES=()
ORIGINAL_ARGS=("$@")

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
        --actor-gpu-id)
            require_arg "$1" "${2:-}"
            ACTOR_GPU_ID="$2"
            shift 2
            ;;
        --learner-gpu-id)
            require_arg "$1" "${2:-}"
            LEARNER_GPU_ID="$2"
            shift 2
            ;;
        --openpi-gpu-id)
            require_arg "$1" "${2:-}"
            OPENPI_GPU_ID="$2"
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
                if [[ "$1" != --* ]]; then
                    COMPOSE_OVERRIDES+=("$1")
                fi
            fi
            shift
            ;;
    esac
done

normalize_async_override() {
    local arg="$1"
    if [[ "$arg" == +* ]]; then
        printf '%s\n' "$arg"
        return 0
    fi
    if [[ "$arg" == training.async.*=* ]] || [[ "$arg" == training.async=* ]]; then
        printf '++%s\n' "$arg"
        return 0
    fi
    printf '%s\n' "$arg"
}

if [[ -z "$CONFIG_ARG" ]]; then
    usage
    exit 1
fi

check_runtime_requirements

NORMALIZED_EXTRA_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
    NORMALIZED_EXTRA_ARGS+=("$(normalize_async_override "$arg")")
done

NORMALIZED_COMPOSE_OVERRIDES=()
for arg in "${COMPOSE_OVERRIDES[@]}"; do
    NORMALIZED_COMPOSE_OVERRIDES+=("$(normalize_async_override "$arg")")
done

if [[ -z "$ACTOR_GPU_ID" ]]; then
    ACTOR_GPU_ID="$GPU_ID"
fi
if [[ -z "$LEARNER_GPU_ID" ]]; then
    LEARNER_GPU_ID="$GPU_ID"
fi
if [[ -z "$OPENPI_GPU_ID" ]]; then
    OPENPI_GPU_ID="$GPU_ID"
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

IFS=$'\t' read -r ENV_BACKEND ENV_HOST ENV_PORT OPENPI_HOST OPENPI_PORT ASYNC_EVAL_ENABLED ASYNC_EVAL_ENV_HOST ASYNC_EVAL_ENV_PORT TRAINER_HOST TRAINER_PORT BROADCAST_PORT <<<"$("$PYTHON_BIN" - "$CONFIG_DIR" "$CONFIG_NAME" "${NORMALIZED_COMPOSE_OVERRIDES[@]}" <<'PY'
from hydra import compose, initialize_config_dir
import sys

config_dir = sys.argv[1]
config_name = sys.argv[2]
overrides = list(sys.argv[3:])
forced = [
    "++training.async.enabled=true",
    "++training.async.backend=agentlace",
    "++training.async.agentlace.spawn_local_worker=false",
]

with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name=config_name, overrides=forced + overrides)

env_cfg = cfg.get("env", {})
remote_cfg = env_cfg.get("remote", {})
openpi_cfg = cfg.get("openpi", {})
training_cfg = cfg.get("training", {})
async_cfg = training_cfg.get("async", {})
async_eval_cfg = training_cfg.get("async_eval", {})

env_backend = str(env_cfg.get("backend", "remote"))
env_host = str(remote_cfg.get("host", "127.0.0.1"))
env_port = int(remote_cfg.get("port", 30000))
openpi_host = str(openpi_cfg.get("host", "localhost"))
openpi_port = int(openpi_cfg.get("port", 30001))
async_eval_enabled = int(bool(async_eval_cfg.get("enabled", False)))
async_eval_env_host = str(async_eval_cfg.get("env_host", env_host))
async_eval_env_port = int(async_eval_cfg.get("env_port", 31014))
trainer_host = str(async_cfg.get("trainer_host", "127.0.0.1"))
trainer_port = int(async_cfg.get("trainer_port", 5488))
broadcast_port = int(async_cfg.get("broadcast_port", 5489))

print(
    f"{env_backend}\t{env_host}\t{env_port}\t{openpi_host}\t{openpi_port}\t"
    f"{async_eval_enabled}\t{async_eval_env_host}\t{async_eval_env_port}\t"
    f"{trainer_host}\t{trainer_port}\t{broadcast_port}"
)
PY
)"

if [[ "$ENV_BACKEND" != "remote" ]]; then
    echo "ERROR: run_async_train.sh expects env.backend=remote, got ${ENV_BACKEND}"
    exit 1
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
if [[ -n "$OUTPUT_DIR" ]]; then
    RUN_PARENT_DIR="$OUTPUT_DIR/$CONFIG_NAME"
    mkdir -p "$RUN_PARENT_DIR"
    RUN_ROOT="$RUN_PARENT_DIR/$STAMP"
    if [[ -e "$RUN_ROOT" ]]; then
        suffix=2
        while [[ -e "${RUN_ROOT}_$suffix" ]]; do
            suffix=$((suffix + 1))
        done
        RUN_ROOT="${RUN_ROOT}_$suffix"
    fi
else
    RUN_ROOT="$ROOT_DIR/outputs/libero/run_async_train_support/${STAMP}_${CONFIG_NAME}_gpu${ACTOR_GPU_ID}_$$"
fi

SUPPORT_DIR="$RUN_ROOT/support"
ACTOR_RUN_DIR="$RUN_ROOT/actor"
LEARNER_RUN_DIR="$RUN_ROOT/learner"
BOOTSTRAP_PATH="$RUN_ROOT/agentlace_bootstrap.pkl"
mkdir -p "$SUPPORT_DIR" "$ACTOR_RUN_DIR" "$LEARNER_RUN_DIR"

ENV_LOG="$SUPPORT_DIR/env_server.log"
OPENPI_LOG="$SUPPORT_DIR/openpi_server.log"
EVAL_ENV_LOG="$SUPPORT_DIR/async_eval_env_server.log"
LEARNER_LOG="$SUPPORT_DIR/learner.log"
LAUNCH_LOG="$SUPPORT_DIR/launcher.log"

ENV_PID=""
OPENPI_PID=""
EVAL_ENV_PID=""
LEARNER_PID=""
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
    stop_service "$LEARNER_PID"
    stop_service "$EVAL_ENV_PID"
    stop_service "$OPENPI_PID"
    stop_service "$ENV_PID"
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

{
    echo "=========================================="
    echo "  LIBERO run_async_train.sh"
    echo "=========================================="
    echo "  Root         : $ROOT_DIR"
    echo "  Launch cmd   : $(format_cmd bash "$TOOLS_DIR/run_async_train.sh" "${ORIGINAL_ARGS[@]}")"
    echo "  Config file  : $CONFIG_PATH"
    echo "  Config name  : $CONFIG_NAME"
    echo "  Actor GPU    : $ACTOR_GPU_ID"
    echo "  Learner GPU  : $LEARNER_GPU_ID"
    echo "  OpenPI GPU   : $OPENPI_GPU_ID"
    echo "  Env server   : ${ENV_HOST}:${ENV_PORT}"
    echo "  OpenPI       : ${OPENPI_HOST}:${OPENPI_PORT}"
    echo "  Trainer      : ${TRAINER_HOST}:${TRAINER_PORT}"
    echo "  Broadcast    : ${BROADCAST_PORT}"
    if [[ "$ASYNC_EVAL_ENABLED" == "1" ]]; then
        echo "  Async eval   : enabled"
        echo "  Eval env     : ${ASYNC_EVAL_ENV_HOST}:${ASYNC_EVAL_ENV_PORT}"
    else
        echo "  Async eval   : disabled"
    fi
    if [[ -n "$OUTPUT_DIR" ]]; then
        echo "  Output root  : $OUTPUT_DIR"
    fi
    echo "  Run root     : $RUN_ROOT"
    echo "  Bootstrap    : $BOOTSTRAP_PATH"
    echo "  Learner log  : $LEARNER_LOG"
    echo "  Extra args   : ${NORMALIZED_EXTRA_ARGS[*]:-<none>}"
    echo "=========================================="
} | tee "$LAUNCH_LOG"

TRAINER_PORT_CHECK_HOST="$TRAINER_HOST"
if [[ "$TRAINER_PORT_CHECK_HOST" == "0.0.0.0" ]]; then
    TRAINER_PORT_CHECK_HOST="127.0.0.1"
fi

if port_open "$ENV_HOST" "$ENV_PORT"; then
    echo "ERROR: env port already in use: ${ENV_HOST}:${ENV_PORT}" | tee -a "$LAUNCH_LOG"
    exit 1
fi
if port_open "$OPENPI_HOST" "$OPENPI_PORT"; then
    echo "ERROR: OpenPI port already in use: ${OPENPI_HOST}:${OPENPI_PORT}" | tee -a "$LAUNCH_LOG"
    exit 1
fi
if port_open "$TRAINER_PORT_CHECK_HOST" "$TRAINER_PORT"; then
    echo "ERROR: trainer port already in use: ${TRAINER_PORT_CHECK_HOST}:${TRAINER_PORT}" | tee -a "$LAUNCH_LOG"
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
    bash "$TOOLS_DIR/serve_openpi.sh" --port "$OPENPI_PORT" --gpu-id "$OPENPI_GPU_ID"
OPENPI_PID="$LAUNCHED_PID"
wait_for_port "OpenPI server" "$OPENPI_HOST" "$OPENPI_PORT" 300 | tee -a "$LAUNCH_LOG"

LEARNER_CMD=(
    bash "$TOOLS_DIR/run_learner.sh"
    "$CONFIG_PATH"
    --bootstrap "$BOOTSTRAP_PATH"
    --gpu_id "$LEARNER_GPU_ID"
    "${NORMALIZED_EXTRA_ARGS[@]}"
    "hydra.run.dir=$LEARNER_RUN_DIR"
)
echo "Starting learner..." | tee -a "$LAUNCH_LOG"
echo "Learner command: $(format_cmd "${LEARNER_CMD[@]}")" | tee -a "$LAUNCH_LOG"
launch_in_own_group "$LEARNER_LOG" "${LEARNER_CMD[@]}"
LEARNER_PID="$LAUNCHED_PID"
sleep 2
if ! kill -0 "$LEARNER_PID" >/dev/null 2>&1; then
    echo "ERROR: learner exited early, see $LEARNER_LOG" | tee -a "$LAUNCH_LOG"
    exit 1
fi

ACTOR_CMD=(
    bash "$TOOLS_DIR/run_actor.sh"
    "$CONFIG_PATH"
    --bootstrap "$BOOTSTRAP_PATH"
    --gpu_id "$ACTOR_GPU_ID"
    "${NORMALIZED_EXTRA_ARGS[@]}"
    "hydra.run.dir=$ACTOR_RUN_DIR"
)
echo "Starting actor..." | tee -a "$LAUNCH_LOG"
echo "Actor command: $(format_cmd "${ACTOR_CMD[@]}")" | tee -a "$LAUNCH_LOG"
cd "$ROOT_DIR"
"${ACTOR_CMD[@]}"
