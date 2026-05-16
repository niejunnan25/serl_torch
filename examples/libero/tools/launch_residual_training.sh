#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$LIBERO_DIR/../.." && pwd)"
DEFAULT_OUTPUTS_ROOT="$LIBERO_DIR/outputs"
DEFAULT_SERL_CONDA_ENV="serl_torch"
DEFAULT_WAIT_TIMEOUT_SEC=120
DEFAULT_MINICONDA_ROOT="/vla/miniconda3"
DEFAULT_CONDA_SH="$DEFAULT_MINICONDA_ROOT/etc/profile.d/conda.sh"

SCRIPT_ID=""
CONFIG_NAME=""
CONFIG_FILE=""
OUTPUT_ROOT=""
SERL_CONDA_ENV="${SERL_CONDA_ENV:-$DEFAULT_SERL_CONDA_ENV}"
LEARNER_GPU=""
ACTOR_GPU=""
ENV_GPU=""
EVAL_ENV_GPU=""
POLICY_GPU=""
BACKFILL_GPU=""
WITH_EVAL_ENV="auto"
LIBERO_ROOT_OVERRIDE=""
LIBERO_DATASETS_ROOT_OVERRIDE=""
POLICY_CONFIG_OVERRIDE="${POLICY_CONFIG:-}"
POLICY_DIR_OVERRIDE="${POLICY_DIR:-}"
OPENPI_ROOT_OVERRIDE="${OPENPI_ROOT:-}"
WAIT_TIMEOUT_SEC="$DEFAULT_WAIT_TIMEOUT_SEC"
DRY_RUN=0
CLEAN_OUTPUT_DIR=0
REUSE_OUTPUT_DIR=0
CONDA_SH="$DEFAULT_CONDA_SH"

declare -a EXTRA_HYDRA_ARGS=()
declare -a STARTED_PROCESS_NAMES=()
declare -A STARTED_PROCESS_PIDS=()
declare -A COMPLETED_PROCESS_STATUSES=()
CLEANUP_IN_PROGRESS=0
TRAINING_DRAINING=0
LEARNER_INTERRUPT_SENT=0
PROCESS_EXIT_STATUS=0

usage() {
    cat <<'EOF'
Usage:
  bash examples/libero/tools/launch_residual_training.sh \
    --script-id {1|2|3|4|5} \
    [--config-name NAME | --config-file /abs/path/to/config.yaml] \
    [--output-root DIR] \
    [--learner-gpu N] [--actor-gpu N] [--env-gpu N] [--eval-env-gpu N] \
    [--policy-gpu N] [--backfill-gpu N] \
    [--with-eval-env | --without-eval-env] \
    [--libero-root DIR] [--libero-datasets-root DIR] \
    [--policy-config NAME] [--policy-dir DIR] [--openpi-root DIR] \
    [--serl-conda-env NAME] \
    [--wait-timeout-sec N] \
    [--clean-output-dir | --reuse-output-dir] \
    [--dry-run] \
    [-- extra hydra overrides...]

Examples:
  bash examples/libero/tools/launch_residual_training.sh \
    --script-id 5 \
    --config-name exp1/train_residual_task4_exp1_scripts_5 \
    --learner-gpu 5 \
    --env-gpu 6 \
    --policy-gpu 6 \
    --eval-env-gpu 7 \
    --backfill-gpu 7 \
    --with-eval-env \
    -- \
    libero_root=/vla/users/niejunnan/codebase/serl_torch/third_party/LIBERO \
    libero_datasets_root=/vla/users/niejunnan/datasets

Notes:
  - The experiment root defaults to:
      examples/libero/outputs/<relative_config_dir>/<config_stem>/
    For configs at the root of examples/libero/configs, this remains:
      examples/libero/outputs/<config_stem>/
  - Hydra outputs are forced under:
      <exp_root>/learner
      <exp_root>/processor
      <exp_root>/actor
  - Raw rollout recycle is forced under:
      <exp_root>/rollout
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

log_note() {
    printf '[launch] %s\n' "$*"
}

die() {
    printf '[launch] ERROR: %s\n' "$*" >&2
    exit 1
}

pid_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1
}

process_group_has_members() {
    local pgid="$1"
    [[ -n "$pgid" ]] || return 1
    if command -v pgrep >/dev/null 2>&1; then
        pgrep -g "$pgid" >/dev/null 2>&1
        return $?
    fi
    ps -eo pgid= 2>/dev/null | awk -v pgid="$pgid" '$1 == pgid { found = 1; exit } END { exit !found }'
}

managed_process_alive() {
    local pid="$1"
    pid_is_running "$pid" || process_group_has_members "$pid"
}

remove_pid_file() {
    local name="$1"
    if [[ -n "${LAUNCHER_DIR:-}" ]]; then
        rm -f "$LAUNCHER_DIR/pids/${name}.pid"
    fi
}

process_completed() {
    local name="$1"
    [[ -n "${COMPLETED_PROCESS_STATUSES[$name]+set}" ]]
}

process_completed_successfully() {
    local name="$1"
    process_completed "$name" && [[ "${COMPLETED_PROCESS_STATUSES[$name]}" == "0" ]]
}

record_process_exit() {
    local name="$1"
    local pid="$2"
    PROCESS_EXIT_STATUS=0
    if [[ -n "$pid" ]] && wait "$pid" >/dev/null 2>&1; then
        PROCESS_EXIT_STATUS=0
    else
        PROCESS_EXIT_STATUS=$?
    fi
    COMPLETED_PROCESS_STATUSES["$name"]="$PROCESS_EXIT_STATUS"
    remove_pid_file "$name"
    log_note "$name exited with status=$PROCESS_EXIT_STATUS"
}

signal_managed_process() {
    local pid="$1"
    local signal="$2"
    [[ -n "$pid" ]] || return 0
    if kill -s "$signal" -- "-$pid" >/dev/null 2>&1; then
        return 0
    fi
    if pid_is_running "$pid"; then
        kill -s "$signal" "$pid" >/dev/null 2>&1 || true
    fi
}

maybe_interrupt_learner_for_final_drain() {
    local learner_pid=""
    (( TRAINING_DRAINING )) || return 0
    (( LEARNER_INTERRUPT_SENT )) && return 0
    process_completed "learner" && return 0
    if (( START_PROCESSOR )) && ! process_completed "processor"; then
        return 0
    fi
    learner_pid="${STARTED_PROCESS_PIDS[learner]:-}"
    if pid_is_running "$learner_pid"; then
        log_note "actor finished and processor drained; sending SIGINT to learner for final eval/summary drain"
        # Signal only the learner process so its async eval worker can keep draining
        # and be joined from the learner's graceful shutdown path.
        kill -s INT "$learner_pid" >/dev/null 2>&1 || true
        LEARNER_INTERRUPT_SENT=1
    fi
}

training_roles_drained() {
    local processor_done=0
    process_completed_successfully "learner" || return 1
    if (( START_PROCESSOR )); then
        process_completed_successfully "processor" && processor_done=1
    else
        processor_done=1
    fi
    (( processor_done ))
}

cleanup_started_processes() {
    local reason="${1:-launcher shutdown}"
    local timeout_sec="${2:-15}"
    local idx=""
    local name=""
    local pid=""
    local deadline=""
    local still_running=0

    (( DRY_RUN )) && return 0
    (( CLEANUP_IN_PROGRESS )) && return 0
    CLEANUP_IN_PROGRESS=1

    if ((${#STARTED_PROCESS_NAMES[@]} == 0)); then
        return 0
    fi

    log_note "stopping managed processes ($reason)"

    for ((idx=${#STARTED_PROCESS_NAMES[@]} - 1; idx >= 0; idx--)); do
        name="${STARTED_PROCESS_NAMES[$idx]}"
        pid="${STARTED_PROCESS_PIDS[$name]:-}"
        if managed_process_alive "$pid"; then
            log_note "sending SIGTERM to $name process group pgid=$pid"
            signal_managed_process "$pid" TERM
        fi
    done

    deadline=$(( $(date +%s) + timeout_sec ))
    while (( $(date +%s) < deadline )); do
        still_running=0
        for name in "${STARTED_PROCESS_NAMES[@]}"; do
            pid="${STARTED_PROCESS_PIDS[$name]:-}"
            if managed_process_alive "$pid"; then
                still_running=1
                break
            fi
        done
        (( still_running == 0 )) && break
        sleep 1
    done

    for ((idx=${#STARTED_PROCESS_NAMES[@]} - 1; idx >= 0; idx--)); do
        name="${STARTED_PROCESS_NAMES[$idx]}"
        pid="${STARTED_PROCESS_PIDS[$name]:-}"
        if managed_process_alive "$pid"; then
            log_note "sending SIGKILL to $name process group pgid=$pid"
            signal_managed_process "$pid" KILL
        fi
        if [[ -n "$pid" ]]; then
            wait "$pid" >/dev/null 2>&1 || true
        fi
        remove_pid_file "$name"
    done
}

handle_signal() {
    local signal_name="$1"
    local exit_code="$2"
    trap - INT TERM HUP EXIT
    log_note "received $signal_name; shutting down all managed processes"
    cleanup_started_processes "signal $signal_name"
    exit "$exit_code"
}

handle_exit() {
    local exit_code="$1"
    trap - EXIT
    cleanup_started_processes "launcher exit status=$exit_code"
    exit "$exit_code"
}

ensure_conda_command() {
    [[ -f "$CONDA_SH" ]] || die "conda.sh not found at $CONDA_SH"
    # shellcheck source=/dev/null
    source "$CONDA_SH"
    command -v conda >/dev/null 2>&1 || die "conda command not available after sourcing $CONDA_SH"
}

build_conda_shell_command() {
    local inner_cmd=""
    local conda_sh_escaped=""
    local conda_env_escaped=""
    inner_cmd="$(format_cmd "$@")"
    printf -v conda_sh_escaped "%q" "$CONDA_SH"
    printf -v conda_env_escaped "%q" "$SERL_CONDA_ENV"
    printf "source %s && conda activate %s && exec %s" \
        "$conda_sh_escaped" \
        "$conda_env_escaped" \
        "$inner_cmd"
}

resolve_path() {
    local input_path="$1"
    python3 - "$input_path" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

is_local_host() {
    local host="$1"
    [[ "$host" == "127.0.0.1" || "$host" == "localhost" || "$host" == "0.0.0.0" ]]
}

port_is_reachable() {
    local host="$1"
    local port="$2"
    timeout 1 bash -lc "</dev/tcp/${host}/${port}" >/dev/null 2>&1
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local label="$3"
    local timeout_sec="$4"
    if (( DRY_RUN )); then
        log_note "[dry-run] skip wait for $label at ${host}:${port}"
        return 0
    fi
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if port_is_reachable "$host" "$port"; then
            log_note "$label is ready at ${host}:${port}"
            return 0
        fi
        if (( "$(date +%s)" - start_ts >= timeout_sec )); then
            die "timed out waiting for $label at ${host}:${port}; check logs"
        fi
        sleep 1
    done
}

assert_port_unused() {
    local host="$1"
    local port="$2"
    local label="$3"
    if (( DRY_RUN )); then
        return 0
    fi
    if port_is_reachable "$host" "$port"; then
        die "$label port already responds at ${host}:${port}; refusing to launch over an existing service"
    fi
}

write_text_file() {
    local path="$1"
    shift
    mkdir -p "$(dirname "$path")"
    printf '%s\n' "$@" >"$path"
}

start_logged_process() {
    local name="$1"
    local log_file="$2"
    shift 2
    local cmd_str
    local launch_cmd
    local pid
    cmd_str="$(format_cmd "$@")"
    write_text_file "$LAUNCHER_DIR/commands/${name}.txt" "$cmd_str"
    if (( DRY_RUN )); then
        log_note "[dry-run] $name"
        printf '  %s\n' "$cmd_str"
        return 0
    fi
    mkdir -p "$(dirname "$log_file")"
    : >"$log_file"
    launch_cmd="cd $(printf '%q' "$REPO_ROOT") && "
    if command -v stdbuf >/dev/null 2>&1; then
        launch_cmd+="exec stdbuf -oL -eL $cmd_str"
    else
        launch_cmd+="exec $cmd_str"
    fi
    setsid bash -lc "$launch_cmd" >>"$log_file" 2>&1 &
    pid="$!"
    write_text_file "$LAUNCHER_DIR/pids/${name}.pid" "$pid"
    STARTED_PROCESS_NAMES+=("$name")
    STARTED_PROCESS_PIDS["$name"]="$pid"
    log_note "started $name pid=$pid log=$log_file"
}

assert_pid_running() {
    local name="$1"
    local pid_file="$LAUNCHER_DIR/pids/${name}.pid"
    local pid
    [[ -f "$pid_file" ]] || die "missing pid file for $name"
    pid="$(<"$pid_file")"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
        die "$name exited early; inspect $LAUNCHER_DIR/commands/${name}.txt and logs"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --script-id)
            SCRIPT_ID="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --serl-conda-env)
            SERL_CONDA_ENV="$2"
            shift 2
            ;;
        --learner-gpu)
            LEARNER_GPU="$2"
            shift 2
            ;;
        --actor-gpu)
            ACTOR_GPU="$2"
            shift 2
            ;;
        --env-gpu)
            ENV_GPU="$2"
            shift 2
            ;;
        --eval-env-gpu)
            EVAL_ENV_GPU="$2"
            shift 2
            ;;
        --policy-gpu)
            POLICY_GPU="$2"
            shift 2
            ;;
        --backfill-gpu)
            BACKFILL_GPU="$2"
            shift 2
            ;;
        --with-eval-env)
            WITH_EVAL_ENV="true"
            shift
            ;;
        --without-eval-env)
            WITH_EVAL_ENV="false"
            shift
            ;;
        --libero-root)
            LIBERO_ROOT_OVERRIDE="$2"
            shift 2
            ;;
        --libero-datasets-root)
            LIBERO_DATASETS_ROOT_OVERRIDE="$2"
            shift 2
            ;;
        --policy-config)
            POLICY_CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --policy-dir)
            POLICY_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --openpi-root)
            OPENPI_ROOT_OVERRIDE="$2"
            shift 2
            ;;
        --wait-timeout-sec)
            WAIT_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --clean-output-dir)
            CLEAN_OUTPUT_DIR=1
            shift
            ;;
        --reuse-output-dir)
            REUSE_OUTPUT_DIR=1
            shift
            ;;
        --)
            shift
            EXTRA_HYDRA_ARGS+=("$@")
            break
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$SCRIPT_ID" ]] || die "--script-id is required"
case "$SCRIPT_ID" in
    1) TRAINING_SCRIPT="$LIBERO_DIR/scripts/run_residual_training_1_baseline.py" ;;
    2) TRAINING_SCRIPT="$LIBERO_DIR/scripts/run_residual_training_2_chunk_local.py" ;;
    3) TRAINING_SCRIPT="$LIBERO_DIR/scripts/run_residual_training_3_split_proto.py" ;;
    4) TRAINING_SCRIPT="$LIBERO_DIR/scripts/run_residual_training_4_split_refined.py" ;;
    5) TRAINING_SCRIPT="$LIBERO_DIR/scripts/run_residual_training_5_split_pipeline.py" ;;
    *) die "--script-id must be one of 1, 2, 3, 4, 5" ;;
esac

if [[ -n "$CONFIG_FILE" && -n "$CONFIG_NAME" ]]; then
    die "use either --config-file or --config-name, not both"
fi
if [[ -z "$CONFIG_FILE" && -z "$CONFIG_NAME" ]]; then
    die "one of --config-file or --config-name is required"
fi
if [[ -n "$CONFIG_NAME" ]]; then
    CONFIG_FILE="$LIBERO_DIR/configs/${CONFIG_NAME}.yaml"
fi
[[ -f "$CONFIG_FILE" ]] || die "config file not found: $CONFIG_FILE"
CONFIG_FILE="$(resolve_path "$CONFIG_FILE")"
CONFIG_DIR="$(dirname "$CONFIG_FILE")"
CONFIG_BASENAME="$(basename "$CONFIG_FILE")"
CONFIG_STEM="${CONFIG_BASENAME%.yaml}"
CONFIGS_ROOT="$(resolve_path "$LIBERO_DIR/configs")"
CONFIG_OUTPUT_SUBDIR=""

if [[ -n "$CONFIG_NAME" ]]; then
    CONFIG_SEARCH_DIR="$CONFIGS_ROOT"
    CONFIG_COMPOSE_NAME="$CONFIG_NAME"
elif [[ "$CONFIG_FILE" == "$CONFIGS_ROOT/"* ]]; then
    CONFIG_SEARCH_DIR="$CONFIGS_ROOT"
    CONFIG_COMPOSE_NAME="${CONFIG_FILE#$CONFIGS_ROOT/}"
    CONFIG_COMPOSE_NAME="${CONFIG_COMPOSE_NAME%.yaml}"
else
    CONFIG_SEARCH_DIR="$CONFIG_DIR"
    CONFIG_COMPOSE_NAME="$CONFIG_STEM"
fi

if [[ "$CONFIG_FILE" == "$CONFIGS_ROOT/"* ]]; then
    CONFIG_REL_PATH="${CONFIG_FILE#$CONFIGS_ROOT/}"
    CONFIG_REL_DIR="$(dirname "$CONFIG_REL_PATH")"
    if [[ "$CONFIG_REL_DIR" != "." ]]; then
        CONFIG_OUTPUT_SUBDIR="$CONFIG_REL_DIR"
    fi
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    if [[ -n "$CONFIG_OUTPUT_SUBDIR" ]]; then
        OUTPUT_ROOT="$DEFAULT_OUTPUTS_ROOT/$CONFIG_OUTPUT_SUBDIR/$CONFIG_STEM"
    else
        OUTPUT_ROOT="$DEFAULT_OUTPUTS_ROOT/$CONFIG_STEM"
    fi
fi
OUTPUT_ROOT="$(resolve_path "$OUTPUT_ROOT")"

if (( CLEAN_OUTPUT_DIR )) && (( REUSE_OUTPUT_DIR )); then
    die "--clean-output-dir and --reuse-output-dir cannot be used together"
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    if (( CLEAN_OUTPUT_DIR )); then
        [[ "$OUTPUT_ROOT" == "$DEFAULT_OUTPUTS_ROOT/"* ]] || die "refusing to clean output dir outside $DEFAULT_OUTPUTS_ROOT"
        rm -rf "$OUTPUT_ROOT"
    elif (( REUSE_OUTPUT_DIR )); then
        log_note "reusing existing output root: $OUTPUT_ROOT"
    elif [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
        die "output root already exists and is not empty: $OUTPUT_ROOT (use --clean-output-dir or --reuse-output-dir)"
    fi
fi

mkdir -p "$OUTPUT_ROOT"
LAUNCHER_DIR="$OUTPUT_ROOT/.launcher"
SERVICES_DIR="$OUTPUT_ROOT/services"
mkdir -p "$LAUNCHER_DIR/commands" "$LAUNCHER_DIR/pids" "$SERVICES_DIR"

trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP
trap 'handle_exit $?' EXIT

ensure_conda_command

CONFIG_PARSER_SCRIPT="$LAUNCHER_DIR/parse_config.py"
cat >"$CONFIG_PARSER_SCRIPT" <<'PY'
import shlex
import sys
from hydra import compose, initialize_config_dir

config_file = sys.argv[1]
config_dir = sys.argv[2]
config_name = sys.argv[3]

with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name=config_name)

def lookup(path, default=None):
    current = cfg
    for part in path.split("."):
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(part, None)
        else:
            return default
        if current is None:
            return default
    return current

def emit(key, value):
    if isinstance(value, bool):
        text = "1" if value else "0"
    elif value is None:
        text = ""
    else:
        text = str(value)
    print(f"{key}={shlex.quote(text)}")

emit("CFG_ENV_BACKEND", lookup("env.backend", "remote"))
emit("CFG_ENV_HOST", lookup("env.remote.host", "127.0.0.1"))
emit("CFG_ENV_PORT", lookup("env.remote.port", "30000"))
emit("CFG_ASYNC_EVAL_ENABLED", lookup("training.async_eval.enabled", False))
emit("CFG_ASYNC_EVAL_ENV_BACKEND", lookup("training.async_eval.env.backend", "remote"))
emit("CFG_ASYNC_EVAL_HOST", lookup("training.async_eval.env.remote.host", "127.0.0.1"))
emit("CFG_ASYNC_EVAL_PORT", lookup("training.async_eval.env.remote.port", "30010"))
emit("CFG_POLICY_TYPE", lookup("policy.type", "openpi"))
emit("CFG_POLICY_HOST", lookup("policy.host", "127.0.0.1"))
emit("CFG_POLICY_PORT", lookup("policy.port", "30001"))
emit("CFG_BACKFILL_ENABLED", lookup("backfill_policy.enabled", False))
emit("CFG_BACKFILL_HOST", lookup("backfill_policy.host", "127.0.0.1"))
emit("CFG_BACKFILL_PORT", lookup("backfill_policy.port", "30002"))
emit("CFG_RECYCLE_ENABLED", lookup("recycle.enabled", False))
PY
CONFIG_EXPORTS="$(
    bash -lc "$(build_conda_shell_command \
        python "$CONFIG_PARSER_SCRIPT" "$CONFIG_FILE" "$CONFIG_SEARCH_DIR" "$CONFIG_COMPOSE_NAME")"
)"
while IFS= read -r line; do
    eval "$line"
done <<< "$CONFIG_EXPORTS"

[[ -n "$CFG_POLICY_TYPE" ]] || die "failed to parse policy.type from config"

if [[ -z "$ACTOR_GPU" && -n "$POLICY_GPU" ]]; then
    ACTOR_GPU="$POLICY_GPU"
fi
if [[ -z "$BACKFILL_GPU" && -n "$POLICY_GPU" ]]; then
    BACKFILL_GPU="$POLICY_GPU"
fi

START_EVAL_ENV=0
if [[ "$WITH_EVAL_ENV" == "true" ]]; then
    START_EVAL_ENV=1
elif [[ "$WITH_EVAL_ENV" == "false" ]]; then
    START_EVAL_ENV=0
elif [[ "$CFG_ASYNC_EVAL_ENABLED" == "1" && "$CFG_ASYNC_EVAL_ENV_BACKEND" == "remote" ]]; then
    START_EVAL_ENV=1
fi

START_PROCESSOR=0
if [[ "$SCRIPT_ID" == "3" || "$SCRIPT_ID" == "4" || "$SCRIPT_ID" == "5" ]]; then
    START_PROCESSOR=1
fi

START_BACKFILL_POLICY=0
if [[ "$CFG_BACKFILL_ENABLED" == "1" ]]; then
    START_BACKFILL_POLICY=1
fi

if [[ "$CFG_POLICY_TYPE" != "openpi" ]]; then
    die "launcher currently supports policy.type=openpi only; got $CFG_POLICY_TYPE"
fi
if [[ "$START_BACKFILL_POLICY" == "1" && "$SCRIPT_ID" == "5" && -z "$BACKFILL_GPU" ]]; then
    die "backfill policy is enabled; provide --backfill-gpu (or --policy-gpu to reuse the same GPU)"
fi
if [[ -z "$POLICY_GPU" ]]; then
    die "launching the policy server requires --policy-gpu"
fi
if [[ "$SCRIPT_ID" == "5" && "$CFG_BACKFILL_ENABLED" != "1" ]]; then
    die "script 5 requires backfill_policy.enabled=true in the config"
fi

if [[ "$CFG_ENV_BACKEND" == "remote" ]]; then
    is_local_host "$CFG_ENV_HOST" || die "train env host must be local for launcher-managed env server; got $CFG_ENV_HOST"
fi
if (( START_EVAL_ENV )) && [[ "$CFG_ASYNC_EVAL_ENV_BACKEND" == "remote" ]]; then
    is_local_host "$CFG_ASYNC_EVAL_HOST" || die "eval env host must be local for launcher-managed eval env server; got $CFG_ASYNC_EVAL_HOST"
fi
if [[ -n "$ENV_GPU" && ! "$ENV_GPU" =~ ^[0-9]+$ ]]; then
    die "--env-gpu must be a single non-negative integer, got $ENV_GPU"
fi
if [[ -n "$EVAL_ENV_GPU" && ! "$EVAL_ENV_GPU" =~ ^[0-9]+$ ]]; then
    die "--eval-env-gpu must be a single non-negative integer, got $EVAL_ENV_GPU"
fi
if [[ -z "$EVAL_ENV_GPU" ]]; then
    EVAL_ENV_GPU="$ENV_GPU"
fi
is_local_host "$CFG_POLICY_HOST" || die "policy host must be local for launcher-managed policy server; got $CFG_POLICY_HOST"
if (( START_BACKFILL_POLICY )); then
    is_local_host "$CFG_BACKFILL_HOST" || die "backfill host must be local for launcher-managed backfill server; got $CFG_BACKFILL_HOST"
fi

mkdir -p "$OUTPUT_ROOT/actor" "$OUTPUT_ROOT/learner"
if (( START_PROCESSOR )); then
    mkdir -p "$OUTPUT_ROOT/processor"
fi
if [[ "${CFG_RECYCLE_ENABLED:-0}" == "1" ]]; then
    mkdir -p "$OUTPUT_ROOT/rollout"
fi

cat >"$OUTPUT_ROOT/launch_manifest.txt" <<EOF
config_file=$CONFIG_FILE
config_dir=$CONFIG_SEARCH_DIR
config_name=$CONFIG_COMPOSE_NAME
script_id=$SCRIPT_ID
training_script=$TRAINING_SCRIPT
output_root=$OUTPUT_ROOT
serl_conda_env=$SERL_CONDA_ENV
policy_type=$CFG_POLICY_TYPE
policy_host=$CFG_POLICY_HOST
policy_port=$CFG_POLICY_PORT
train_env_gpu=$ENV_GPU
eval_env_gpu=$EVAL_ENV_GPU
backfill_enabled=$CFG_BACKFILL_ENABLED
backfill_host=$CFG_BACKFILL_HOST
backfill_port=$CFG_BACKFILL_PORT
train_env_backend=$CFG_ENV_BACKEND
train_env_host=$CFG_ENV_HOST
train_env_port=$CFG_ENV_PORT
async_eval_enabled=$CFG_ASYNC_EVAL_ENABLED
async_eval_host=$CFG_ASYNC_EVAL_HOST
async_eval_port=$CFG_ASYNC_EVAL_PORT
EOF
cp "$CONFIG_FILE" "$OUTPUT_ROOT/config_source.yaml"

declare -a COMMON_HYDRA_ARGS
COMMON_HYDRA_ARGS=(
    --config-path "$CONFIG_SEARCH_DIR"
    --config-name "$CONFIG_COMPOSE_NAME"
    "launch.output_root=$OUTPUT_ROOT"
    "recycle.output_root=$OUTPUT_ROOT/rollout"
)
if [[ -n "$LIBERO_ROOT_OVERRIDE" ]]; then
    COMMON_HYDRA_ARGS+=("libero_root=$LIBERO_ROOT_OVERRIDE")
fi
if [[ -n "$LIBERO_DATASETS_ROOT_OVERRIDE" ]]; then
    COMMON_HYDRA_ARGS+=("libero_datasets_root=$LIBERO_DATASETS_ROOT_OVERRIDE")
fi
if ((${#EXTRA_HYDRA_ARGS[@]} > 0)); then
    COMMON_HYDRA_ARGS+=("${EXTRA_HYDRA_ARGS[@]}")
fi

build_training_cmd() {
    local role="$1"
    local run_dir="$2"
    local gpu_id="$3"
    local training_shell_cmd=""
    local -a cmd
    cmd=()
    if [[ -n "$gpu_id" ]]; then
        cmd+=(env "CUDA_VISIBLE_DEVICES=$gpu_id")
    fi
    training_shell_cmd="$(build_conda_shell_command \
        python "$TRAINING_SCRIPT" \
        "${COMMON_HYDRA_ARGS[@]}" \
        "runtime.role=$role" \
        "hydra.run.dir=$run_dir")"
    cmd+=(
        bash -lc "$training_shell_cmd"
    )
    printf '%s\0' "${cmd[@]}"
}

readarray -d '' -t LEARNER_CMD < <(build_training_cmd learner "$OUTPUT_ROOT/learner" "$LEARNER_GPU")
readarray -d '' -t ACTOR_CMD < <(build_training_cmd actor "$OUTPUT_ROOT/actor" "$ACTOR_GPU")
if (( START_PROCESSOR )); then
    readarray -d '' -t PROCESSOR_CMD < <(build_training_cmd processor "$OUTPUT_ROOT/processor" "")
fi

log_note "experiment root: $OUTPUT_ROOT"
log_note "config file     : $CONFIG_FILE"
log_note "training script : $TRAINING_SCRIPT"

if [[ "$CFG_ENV_BACKEND" == "remote" ]]; then
    assert_port_unused "$CFG_ENV_HOST" "$CFG_ENV_PORT" "train env"
    declare -a TRAIN_ENV_CMD
    TRAIN_ENV_CMD=(
        bash "$LIBERO_DIR/tools/serve_env.sh"
        --host "$CFG_ENV_HOST"
        --port "$CFG_ENV_PORT"
    )
    if [[ -n "$ENV_GPU" ]]; then
        TRAIN_ENV_CMD+=(--gpu-id "$ENV_GPU")
    fi
    start_logged_process \
        "train_env" \
        "$SERVICES_DIR/train_env.log" \
        "${TRAIN_ENV_CMD[@]}"
    wait_for_port "$CFG_ENV_HOST" "$CFG_ENV_PORT" "train env" "$WAIT_TIMEOUT_SEC"
fi

if (( START_EVAL_ENV )) && [[ "$CFG_ASYNC_EVAL_ENV_BACKEND" == "remote" ]]; then
    if [[ "$CFG_ASYNC_EVAL_PORT" == "$CFG_ENV_PORT" && "$CFG_ASYNC_EVAL_HOST" == "$CFG_ENV_HOST" ]]; then
        die "eval env port must differ from train env port"
    fi
    assert_port_unused "$CFG_ASYNC_EVAL_HOST" "$CFG_ASYNC_EVAL_PORT" "eval env"
    declare -a EVAL_ENV_CMD
    EVAL_ENV_CMD=(
        bash "$LIBERO_DIR/tools/serve_env.sh"
        --host "$CFG_ASYNC_EVAL_HOST"
        --port "$CFG_ASYNC_EVAL_PORT"
    )
    if [[ -n "$EVAL_ENV_GPU" ]]; then
        EVAL_ENV_CMD+=(--gpu-id "$EVAL_ENV_GPU")
    fi
    start_logged_process \
        "eval_env" \
        "$SERVICES_DIR/eval_env.log" \
        "${EVAL_ENV_CMD[@]}"
    wait_for_port "$CFG_ASYNC_EVAL_HOST" "$CFG_ASYNC_EVAL_PORT" "eval env" "$WAIT_TIMEOUT_SEC"
fi

assert_port_unused "$CFG_POLICY_HOST" "$CFG_POLICY_PORT" "policy"
declare -a POLICY_SERVER_CMD
POLICY_SERVER_CMD=()
if [[ -n "$OPENPI_ROOT_OVERRIDE" || -n "$POLICY_CONFIG_OVERRIDE" || -n "$POLICY_DIR_OVERRIDE" ]]; then
    POLICY_SERVER_CMD+=(env)
    if [[ -n "$OPENPI_ROOT_OVERRIDE" ]]; then
        POLICY_SERVER_CMD+=("OPENPI_ROOT=$OPENPI_ROOT_OVERRIDE")
    fi
    if [[ -n "$POLICY_CONFIG_OVERRIDE" ]]; then
        POLICY_SERVER_CMD+=("POLICY_CONFIG=$POLICY_CONFIG_OVERRIDE")
    fi
    if [[ -n "$POLICY_DIR_OVERRIDE" ]]; then
        POLICY_SERVER_CMD+=("POLICY_DIR=$POLICY_DIR_OVERRIDE")
    fi
fi
POLICY_SERVER_CMD+=(
    bash "$LIBERO_DIR/tools/serve_openpi_10000_policy.sh"
    --gpu-id "$POLICY_GPU"
    --port "$CFG_POLICY_PORT"
)
start_logged_process "policy" "$SERVICES_DIR/policy.log" "${POLICY_SERVER_CMD[@]}"
wait_for_port "$CFG_POLICY_HOST" "$CFG_POLICY_PORT" "policy" "$WAIT_TIMEOUT_SEC"

if (( START_BACKFILL_POLICY )); then
    if [[ "$CFG_BACKFILL_PORT" == "$CFG_POLICY_PORT" && "$CFG_BACKFILL_HOST" == "$CFG_POLICY_HOST" ]]; then
        die "backfill policy port must differ from policy.port when launcher manages both services"
    fi
    assert_port_unused "$CFG_BACKFILL_HOST" "$CFG_BACKFILL_PORT" "backfill policy"
    declare -a BACKFILL_SERVER_CMD
    BACKFILL_SERVER_CMD=()
    if [[ -n "$OPENPI_ROOT_OVERRIDE" || -n "$POLICY_CONFIG_OVERRIDE" || -n "$POLICY_DIR_OVERRIDE" ]]; then
        BACKFILL_SERVER_CMD+=(env)
        if [[ -n "$OPENPI_ROOT_OVERRIDE" ]]; then
            BACKFILL_SERVER_CMD+=("OPENPI_ROOT=$OPENPI_ROOT_OVERRIDE")
        fi
        if [[ -n "$POLICY_CONFIG_OVERRIDE" ]]; then
            BACKFILL_SERVER_CMD+=("POLICY_CONFIG=$POLICY_CONFIG_OVERRIDE")
        fi
        if [[ -n "$POLICY_DIR_OVERRIDE" ]]; then
            BACKFILL_SERVER_CMD+=("POLICY_DIR=$POLICY_DIR_OVERRIDE")
        fi
    fi
    BACKFILL_SERVER_CMD+=(
        bash "$LIBERO_DIR/tools/serve_openpi_10000_policy.sh"
        --gpu-id "$BACKFILL_GPU"
        --port "$CFG_BACKFILL_PORT"
    )
    start_logged_process "backfill_policy" "$SERVICES_DIR/backfill_policy.log" "${BACKFILL_SERVER_CMD[@]}"
    wait_for_port "$CFG_BACKFILL_HOST" "$CFG_BACKFILL_PORT" "backfill policy" "$WAIT_TIMEOUT_SEC"
fi

start_logged_process "learner" "$OUTPUT_ROOT/learner/launcher.log" "${LEARNER_CMD[@]}"
if (( ! DRY_RUN )); then
    sleep 2
    assert_pid_running "learner"
fi

if (( START_PROCESSOR )); then
    start_logged_process "processor" "$OUTPUT_ROOT/processor/launcher.log" "${PROCESSOR_CMD[@]}"
    if (( ! DRY_RUN )); then
        sleep 2
        assert_pid_running "processor"
    fi
fi

start_logged_process "actor" "$OUTPUT_ROOT/actor/launcher.log" "${ACTOR_CMD[@]}"
if (( ! DRY_RUN )); then
    sleep 2
    assert_pid_running "actor"
fi

if (( DRY_RUN )); then
    log_note "dry run completed; nothing was launched"
    exit 0
fi

log_note "launch complete"
log_note "experiment root : $OUTPUT_ROOT"
log_note "service logs    : $SERVICES_DIR"
log_note "learner log     : $OUTPUT_ROOT/learner/launcher.log"
if (( START_PROCESSOR )); then
    log_note "processor log   : $OUTPUT_ROOT/processor/launcher.log"
fi
log_note "actor log       : $OUTPUT_ROOT/actor/launcher.log"
log_note "pid files       : $LAUNCHER_DIR/pids"
log_note "watch learner   : $(format_cmd tail -f "$OUTPUT_ROOT/learner/launcher.log")"
if (( START_PROCESSOR )); then
    log_note "watch processor : $(format_cmd tail -f "$OUTPUT_ROOT/processor/launcher.log")"
fi
log_note "watch actor     : $(format_cmd tail -f "$OUTPUT_ROOT/actor/launcher.log")"
if [[ "$CFG_ENV_BACKEND" == "remote" ]]; then
    log_note "watch train env : $(format_cmd tail -f "$SERVICES_DIR/train_env.log")"
fi
if (( START_EVAL_ENV )) && [[ "$CFG_ASYNC_EVAL_ENV_BACKEND" == "remote" ]]; then
    log_note "watch eval env  : $(format_cmd tail -f "$SERVICES_DIR/eval_env.log")"
fi
log_note "watch policy    : $(format_cmd tail -f "$SERVICES_DIR/policy.log")"
if (( START_BACKFILL_POLICY )); then
    log_note "watch backfill  : $(format_cmd tail -f "$SERVICES_DIR/backfill_policy.log")"
fi
log_note "launcher mode   : attached (press Ctrl+C to stop all managed processes)"

while true; do
    for name in "${STARTED_PROCESS_NAMES[@]}"; do
        process_completed "$name" && continue
        pid="${STARTED_PROCESS_PIDS[$name]:-}"
        if ! pid_is_running "$pid"; then
            record_process_exit "$name" "$pid"
            if (( TRAINING_DRAINING )); then
                case "$name" in
                    learner|processor)
                        if (( PROCESS_EXIT_STATUS != 0 )); then
                            die "$name exited with status=$PROCESS_EXIT_STATUS during final drain; launcher is shutting down the remaining managed processes"
                        fi
                        ;;
                    *)
                        die "$name exited with status=$PROCESS_EXIT_STATUS during final drain; launcher is shutting down the remaining managed processes"
                        ;;
                esac
            else
                case "$name" in
                    actor)
                        if (( PROCESS_EXIT_STATUS == 0 )); then
                            TRAINING_DRAINING=1
                            log_note "actor exited cleanly; waiting for processor drain, then learner final eval/summary"
                        else
                            die "actor exited with status=$PROCESS_EXIT_STATUS; launcher is shutting down the remaining managed processes"
                        fi
                        ;;
                    processor)
                        if (( PROCESS_EXIT_STATUS == 0 )); then
                            log_note "processor exited cleanly; waiting for actor completion before learner final drain"
                        else
                            die "processor exited with status=$PROCESS_EXIT_STATUS; launcher is shutting down the remaining managed processes"
                        fi
                        ;;
                    *)
                        die "$name exited with status=$PROCESS_EXIT_STATUS; launcher is shutting down the remaining managed processes"
                        ;;
                esac
            fi
        fi
    done
    maybe_interrupt_learner_for_final_drain
    if (( TRAINING_DRAINING )) && training_roles_drained; then
        log_note "learner and processor completed final drain; shutting down remaining services"
        cleanup_started_processes "training complete" 30
        trap - EXIT
        exit 0
    fi
    sleep 2
done
