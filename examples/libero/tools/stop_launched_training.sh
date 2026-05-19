#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=""
TIMEOUT_SEC=15
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash examples/libero/tools/stop_launched_training.sh \
    --output-root /abs/path/to/examples/libero/outputs/<run_dir> \
    [--timeout-sec 15] \
    [--dry-run]

Notes:
  - Run this on the same host where the launcher started the processes.
  - This script reads <output_root>/.launcher/pids/*.pid and sends:
      1. SIGTERM
      2. SIGKILL after the timeout for survivors
EOF
}

log_note() {
    printf '[stop] %s\n' "$*"
}

die() {
    printf '[stop] ERROR: %s\n' "$*" >&2
    exit 1
}

pid_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1
}

signal_managed_process() {
    local pid="$1"
    local signal="$2"
    kill -s "$signal" -- "-$pid" >/dev/null 2>&1 || kill -s "$signal" "$pid" >/dev/null 2>&1 || true
}

pid_cmdline() {
    local pid="$1"
    ps -o args= -p "$pid" 2>/dev/null || true
}

build_expected_markers() {
    local name="$1"
    local cmd_file="$LAUNCHER_DIR/commands/${name}.txt"
    local cmd_text=""
    local port=""

    [[ -f "$cmd_file" ]] || return 0
    cmd_text="$(<"$cmd_file")"

    if [[ "$cmd_text" == *"launch.output_root=$OUTPUT_ROOT"* ]]; then
        printf '%s\n' "launch.output_root=$OUTPUT_ROOT"
        return 0
    fi

    if [[ "$cmd_text" == *"serve_env.sh"* ]]; then
        printf '%s\n' "serve_env.sh"
        port="$(printf '%s' "$cmd_text" | sed -n 's/.*--port \([0-9][0-9]*\).*/\1/p' | head -n 1)"
        [[ -n "$port" ]] && printf '%s\n' "--port $port"
        return 0
    fi

    if [[ "$cmd_text" == *"serve_openpi_10000_policy.sh"* ]]; then
        printf '%s\n' "serve_openpi_10000_policy.sh"
        port="$(printf '%s' "$cmd_text" | sed -n 's/.*--port \([0-9][0-9]*\).*/\1/p' | head -n 1)"
        [[ -n "$port" ]] && printf '%s\n' "--port $port"
        return 0
    fi

    if [[ "$cmd_text" == *"reserve_cuda_memory.py"* ]]; then
        printf '%s\n' "reserve_cuda_memory.py"
        return 0
    fi
}

pid_matches_expected() {
    local pid="$1"
    local name="$2"
    local cmdline=""
    local matched_any=0
    local marker=""

    cmdline="$(pid_cmdline "$pid")"
    [[ -n "$cmdline" ]] || return 1

    while IFS= read -r marker; do
        [[ -n "$marker" ]] || continue
        matched_any=1
        if [[ "$cmdline" != *"$marker"* ]]; then
            return 1
        fi
    done < <(build_expected_markers "$name")

    (( matched_any == 1 )) || return 1
    return 0
}

remove_pid_file() {
    local pid_file="$1"
    rm -f "$pid_file"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --timeout-sec)
            TIMEOUT_SEC="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
OUTPUT_ROOT="$(python3 - "$OUTPUT_ROOT" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"

LAUNCHER_DIR="$OUTPUT_ROOT/.launcher"
PID_DIR="$LAUNCHER_DIR/pids"
[[ -d "$PID_DIR" ]] || die "pid directory not found: $PID_DIR"

declare -a ORDERED_NAMES=(
    actor
    processor
    learner_gpu_memory_guard
    learner
    backfill_policy
    policy
    eval_env
    train_env
)
declare -a PID_FILES=()
declare -a PID_NAMES=()
declare -a PID_VALUES=()
declare -A SEEN_NAMES=()

for name in "${ORDERED_NAMES[@]}"; do
    pid_file="$PID_DIR/${name}.pid"
    if [[ -f "$pid_file" ]]; then
        PID_FILES+=("$pid_file")
        PID_NAMES+=("$name")
        PID_VALUES+=("$(<"$pid_file")")
        SEEN_NAMES["$name"]=1
    fi
done

while IFS= read -r extra_pid_file; do
    extra_name="$(basename "$extra_pid_file" .pid)"
    if [[ -z "${SEEN_NAMES[$extra_name]:-}" ]]; then
        PID_FILES+=("$extra_pid_file")
        PID_NAMES+=("$extra_name")
        PID_VALUES+=("$(<"$extra_pid_file")")
    fi
done < <(find "$PID_DIR" -maxdepth 1 -type f -name '*.pid' | sort)

((${#PID_FILES[@]} > 0)) || die "no pid files found in $PID_DIR"

log_note "output root: $OUTPUT_ROOT"
log_note "pid dir    : $PID_DIR"

for idx in "${!PID_FILES[@]}"; do
    name="${PID_NAMES[$idx]}"
    pid="${PID_VALUES[$idx]}"
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        log_note "skip $name: invalid pid '$pid'"
        continue
    fi
    if (( DRY_RUN )); then
        log_note "[dry-run] would send SIGTERM to $name pid=$pid"
        continue
    fi
    if pid_is_running "$pid"; then
        if ! pid_matches_expected "$pid" "$name"; then
            log_note "skip $name: pid=$pid is running, but command line does not match this launch on the current host"
            continue
        fi
        log_note "sending SIGTERM to $name pid=$pid"
        signal_managed_process "$pid" TERM
    else
        log_note "skip $name: pid=$pid is not running on this host"
    fi
done

if (( DRY_RUN )); then
    log_note "dry run completed; nothing was stopped"
    exit 0
fi

deadline=$(( $(date +%s) + TIMEOUT_SEC ))
while (( $(date +%s) < deadline )); do
    survivors=0
    for pid in "${PID_VALUES[@]}"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && pid_is_running "$pid"; then
            survivors=1
            break
        fi
    done
    (( survivors == 0 )) && break
    sleep 1
done

for idx in "${!PID_FILES[@]}"; do
    name="${PID_NAMES[$idx]}"
    pid="${PID_VALUES[$idx]}"
    pid_file="${PID_FILES[$idx]}"
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        remove_pid_file "$pid_file"
        continue
    fi
    if pid_is_running "$pid"; then
        if ! pid_matches_expected "$pid" "$name"; then
            log_note "skip $name: pid=$pid still exists, but command line does not match this launch on the current host"
            continue
        fi
        log_note "sending SIGKILL to $name pid=$pid"
        signal_managed_process "$pid" KILL
    fi
    if ! pid_is_running "$pid"; then
        remove_pid_file "$pid_file"
    fi
done

log_note "stop sequence finished"
