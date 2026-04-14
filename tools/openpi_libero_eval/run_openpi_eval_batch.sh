#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="127.0.0.1"
PORT=""
OUTPUT_ROOT=""
STRATEGY_NAME=""
NUM_TRIALS_PER_TASK="50"
OPENPI_ROOT=""
LIBERO_ROOT=""
LIBERO_CONFIG_DIR=""
LIBERO_DATASETS_ROOT=""
SAVE_VIDEOS="false"
SUITES=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --strategy-name)
            STRATEGY_NAME="$2"
            shift 2
            ;;
        --num-trials-per-task)
            NUM_TRIALS_PER_TASK="$2"
            shift 2
            ;;
        --openpi-root)
            OPENPI_ROOT="$2"
            shift 2
            ;;
        --libero-root)
            LIBERO_ROOT="$2"
            shift 2
            ;;
        --libero-config-dir)
            LIBERO_CONFIG_DIR="$2"
            shift 2
            ;;
        --libero-datasets-root)
            LIBERO_DATASETS_ROOT="$2"
            shift 2
            ;;
        --save-videos)
            SAVE_VIDEOS="true"
            shift
            ;;
        --no-save-videos)
            SAVE_VIDEOS="false"
            shift
            ;;
        --suite)
            SUITES+=("$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$PORT" || -z "$OUTPUT_ROOT" || -z "$STRATEGY_NAME" ]]; then
    echo "Usage: bash run_openpi_eval_batch.sh --port PORT --output-root DIR --strategy-name NAME [--host HOST] [--num-trials-per-task 50]"
    exit 1
fi

if [[ ${#SUITES[@]} -eq 0 ]]; then
    SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
fi

mkdir -p "$OUTPUT_ROOT"
STATUS_FILE="$OUTPUT_ROOT/batch_status.txt"
echo "starting strategy=$STRATEGY_NAME port=$PORT output_root=$OUTPUT_ROOT" | tee "$STATUS_FILE"

for suite in "${SUITES[@]}"; do
    suite_output="$OUTPUT_ROOT/$suite"
    echo "running suite=$suite output=$suite_output" | tee "$STATUS_FILE"
    PY_ARGS=(
        --host "$HOST"
        --port "$PORT"
        --task-suite-name "$suite"
        --num-trials-per-task "$NUM_TRIALS_PER_TASK"
        --output-root "$suite_output"
        --strategy-name "$STRATEGY_NAME"
    )
    if [[ "$SAVE_VIDEOS" == "true" ]]; then
        PY_ARGS+=(--save-videos)
    else
        PY_ARGS+=(--no-save-videos)
    fi
    if [[ -n "$OPENPI_ROOT" ]]; then
        PY_ARGS+=(--openpi-root "$OPENPI_ROOT")
    fi
    if [[ -n "$LIBERO_ROOT" ]]; then
        PY_ARGS+=(--libero-root "$LIBERO_ROOT")
    fi
    if [[ -n "$LIBERO_CONFIG_DIR" ]]; then
        PY_ARGS+=(--libero-config-dir "$LIBERO_CONFIG_DIR")
    fi
    if [[ -n "$LIBERO_DATASETS_ROOT" ]]; then
        PY_ARGS+=(--libero-datasets-root "$LIBERO_DATASETS_ROOT")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        PY_ARGS+=("${EXTRA_ARGS[@]}")
    fi

    python "$TOOLS_DIR/evaluate_openpi_libero_suite.py" "${PY_ARGS[@]}"
    echo "completed suite=$suite" | tee "$STATUS_FILE"
done

echo "completed strategy=$STRATEGY_NAME" | tee "$STATUS_FILE"
