#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/eval.sh [hydra overrides...]"
    echo
    echo "This wrapper runs the local AgiBot env directly."
    echo "conf/eval_residual_fast.yaml already matches the local real-robot workflow."
    exit 0
fi

echo "=========================================="
echo "  AgiBot Residual SAC Evaluation"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Config      : conf/eval_residual_fast.yaml"
echo "  Env mode    : local AgiBot env"
echo "  Extra args  : $*"
echo "=========================================="

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

exec "$PYTHON_BIN" scripts/eval/evaluate_checkpoint.py "$@"
