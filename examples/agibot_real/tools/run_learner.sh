#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
REPO_PARENT="$(cd "$REPO_ROOT/.." && pwd)"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"
PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/run_learner.sh [hydra overrides...]"
    echo
    echo "Standard AgiBot learner wrapper for scripts/run_residual_training.py."
    echo "Default config: configs/train_residual.yaml"
    echo "Example:"
    echo "  bash tools/run_learner.sh training.max_update_steps=10000"
    exit 0
fi

export PYTHONPATH="$REPO_PARENT:$REPO_ROOT/serl_launcher${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_residual_training.py" \
    runtime.role=learner \
    "$@"
