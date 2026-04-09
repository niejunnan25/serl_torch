#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/eval.sh [hydra overrides...]"
    echo
    echo "This wrapper respects the config-selected env backend."
    echo "Default: conf/eval_residual_fast.yaml uses env.backend=local for direct real-robot eval."
    echo "Optional remote bridge: pass env.backend=remote env.remote.host=... env.remote.port=..."
    exit 0
fi

echo "=========================================="
echo "  AgiBot Residual SAC Evaluation"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Config      : conf/eval_residual_fast.yaml"
echo "  Env backend : from config/overrides"
echo "  Default     : env.backend=local"
echo "  Extra args  : $*"
echo "=========================================="

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

exec "$PYTHON_BIN" scripts/eval/evaluate_checkpoint.py "$@"
