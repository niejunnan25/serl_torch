#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

echo "=========================================="
echo "  AgiBot Residual SAC Evaluation"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Config      : conf/eval_residual_fast.yaml"
echo "  Extra args  : $*"
echo "=========================================="

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

"$PYTHON_BIN" scripts/eval/evaluate_checkpoint.py \
    env.backend=remote \
    env.remote.host=127.0.0.1 \
    env.remote.port=32000 \
    openpi.host=localhost \
    openpi.port=30001 \
    "$@"
