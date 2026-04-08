#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "  AgiBot Residual SAC Evaluation"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Config      : conf/eval_residual_fast.yaml"
echo "  Extra args  : $*"
echo "=========================================="

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
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" scripts/eval/evaluate_checkpoint.py \
    env.backend=remote \
    env.remote.host=127.0.0.1 \
    env.remote.port=32000 \
    openpi.host=localhost \
    openpi.port=30001 \
    "$@"
