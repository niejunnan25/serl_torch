#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash tools/collect_online_prefill.sh <yaml_file_name.yaml|/abs/path/to/config.yaml> [--episodes N] [--output_dir DIR] [extra hydra overrides...]

Examples:
  bash tools/collect_online_prefill.sh train_residual_sac.yaml --episodes 100
  bash tools/collect_online_prefill.sh /abs/path/to/train.yaml openpi.port=30011 env.remote.port=30010
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    if [[ -n "${PREFILL_CONDA_PREFIX:-}" ]]; then
        conda activate "$PREFILL_CONDA_PREFIX"
    elif [[ -n "${PREFILL_CONDA_ENV:-}" ]]; then
        conda activate "$PREFILL_CONDA_ENV"
    elif [[ -d "/vla/users/niejunnan/envs/libero" ]]; then
        conda activate "/vla/users/niejunnan/envs/libero"
    elif conda env list | awk '{print $1}' | grep -qx "libero"; then
        conda activate libero
    elif [[ -n "${SERL_CONDA_PREFIX:-}" ]]; then
        conda activate "$SERL_CONDA_PREFIX"
    elif [[ -n "${SERL_CONDA_ENV:-}" ]]; then
        conda activate "$SERL_CONDA_ENV"
    elif [[ -d "/vla/miniconda3/envs/serl_torch" ]]; then
        conda activate serl_torch
    fi
fi

PYTHON_BIN="${SERL_PYTHON_BIN:-${PREFILL_PYTHON_BIN:-python}}"
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
    echo "ERROR: collect_online_prefill.sh requires Python 3; current PYTHON_BIN=${PYTHON_BIN}"
    echo "Hint: set PREFILL_CONDA_ENV/PREFILL_CONDA_PREFIX (or reuse SERL_CONDA_ENV)."
    exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import hydra  # noqa: F401
PY
then
    echo "ERROR: hydra is not available in PYTHON_BIN=${PYTHON_BIN}"
    echo "Hint: ensure the LIBERO/SERL training env is activated before collecting prefill."
    exit 1
fi

"$PYTHON_BIN" scripts/collect_online_prefill.py "$@"
