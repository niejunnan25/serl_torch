#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "  LIBERO Residual SAC Training"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Config      : conf/train_residual_sac.yaml"
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
    echo "ERROR: tools/train.sh requires Python 3; current PYTHON_BIN=${PYTHON_BIN}"
    echo "Hint: set SERL_CONDA_ENV=serl_torch or SERL_CONDA_PREFIX=/abs/env"
    exit 1
fi

HYDRA_OPTIONS=()
HYDRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-name|--config-path|--config-dir|--cfg|--package|--info|--experimental-rerun)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires a value"
                exit 1
            fi
            HYDRA_OPTIONS+=("$1" "$2")
            shift 2
            ;;
        --config-name=*|--config-path=*|--config-dir=*|--cfg=*|--package=*|--info=*|--experimental-rerun=*)
            HYDRA_OPTIONS+=("$1")
            shift
            ;;
        --help|--hydra-help|--version|--resolve|--run|--multirun|--shell-completion)
            HYDRA_OPTIONS+=("$1")
            shift
            ;;
        *)
            HYDRA_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

"$PYTHON_BIN" scripts/train/run_actor.py \
    "${HYDRA_OPTIONS[@]}" \
    env.backend=remote \
    env.remote.host=127.0.0.1 \
    env.remote.port=30000 \
    openpi.host=localhost \
    openpi.port=30001 \
    "${HYDRA_OVERRIDES[@]}"
