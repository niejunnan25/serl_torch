#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=========================================="
echo "  LIBERO HDF5 Stats"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Extra args  : $*"
echo "=========================================="

CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    if [[ -n "${STATS_CONDA_PREFIX:-}" ]]; then
        conda activate "$STATS_CONDA_PREFIX"
    elif [[ -n "${STATS_CONDA_ENV:-}" ]]; then
        conda activate "$STATS_CONDA_ENV"
    elif [[ -d "/vla/users/niejunnan/envs/libero" ]]; then
        conda activate "/vla/users/niejunnan/envs/libero"
    elif [[ -d "/vla/users/niejunnan/envs/hf_download" ]]; then
        conda activate "/vla/users/niejunnan/envs/hf_download"
    elif conda env list | awk '{print $1}' | grep -qx "libero"; then
        conda activate libero
    elif conda env list | awk '{print $1}' | grep -qx "hf_download"; then
        conda activate hf_download
    fi
fi

python scripts/compute_hdf5_stats.py "$@"
