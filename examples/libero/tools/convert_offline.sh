#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_OPENPI=false
OPENPI_HOST="localhost"
OPENPI_PORT="30001"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openpi)
            USE_OPENPI=true
            shift
            ;;
        --openpi_host)
            OPENPI_HOST="$2"
            USE_OPENPI=true
            shift 2
            ;;
        --openpi_port)
            OPENPI_PORT="$2"
            USE_OPENPI=true
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "  LIBERO HDF5 -> Offline PKL"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  OpenPI      : $USE_OPENPI"
echo "=========================================="

CONDA_SH="/vla/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    if [[ -n "${CONVERT_CONDA_PREFIX:-}" ]]; then
        conda activate "$CONVERT_CONDA_PREFIX"
    elif [[ -n "${CONVERT_CONDA_ENV:-}" ]]; then
        conda activate "$CONVERT_CONDA_ENV"
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

CMD=(python scripts/convert_hdf5_to_offline.py)
if [[ "$USE_OPENPI" == "true" ]]; then
    CMD+=(--openpi_host "$OPENPI_HOST" --openpi_port "$OPENPI_PORT")
fi
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
