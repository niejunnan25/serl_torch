#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/collect_online_prefill.sh <yaml|path/to/config.yaml> [--episodes N] [--output_dir DIR] [extra hydra overrides...]"
    exit 0
fi

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

"$PYTHON_BIN" scripts/data/collect_online_prefill.py "$@"
