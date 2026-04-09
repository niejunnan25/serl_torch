#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/collect_online_prefill.sh <yaml|path/to/config.yaml> [--episodes N] [--output_dir DIR] [extra hydra overrides...]"
    echo
    echo "This wrapper respects the config-selected env backend."
    echo "Recommended real-robot path: env.backend=local in the training config."
    echo "Optional remote bridge: pass env.backend=remote env.remote.host=... env.remote.port=..."
    exit 0
fi

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

echo "=========================================="
echo "  AgiBot Online Prefill Collection"
echo "=========================================="
echo "  Working dir : $ROOT_DIR"
echo "  Env backend : from config/overrides"
echo "  Default     : training config uses env.backend=local"
echo "  Extra args  : $*"
echo "=========================================="

exec "$PYTHON_BIN" scripts/data/collect_online_prefill.py "$@"
