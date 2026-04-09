#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

HOST="127.0.0.1"
PORT="32000"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "  AgiBot Remote Env Server"
echo "=========================================="
echo "  Mode        : optional RPC bridge for env.backend=remote"
echo "  Working dir : $ROOT_DIR"
echo "  Address     : http://${HOST}:${PORT}"
echo "=========================================="

AGIBOT_CONDA_PREFIX_EFFECTIVE="${AGIBOT_CONDA_PREFIX:-${SERL_CONDA_PREFIX:-}}"
AGIBOT_CONDA_ENV_EFFECTIVE="${AGIBOT_CONDA_ENV:-${SERL_CONDA_ENV:-}}"
codex_activate_conda "$AGIBOT_CONDA_PREFIX_EFFECTIVE" "$AGIBOT_CONDA_ENV_EFFECTIVE" "serl_torch"

PYTHON_BIN="$(codex_python_bin "${AGIBOT_PYTHON_BIN:-${SERL_PYTHON_BIN:-python}}")"

exec "$PYTHON_BIN" scripts/services/serve_env.py --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
