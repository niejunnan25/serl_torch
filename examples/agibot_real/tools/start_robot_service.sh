#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ENV_SH="$ROOT_DIR/robot_service/env.sh"
LOCAL_CONF="$ROOT_DIR/robot_service/conf/copilot.pbtxt"
ROBOT_SERVICE_RUNNER="$ROOT_DIR/scripts/services/start_robot_service.py"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/start_robot_service.sh"
    echo
    echo "Starts robot-service using repo-local env/config/files vendored under examples/agibot_real:"
    echo "  $LOCAL_ENV_SH"
    echo "  $LOCAL_CONF"
    echo "  $ROBOT_SERVICE_RUNNER"
    echo
    echo "Useful env vars:"
    echo "  SERL_CONDA_ENV / SERL_CONDA_PREFIX    Python env before launch"
    echo "  AGIBOT_FORWARDER_DIR                  Use an existing extracted forwarder bundle"
    echo "  AGIBOT_FORWARDER_TAR                  Use a local forwarder tarball to prepare repo-local runtime"
    echo "  AGIBOT_FORWARDER_URL                  With prepare_robot_runtime.sh, download forwarder from an artifact URL"
    echo "  AGIBOT_NO_ROS=1                       Forward --no-ros to the repo-local runner"
    echo
    echo "One-time setup helper:"
    echo "  bash tools/prepare_robot_runtime.sh --from-tar /path/to/forwarder_x86_v1.7.0.tar.gz"
    exit 0
fi

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"
PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

if [[ ! -f "$LOCAL_ENV_SH" ]]; then
    echo "ERROR: repo-local robot-service env not found: $LOCAL_ENV_SH"
    exit 1
fi
if [[ ! -f "$LOCAL_CONF" ]]; then
    echo "ERROR: repo-local robot-service config not found: $LOCAL_CONF"
    exit 1
fi
if [[ ! -f "$ROBOT_SERVICE_RUNNER" ]]; then
    echo "ERROR: repo-local robot-service runner not found: $ROBOT_SERVICE_RUNNER"
    exit 1
fi

# shellcheck source=/dev/null
source "$LOCAL_ENV_SH"

USER_ARGS=("$@")
if [[ ${#USER_ARGS[@]} -eq 0 ]]; then
    USER_ARGS=(-s -c "$LOCAL_CONF")
fi

HAS_CONFIG=0
HAS_NO_ROS=0
for arg in "${USER_ARGS[@]}"; do
    case "$arg" in
        -c|--config|--config=*)
            HAS_CONFIG=1
            ;;
        --no-ros)
            HAS_NO_ROS=1
            ;;
    esac
done
if [[ "$HAS_CONFIG" != "1" ]]; then
    USER_ARGS+=(-c "$LOCAL_CONF")
fi
if [[ "${AGIBOT_NO_ROS:-0}" == "1" && "$HAS_NO_ROS" != "1" ]]; then
    USER_ARGS+=(--no-ros)
fi

echo "=========================================="
echo "  AgiBot Robot Service"
echo "=========================================="
echo "  Runner : $ROBOT_SERVICE_RUNNER"
echo "  Python : $PYTHON_BIN"
echo "  Env    : $LOCAL_ENV_SH"
echo "  Config : $LOCAL_CONF"
echo "  Args   : ${USER_ARGS[*]}"
echo "=========================================="

exec "$PYTHON_BIN" "$ROBOT_SERVICE_RUNNER" "${USER_ARGS[@]}"
