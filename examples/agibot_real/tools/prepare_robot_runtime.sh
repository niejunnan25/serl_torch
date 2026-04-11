#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORWARDER_TAR_NAME="forwarder_x86_v1.7.0.tar.gz"
LOCAL_FORWARDER_TAR="$ROOT_DIR/vendor/a2d_sdk/$FORWARDER_TAR_NAME"
LOCAL_FORWARDER_ROOT="$ROOT_DIR/robot_service/forwarder"

# shellcheck source=examples/agibot_real/tools/common.sh
source "$ROOT_DIR/tools/common.sh"

print_help() {
    echo "Usage: bash tools/prepare_robot_runtime.sh [--from-dir DIR | --from-tar TAR | --from-url URL]"
    echo
    echo "Prepare repo-local AgiBot runtime assets for robot-service startup."
    echo
    echo "Source selection priority:"
    echo "  1. --from-dir / AGIBOT_FORWARDER_DIR"
    echo "  2. --from-tar / AGIBOT_FORWARDER_TAR"
    echo "  3. --from-url / AGIBOT_FORWARDER_URL"
    echo "  4. existing robot_service/forwarder"
    echo "  5. local cache $LOCAL_FORWARDER_TAR"
    echo
    echo "If you plan to start robot-service with --no-ros, the forwarder bundle is optional."
}

download_file() {
    local url="$1"
    local out_path="$2"
    mkdir -p "$(dirname "$out_path")"
    if command -v curl >/dev/null 2>&1; then
        curl -fL "$url" -o "$out_path"
        return 0
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -O "$out_path" "$url"
        return 0
    fi
    echo "ERROR: neither curl nor wget is available for downloading $url" >&2
    return 1
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "$value" ]]; then
        echo "ERROR: $flag requires a non-empty value" >&2
        exit 1
    fi
}

FROM_DIR=""
FROM_TAR=""
FROM_URL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --from-dir)
            require_value "$1" "${2:-}"
            FROM_DIR="${2:-}"
            shift 2
            ;;
        --from-dir=*)
            FROM_DIR="${1#*=}"
            require_value "--from-dir" "$FROM_DIR"
            shift
            ;;
        --from-tar)
            require_value "$1" "${2:-}"
            FROM_TAR="${2:-}"
            shift 2
            ;;
        --from-tar=*)
            FROM_TAR="${1#*=}"
            require_value "--from-tar" "$FROM_TAR"
            shift
            ;;
        --from-url)
            require_value "$1" "${2:-}"
            FROM_URL="${2:-}"
            shift 2
            ;;
        --from-url=*)
            FROM_URL="${1#*=}"
            require_value "--from-url" "$FROM_URL"
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            print_help >&2
            exit 1
            ;;
    esac
done

SOURCE_COUNT=0
[[ -n "$FROM_DIR" ]] && SOURCE_COUNT=$((SOURCE_COUNT + 1))
[[ -n "$FROM_TAR" ]] && SOURCE_COUNT=$((SOURCE_COUNT + 1))
[[ -n "$FROM_URL" ]] && SOURCE_COUNT=$((SOURCE_COUNT + 1))
if [[ "$SOURCE_COUNT" -gt 1 ]]; then
    echo "ERROR: choose only one of --from-dir, --from-tar, or --from-url" >&2
    exit 1
fi

codex_activate_conda "${SERL_CONDA_PREFIX:-}" "${SERL_CONDA_ENV:-}" "serl_torch"
PYTHON_BIN="$(codex_python_bin "${SERL_PYTHON_BIN:-python}")"

if [[ -n "$FROM_DIR" ]]; then
    export AGIBOT_FORWARDER_DIR="$FROM_DIR"
elif [[ -n "$FROM_TAR" ]]; then
    export AGIBOT_FORWARDER_TAR="$FROM_TAR"
elif [[ -n "$FROM_URL" ]]; then
    echo "Downloading forwarder bundle to $LOCAL_FORWARDER_TAR"
    download_file "$FROM_URL" "$LOCAL_FORWARDER_TAR"
    export AGIBOT_FORWARDER_TAR="$LOCAL_FORWARDER_TAR"
elif [[ -n "${AGIBOT_FORWARDER_DIR:-}" || -n "${AGIBOT_FORWARDER_TAR:-}" || -n "${AGIBOT_FORWARDER_URL:-}" ]]; then
    if [[ -n "${AGIBOT_FORWARDER_URL:-}" && -z "${AGIBOT_FORWARDER_TAR:-}" && -z "${AGIBOT_FORWARDER_DIR:-}" ]]; then
        echo "Downloading forwarder bundle to $LOCAL_FORWARDER_TAR"
        download_file "${AGIBOT_FORWARDER_URL}" "$LOCAL_FORWARDER_TAR"
        export AGIBOT_FORWARDER_TAR="$LOCAL_FORWARDER_TAR"
    fi
elif [[ -d "$LOCAL_FORWARDER_ROOT/app/bin" ]]; then
    :
elif [[ -f "$LOCAL_FORWARDER_TAR" ]]; then
    export AGIBOT_FORWARDER_TAR="$LOCAL_FORWARDER_TAR"
fi

export ROOT_DIR
"$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

root_dir = Path(os.environ["ROOT_DIR"]).resolve()
repo_root = root_dir.parents[1]
repo_parent = repo_root.parent
serl_launcher_root = repo_root / "serl_launcher"
for path in (repo_parent, serl_launcher_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from serl_torch.examples.agibot_real.robot.sdk_bootstrap import ensure_repo_local_a2d_sdk
from serl_torch.examples.agibot_real.robot.sdk_bootstrap import ensure_repo_local_forwarder

site_dir = ensure_repo_local_a2d_sdk()
print(f"SDK ready: {site_dir}")

if os.environ.get("AGIBOT_NO_ROS", "0") == "1":
    print("Skipping forwarder preparation because AGIBOT_NO_ROS=1")
else:
    forwarder_root = ensure_repo_local_forwarder()
    print(f"Forwarder ready: {forwarder_root}")
PY
