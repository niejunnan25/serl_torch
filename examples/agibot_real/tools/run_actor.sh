#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/run_actor.sh [hydra overrides...]"
    echo
    echo "Default actor wrapper for the canonical AgiBot residual training flow."
    echo "This alias forwards to tools/run_actor_canonical.sh."
    exit 0
fi

exec bash "$ROOT_DIR/tools/run_actor_canonical.sh" "$@"
