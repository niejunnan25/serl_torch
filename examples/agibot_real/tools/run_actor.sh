#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

use_legacy=0
for arg in "$@"; do
    case "$arg" in
        --legacy|--bootstrap|--bootstrap=*)
            use_legacy=1
            ;;
    esac
done

if [[ "${1:-}" == "--legacy" ]]; then
    shift
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/run_actor.sh [hydra overrides...]"
    echo
    echo "Default actor wrapper for the canonical AgiBot residual training flow."
    echo "This alias forwards to tools/run_actor_canonical.sh."
    echo
    echo "Legacy Agentlace wrapper is still available:"
    echo "  bash tools/run_actor.sh --legacy <yaml|path/to/config.yaml> --bootstrap /path/to/bootstrap.pkl"
    exit 0
fi

if [[ "$use_legacy" == "1" ]]; then
    exec bash "$ROOT_DIR/tools/run_actor_agentlace.sh" "$@"
fi

exec bash "$ROOT_DIR/tools/run_actor_canonical.sh" "$@"
