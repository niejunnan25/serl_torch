#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/run_learner_generic.sh <yaml|path/to/config.yaml> --bootstrap path/to/bootstrap.pkl [--gpu_id N] [extra overrides...]"
    echo
    echo "Legacy generic learner wrapper."
    echo "AgiBot learner is now Agentlace-only."
    echo "This alias forwards to tools/run_learner_agentlace.sh."
    exit 0
fi

exec bash "$ROOT_DIR/tools/run_learner_agentlace.sh" "$@"
