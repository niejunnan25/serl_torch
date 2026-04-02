#!/usr/bin/env bash
set -euo pipefail

# exp9 formal runs launcher
# Usage:
#   bash examples/libero/configs/exp9/COMMANDS.sh <run> <role>
# Example:
#   bash examples/libero/configs/exp9/COMMANDS.sh fused env
#
# run:
#   - fused
#   - raw
# role:
#   - env
#   - async_eval_env
#   - openpi
#   - learner
#   - actor

ROOT="/vla/users/niejunnan/codebase/serl_torch"
TOOLS_DIR="${ROOT}/examples/libero/tools"
CFG_DIR="${ROOT}/examples/libero/configs/exp9/chunk"
OUT_BASE="${ROOT}/examples/libero/outputs/exp9/chunk"

RUN="${1:-}"
ROLE="${2:-}"

if [[ -z "${RUN}" || -z "${ROLE}" ]]; then
  echo "Usage: bash examples/libero/configs/exp9/COMMANDS.sh <run> <role>"
  echo "run: fused | raw"
  echo "role: env | async_eval_env | openpi | learner | actor"
  exit 1
fi

case "${RUN}" in
  fused)
    RUN_DIR="${OUT_BASE}/fused_async_full"
    CFG="${CFG_DIR}/libero_10_task_6_chunk_state-fused_alpha-01_async_full.yaml"
    ENV_PORT=35490
    EVAL_ENV_PORT=35492
    OPENPI_PORT=35491
    OPENPI_GPU=2
    LEARNER_GPU=1
    ACTOR_GPU=0
    ;;
  raw)
    RUN_DIR="${OUT_BASE}/raw_async_full"
    CFG="${CFG_DIR}/libero_10_task_6_chunk_state-raw_alpha-01_async_full.yaml"
    ENV_PORT=35530
    EVAL_ENV_PORT=35532
    OPENPI_PORT=35531
    OPENPI_GPU=2
    LEARNER_GPU=1
    ACTOR_GPU=0
    ;;
  *)
    echo "Invalid run: ${RUN}. Expected: fused | raw"
    exit 1
    ;;
esac

BOOTSTRAP="${RUN_DIR}/agentlace_bootstrap.pkl"

case "${ROLE}" in
  env)
    mkdir -p "${RUN_DIR}"
    bash "${TOOLS_DIR}/serve_env.sh" \
      --host 127.0.0.1 \
      --port "${ENV_PORT}" \
      2>&1 | tee "${RUN_DIR}/env.log"
    ;;
  async_eval_env)
    mkdir -p "${RUN_DIR}"
    bash "${TOOLS_DIR}/serve_env.sh" \
      --host 127.0.0.1 \
      --port "${EVAL_ENV_PORT}" \
      2>&1 | tee "${RUN_DIR}/async_eval_env.log"
    ;;
  openpi)
    mkdir -p "${RUN_DIR}"
    bash "${TOOLS_DIR}/serve_openpi.sh" \
      --port "${OPENPI_PORT}" \
      --gpu-id "${OPENPI_GPU}" \
      2>&1 | tee "${RUN_DIR}/openpi.log"
    ;;
  learner)
    mkdir -p "${RUN_DIR}/learner"
    bash "${TOOLS_DIR}/run_learner.sh" \
      "${CFG}" \
      --bootstrap "${BOOTSTRAP}" \
      --gpu_id "${LEARNER_GPU}" \
      hydra.run.dir="${RUN_DIR}/learner" \
      2>&1 | tee "${RUN_DIR}/learner_launcher.log"
    ;;
  actor)
    mkdir -p "${RUN_DIR}/actor"
    bash "${TOOLS_DIR}/run_actor.sh" \
      "${CFG}" \
      --bootstrap "${BOOTSTRAP}" \
      --gpu_id "${ACTOR_GPU}" \
      hydra.run.dir="${RUN_DIR}/actor" \
      2>&1 | tee "${RUN_DIR}/actor_launcher.log"
    ;;
  *)
    echo "Invalid role: ${ROLE}. Expected: env | async_eval_env | openpi | learner | actor"
    exit 1
    ;;
esac
