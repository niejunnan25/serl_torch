#!/usr/bin/env bash
set -euo pipefail

# exp11 task8 formal runs launcher
# Usage:
#   bash examples/libero/configs/exp11/COMMANDS.sh <run> <role>
#   bash examples/libero/configs/exp11/COMMANDS.sh <role>   # shorthand for run=null
#
# run:
#   - null
#   - null_utdeff1p
#   - null_unfreeze
#   - null_utdeff1p_unfreeze
#   - null_pi05
#   - null_utdeff1p_pi05
#   - null_unfreeze_pi05
#   - null_utdeff1p_unfreeze_pi05
#
# role:
#   - env
#   - async_eval_env
#   - openpi
#   - learner
#   - actor
#   - launch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TOOLS_DIR="${ROOT}/examples/libero/tools"
CFG_DIR="${SCRIPT_DIR}/chunk"
PI0_OUT_BASE="${ROOT}/examples/libero/outputs/exp11/chunk/pi0"
PI05_OUT_BASE="${ROOT}/examples/libero/outputs/exp11/chunk/pi05"
PI05_CKPT="/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero"

RUN="${1:-}"
ROLE="${2:-}"

case "${RUN}" in
  env|async_eval_env|openpi|learner|actor|launch)
    if [[ -z "${ROLE}" ]]; then
      ROLE="${RUN}"
      RUN="null"
    fi
    ;;
esac

if [[ -z "${RUN}" || -z "${ROLE}" ]]; then
  echo "Usage: bash examples/libero/configs/exp11/COMMANDS.sh <run> <role>"
  echo "   or: bash examples/libero/configs/exp11/COMMANDS.sh <role>   # defaults to run=null"
  echo "run: null|null_utdeff1p|null_unfreeze|null_utdeff1p_unfreeze|null_pi05|null_utdeff1p_pi05|null_unfreeze_pi05|null_utdeff1p_unfreeze_pi05"
  echo "role: env | async_eval_env | openpi | learner | actor | launch"
  exit 1
fi

OPENPI_POLICY_CONFIG=""
OPENPI_POLICY_DIR=""

case "${RUN}" in
  null)
    RUN_DIR="${PI0_OUT_BASE}/null"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null.yaml"
    ENV_PORT=36790
    EVAL_ENV_PORT=36792
    OPENPI_PORT=36791
    OPENPI_GPU=0
    LEARNER_GPU=1
    ACTOR_GPU=0
    ;;
  null_utdeff1p)
    RUN_DIR="${PI0_OUT_BASE}/null_utdeff1p"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p.yaml"
    ENV_PORT=36920
    EVAL_ENV_PORT=36922
    OPENPI_PORT=36921
    OPENPI_GPU=2
    LEARNER_GPU=3
    ACTOR_GPU=2
    ;;
  null_unfreeze)
    RUN_DIR="${PI0_OUT_BASE}/null_unfreeze"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null_unfreeze.yaml"
    ENV_PORT=37010
    EVAL_ENV_PORT=37012
    OPENPI_PORT=37011
    OPENPI_GPU=4
    LEARNER_GPU=5
    ACTOR_GPU=4
    ;;
  null_utdeff1p_unfreeze)
    RUN_DIR="${PI0_OUT_BASE}/null_utdeff1p_unfreeze"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze.yaml"
    ENV_PORT=37120
    EVAL_ENV_PORT=37122
    OPENPI_PORT=37121
    OPENPI_GPU=6
    LEARNER_GPU=7
    ACTOR_GPU=6
    ;;
  null_pi05)
    RUN_DIR="${PI05_OUT_BASE}/null"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null_pi05.yaml"
    ENV_PORT=40010
    EVAL_ENV_PORT=40012
    OPENPI_PORT=40011
    OPENPI_GPU=0
    LEARNER_GPU=1
    ACTOR_GPU=0
    OPENPI_POLICY_CONFIG="pi05_libero"
    OPENPI_POLICY_DIR="${PI05_CKPT}"
    ;;
  null_utdeff1p_pi05)
    RUN_DIR="${PI05_OUT_BASE}/null_utdeff1p"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p_pi05.yaml"
    ENV_PORT=40020
    EVAL_ENV_PORT=40022
    OPENPI_PORT=40021
    OPENPI_GPU=2
    LEARNER_GPU=3
    ACTOR_GPU=2
    OPENPI_POLICY_CONFIG="pi05_libero"
    OPENPI_POLICY_DIR="${PI05_CKPT}"
    ;;
  null_unfreeze_pi05)
    RUN_DIR="${PI05_OUT_BASE}/null_unfreeze"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null_unfreeze_pi05.yaml"
    ENV_PORT=40030
    EVAL_ENV_PORT=40032
    OPENPI_PORT=40031
    OPENPI_GPU=4
    LEARNER_GPU=5
    ACTOR_GPU=4
    OPENPI_POLICY_CONFIG="pi05_libero"
    OPENPI_POLICY_DIR="${PI05_CKPT}"
    ;;
  null_utdeff1p_unfreeze_pi05)
    RUN_DIR="${PI05_OUT_BASE}/null_utdeff1p_unfreeze"
    CFG="${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml"
    ENV_PORT=40040
    EVAL_ENV_PORT=40042
    OPENPI_PORT=40041
    OPENPI_GPU=6
    LEARNER_GPU=7
    ACTOR_GPU=6
    OPENPI_POLICY_CONFIG="pi05_libero"
    OPENPI_POLICY_DIR="${PI05_CKPT}"
    ;;
  *)
    echo "Invalid run: ${RUN}"
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
    if [[ -n "${OPENPI_POLICY_CONFIG}" ]]; then
      POLICY_CONFIG="${OPENPI_POLICY_CONFIG}" POLICY_DIR="${OPENPI_POLICY_DIR}" \
        bash "${TOOLS_DIR}/serve_openpi.sh" \
        --port "${OPENPI_PORT}" \
        --gpu-id "${OPENPI_GPU}" \
        2>&1 | tee "${RUN_DIR}/openpi.log"
    else
      bash "${TOOLS_DIR}/serve_openpi.sh" \
        --port "${OPENPI_PORT}" \
        --gpu-id "${OPENPI_GPU}" \
        2>&1 | tee "${RUN_DIR}/openpi.log"
    fi
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
  launch)
    if [[ -n "${OPENPI_POLICY_CONFIG}" ]]; then
      POLICY_CONFIG="${OPENPI_POLICY_CONFIG}" POLICY_DIR="${OPENPI_POLICY_DIR}" \
        bash "${TOOLS_DIR}/launch_async_train.sh" "${CFG}"
    else
      bash "${TOOLS_DIR}/launch_async_train.sh" "${CFG}"
    fi
    ;;
  *)
    echo "Invalid role: ${ROLE}. Expected: env | async_eval_env | openpi | learner | actor | launch"
    exit 1
    ;;
esac
