#!/usr/bin/env bash
set -euo pipefail

# exp11: task8 null-pacing runs for pi0 and pi05
# This file intentionally keeps ONLY Hydra one-command launches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LAUNCHER="${ROOT}/examples/libero/tools/launch_async_train.sh"
CFG_DIR="${SCRIPT_DIR}/chunk"
PI05_CKPT="/vla/users/niejunnan/openpi-assets/checkpoints/pi05_libero"

# -------------------------
# pi0
# -------------------------
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null_unfreeze.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze.yaml"

# -------------------------
# pi05
# -------------------------
POLICY_CONFIG=pi05_libero POLICY_DIR="${PI05_CKPT}" \
  bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null_pi05.yaml"
POLICY_CONFIG=pi05_libero POLICY_DIR="${PI05_CKPT}" \
  bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p_pi05.yaml"
POLICY_CONFIG=pi05_libero POLICY_DIR="${PI05_CKPT}" \
  bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null_unfreeze_pi05.yaml"
POLICY_CONFIG=pi05_libero POLICY_DIR="${PI05_CKPT}" \
  bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml"
