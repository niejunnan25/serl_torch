#!/usr/bin/env bash
set -euo pipefail

# exp9: fused-state formal runs
# This file intentionally keeps ONLY Hydra one-command launches.
# Rationale: avoid parallel maintenance of legacy 5-process manual commands and
# prevent output-path/log-layout divergence across workflows.

LAUNCHER="/vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh"
CFG_DIR="/vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk"

# -------------------------
# Frozen ResNet
# -------------------------
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag6.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag3.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag10.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_alpha-01_async_full_null.yaml"

# -------------------------
# Unfrozen ResNet
# -------------------------
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_async_lag6_unfreeze.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_async_lag3_unfreeze.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_async_lag10_unfreeze.yaml"
bash "${LAUNCHER}" "${CFG_DIR}/libero_10_task_6_chunk_state-fused_async_null_unfreeze.yaml"
