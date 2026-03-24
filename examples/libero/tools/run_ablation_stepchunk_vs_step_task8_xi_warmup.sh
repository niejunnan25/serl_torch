#!/usr/bin/env bash
set -euo pipefail

ROOT="/vla/users/niejunnan/codebase/serl_torch"
RUN_TRAIN_SH="${ROOT}/examples/libero/tools/run_train.sh"
CONF_DIR="${ROOT}/examples/libero/conf/ablation_stepchunk_vs_step_task8_xi_warmup"

bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_step_xi10_nowarmup.yaml" --gpu_id 0 &
bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_step_xi10_warmup100ep.yaml" --gpu_id 1 &
bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_step_xi50_nowarmup.yaml" --gpu_id 2 &
bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_step_xi50_warmup100ep.yaml" --gpu_id 3 &

bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_stepchunk_xi10_nowarmup.yaml" --gpu_id 4 &
bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_stepchunk_xi10_warmup100ep.yaml" --gpu_id 5 &
bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_stepchunk_xi50_nowarmup.yaml" --gpu_id 6 &
bash "${RUN_TRAIN_SH}" "${CONF_DIR}/train_residual_sac_ablation_stepchunk_xi50_warmup100ep.yaml" --gpu_id 7 &

wait
