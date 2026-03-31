#!/usr/bin/env bash

# exp4: copied from exp2 (libero_spatial only)
# training.calql_pretrain:
#   enabled: true
#   steps: 200
#   batch_size: 128
#   alpha: 0.1
#   n_actions: 10
#   temperature: 1.0

# libero_spatial -> use run_train_10000.sh
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/chunk/libero_spatial_task_0_chunk_state-fused_alpha-01_async_true.yaml --gpu_id 0 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/chunk/libero_spatial_task_0_chunk_state-raw_alpha-01_async_true.yaml --gpu_id 1 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/step/libero_spatial_task_0_step_state-fused_alpha-01_async_true.yaml --gpu_id 2 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/step/libero_spatial_task_0_step_state-raw_alpha-01_async_true.yaml --gpu_id 3 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/chunk/libero_spatial_task_4_chunk_state-fused_alpha-01_async_true.yaml --gpu_id 4 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/chunk/libero_spatial_task_4_chunk_state-raw_alpha-01_async_true.yaml --gpu_id 5 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/step/libero_spatial_task_4_step_state-fused_alpha-01_async_true.yaml --gpu_id 6 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train_10000.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp4/step/libero_spatial_task_4_step_state-raw_alpha-01_async_true.yaml --gpu_id 7 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp4
