#!/usr/bin/env bash

# exp3: copied from exp1 (libero_10 only)
# training.calql_pretrain:
#   enabled: true
#   steps: 200
#   batch_size: 128
#   alpha: 0.1
#   n_actions: 10
#   temperature: 1.0

# libero_10 -> use run_train.sh
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_false.yaml --gpu_id 0 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_true.yaml --gpu_id 1 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_false.yaml --gpu_id 2 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_true.yaml --gpu_id 3 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/step/libero_10_task_6_step_state-fused_alpha-01_async_false.yaml --gpu_id 4 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/step/libero_10_task_6_step_state-fused_alpha-01_async_true.yaml --gpu_id 5 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/step/libero_10_task_6_step_state-raw_alpha-01_async_false.yaml --gpu_id 6 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_train.sh /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp3/step/libero_10_task_6_step_state-raw_alpha-01_async_true.yaml --gpu_id 7 --output-dir /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp3
