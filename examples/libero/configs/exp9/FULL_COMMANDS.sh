#!/usr/bin/env bash

# exp9: two fused-state formal runs with different manual pacing
# 1) env_steps_per_update_call=6 (YAML: ...fused...yaml)
# 2) env_steps_per_update_call=3 (YAML: ...raw...yaml but state_mode has been changed to fused)
#
# Run each command in a separate terminal/tmux pane.

# ===========================
# fused lag6
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35490 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35492 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35491 \
  --gpu-id 1 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/agentlace_bootstrap.pkl \
  --gpu_id 2 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/agentlace_bootstrap.pkl \
  --gpu_id 1 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/actor_launcher.log

# ===========================
# fused lag3
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35530 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35532 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35531 \
  --gpu-id 3 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/agentlace_bootstrap.pkl \
  --gpu_id 4 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/agentlace_bootstrap.pkl \
  --gpu_id 3 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/actor_launcher.log
