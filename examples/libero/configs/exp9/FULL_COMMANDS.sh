#!/usr/bin/env bash

# exp9: fused-state formal runs with different manual pacing
# lag6 / lag3 / lag10 / null
#
# Run each command in a separate terminal/tmux pane.

# ------------------------------------------------------------
# Recommended: one-command Hydra launcher (RLinf-style)
# ------------------------------------------------------------
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag6.yaml

bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag3.yaml

bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag10.yaml

bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_null.yaml

bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag6_unfreeze.yaml

bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag3_unfreeze.yaml

bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag10_unfreeze.yaml

bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/launch_async_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_null_unfreeze.yaml

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
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag6.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/agentlace_bootstrap.pkl \
  --gpu_id 2 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag6.yaml \
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
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag3.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/agentlace_bootstrap.pkl \
  --gpu_id 4 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag3.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/agentlace_bootstrap.pkl \
  --gpu_id 3 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3/actor_launcher.log

# ===========================
# fused lag10
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35690 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35692 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10 && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35691 \
  --gpu-id 7 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag10.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/agentlace_bootstrap.pkl \
  --gpu_id 6 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_lag10.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/agentlace_bootstrap.pkl \
  --gpu_id 7 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10/actor_launcher.log

# ===========================
# fused null (disable bounded-lag throttling; true unbounded async)
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35790 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35792 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35791 \
  --gpu-id 5 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_null.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/agentlace_bootstrap.pkl \
  --gpu_id 6 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full_null.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/agentlace_bootstrap.pkl \
  --gpu_id 5 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null/actor_launcher.log

# ===========================
# fused lag6 unfreeze_resnet (openpi+actor: gpu0, learner: gpu1)
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35890 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35892 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35891 \
  --gpu-id 0 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag6_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 1 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag6_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 0 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag6_unfreeze_resnet/actor_launcher.log

# ===========================
# fused lag3 unfreeze_resnet (openpi+actor: gpu2, learner: gpu3)
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35930 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35932 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35931 \
  --gpu-id 2 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag3_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 3 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag3_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 2 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag3_unfreeze_resnet/actor_launcher.log

# ===========================
# fused lag10 unfreeze_resnet (openpi+actor: gpu4, learner: gpu5)
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35970 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35972 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35971 \
  --gpu-id 4 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag10_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 5 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_lag10_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 4 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_lag10_unfreeze_resnet/actor_launcher.log

# ===========================
# fused null unfreeze_resnet (openpi+actor: gpu6, learner: gpu7)
# ===========================
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 36010 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 36012 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 36011 \
  --gpu-id 6 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_null_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 7 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_async_null_unfreeze.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/agentlace_bootstrap.pkl \
  --gpu_id 6 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full_null_unfreeze_resnet/actor_launcher.log
