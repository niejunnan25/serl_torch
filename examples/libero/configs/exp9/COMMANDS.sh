#!/usr/bin/env bash

# exp9: quick profiling-only async chunk runs for learner bottleneck diagnosis.
# Notes:
# - async_eval is disabled in these configs, so no async-eval env server is needed.
# - checkpointing is disabled to reduce IO noise in profiling traces.

# fused
mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35290 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35291 \
  --gpu-id 2 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_profile.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/agentlace_bootstrap.pkl \
  --gpu_id 1 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_profile.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/agentlace_bootstrap.pkl \
  --gpu_id 0 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/fused_async_profile/actor_launcher.log

# raw
mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35330 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35331 \
  --gpu-id 2 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_profile.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/agentlace_bootstrap.pkl \
  --gpu_id 1 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/learner \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_profile.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/agentlace_bootstrap.pkl \
  --gpu_id 0 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/actor \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/outputs/exp9/chunk/raw_async_profile/actor_launcher.log
