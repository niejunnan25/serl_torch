#!/usr/bin/env bash

# exp9: full async chunk training runs with async eval enabled.
# Notes:
# - These commands also enable runtime profiling for learner timings.
# - Profiling is surfaced in learner_launcher.log and learner/profiling_logs.jsonl.
# - We disable warmup/online prefill here so timing reflects learner update cost
#   rather than a long startup collection phase or a large one-time RPC flood.
# - Bounded-lag polling is relaxed to reduce request-channel saturation.
# - Each run needs two env servers: one for training, one for async eval.

# fused
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35490 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35492 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35491 \
  --gpu-id 0 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/agentlace_bootstrap.pkl \
  --gpu_id 1 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/learner \
  training.profiling.enabled=true \
  training.profiling.log_period_steps=100 \
  training.warmup.episodes=0 \
  training.online_prefill.enabled=false \
  training.async.bounded_lag.poll_sec=0.2 \
  training.async.bounded_lag.sync_on_wait=false \
  training.async.bounded_lag.max_update_lag_calls=10 \
  training.async.stats_period_steps=500 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-fused_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/agentlace_bootstrap.pkl \
  --gpu_id 0 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/actor \
  training.profiling.enabled=true \
  training.profiling.log_period_steps=100 \
  training.warmup.episodes=0 \
  training.online_prefill.enabled=false \
  training.async.bounded_lag.poll_sec=0.2 \
  training.async.bounded_lag.sync_on_wait=false \
  training.async.bounded_lag.max_update_lag_calls=10 \
  training.async.stats_period_steps=500 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/fused_async_full/actor_launcher.log

# raw
mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35530 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_env.sh \
  --host 127.0.0.1 \
  --port 35532 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/async_eval_env.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/serve_openpi.sh \
  --port 35531 \
  --gpu-id 2 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/openpi.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/learner && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_learner.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/agentlace_bootstrap.pkl \
  --gpu_id 1 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/learner \
  training.profiling.enabled=true \
  training.profiling.log_period_steps=100 \
  training.warmup.episodes=0 \
  training.online_prefill.enabled=false \
  training.async.bounded_lag.poll_sec=0.2 \
  training.async.bounded_lag.sync_on_wait=false \
  training.async.bounded_lag.max_update_lag_calls=10 \
  training.async.stats_period_steps=500 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/learner_launcher.log

mkdir -p /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/actor && \
bash /vla/users/niejunnan/codebase/serl_torch/examples/libero/tools/run_actor.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/configs/exp9/chunk/libero_10_task_6_chunk_state-raw_alpha-01_async_full.yaml \
  --bootstrap /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/agentlace_bootstrap.pkl \
  --gpu_id 0 \
  hydra.run.dir=/vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/actor \
  training.profiling.enabled=true \
  training.profiling.log_period_steps=100 \
  training.warmup.episodes=0 \
  training.online_prefill.enabled=false \
  training.async.bounded_lag.poll_sec=0.2 \
  training.async.bounded_lag.sync_on_wait=false \
  training.async.bounded_lag.max_update_lag_calls=10 \
  training.async.stats_period_steps=500 \
  2>&1 | tee /vla/users/niejunnan/codebase/serl_torch/examples/libero/outputs/exp9/chunk/raw_async_full/actor_launcher.log
