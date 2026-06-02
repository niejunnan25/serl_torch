# LIBERO RLT Runbook

This example runs the RL-Token Stage 2 policy optimization path on LIBERO using
the shared `serl_torch` actor/learner, replay, checkpoint, and async-eval
infrastructure.

## Required Artifacts

- A Pi0/OpenPI policy checkpoint compatible with `pi0_libero`.
- A frozen RLT Stage 1 encoder checkpoint.
- A LIBERO remote env server runtime. The launch script starts the env server
  through `examples/libero/tools/serve_env.sh`.
- The OpenPI Python environment used by the VLA feature server.

The checked-in configs include example artifact paths from the development
cluster. Override them with `--pi0-path` and `--rlt-encoder-path` when launching.

## Launch

Start a smoke run without async eval:

```bash
cd /vla/users/niejunnan/codebase/serl_torch-rlt-integration

bash examples/libero_rlt/tools/launch_rlt_training.sh \
  --session libero_rlt_smoke \
  --config-name smoke_rlt \
  --gpu 0 \
  --learner-gpu 1 \
  --pi0-path /path/to/pi0_libero_checkpoint \
  --rlt-encoder-path /path/to/rlt_stage1_encoder.pt \
  --pi0-config pi0_libero \
  -- \
  rlt.warmup_steps=0 \
  training.max_env_steps=1000 \
  training.max_update_steps=1000
```

Start with async eval:

```bash
bash examples/libero_rlt/tools/launch_rlt_training.sh \
  --session libero_rlt_eval_smoke \
  --config-name smoke_rlt \
  --gpu 0 \
  --learner-gpu 1 \
  --with-eval \
  --eval-gpu 2 \
  --env-port 21000 \
  --vla-port 8777 \
  --eval-env-port 21001 \
  --eval-vla-port 8877 \
  --pi0-path /path/to/pi0_libero_checkpoint \
  --rlt-encoder-path /path/to/rlt_stage1_encoder.pt \
  --pi0-config pi0_libero \
  -- \
  rlt.warmup_steps=0 \
  training.max_env_steps=300 \
  training.max_update_steps=300 \
  training.async_eval.enabled=true \
  training.async_eval.every_episodes=1 \
  training.async_eval.episodes=1 \
  training.async_eval.max_env_steps_per_episode=20
```

## Key Config Semantics

- `rlt.chunk_size`: number of actions predicted by the RLT actor.
- `rlt.execute_horizon`: number of predicted actions executed before replanning.
- Replay transitions store `executed_steps` and `discounts =
  gamma ** executed_steps`; the learner uses the stored `discounts` for TD
  bootstrap.
- Terminal transitions skip next-state VLA inference because terminal TD targets
  do not bootstrap.
- The actor and learner share one run directory. The learner owns Hydra metadata;
  the actor is launched with `hydra.output_subdir=null`.
- Async eval writes queue, summary, worker log, and eval checkpoints inside the
  run directory. `training.async_eval.checkpoint.keep <= 0` keeps all eval
  checkpoints.

## Runtime Outputs

- `summary.json`: learner terminal summary with update/env/replay counts.
- `episode_logs.jsonl`: actor episode summaries.
- `actor_timers.jsonl` / `learner_timers.jsonl`: lightweight timing records.
- `checkpoints/`: learner checkpoints.
- `eval_summary.jsonl`: async eval results when enabled.

## Review Scope

This first integration pass intentionally covers LIBERO RLT only. Real-robot
AgiBot/JoyRA code remains outside this review unit and should be staged in a
separate follow-up.
