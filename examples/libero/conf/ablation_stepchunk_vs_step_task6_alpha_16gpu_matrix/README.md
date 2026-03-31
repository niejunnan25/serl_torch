# Task6 Residual RL 16-GPU Matrix

This folder contains 16 standalone Hydra YAML configs:

- `step/`: 8 configs (`chunk_step.enabled=false`)
- `chunk/`: 8 configs (`chunk_step.enabled=true`)

## Factors (2^3 per mode)

- `residual.observation.state_mode`: `fused` / `raw`
- `residual.epsilon_gating.enabled`: `false` / `true`
- `residual.alpha`: `0.1` / `0.5`

## Shared settings in all 16 configs

- `normalization.enabled=false`
- `training.warmup.episodes=0`
- `training.online_prefill.enabled=false`
- `training.async.enabled=false`
- `training.async_eval.enabled=false`
- Unique `env.remote.port`, `openpi.port`, `training.async_eval.env_port`
- Unique `hydra.run.dir`

## Recommended GPU mapping

- `step/*` -> GPUs `0..7`
- `chunk/*` -> GPUs `8..15`

## Run examples

```bash
# one step run
bash examples/libero/tools/run_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/ablation_stepchunk_vs_step_task6_alpha_16gpu_matrix/step/train_residual_sac_task6_step_state-fused_gate-off_alpha-01.yaml \
  --gpu_id 0

# one chunk run
bash examples/libero/tools/run_train.sh \
  /vla/users/niejunnan/codebase/serl_torch/examples/libero/conf/ablation_stepchunk_vs_step_task6_alpha_16gpu_matrix/chunk/train_residual_sac_task6_chunk_state-fused_gate-off_alpha-01.yaml \
  --gpu_id 8
```
