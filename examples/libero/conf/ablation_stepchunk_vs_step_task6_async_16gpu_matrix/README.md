# Task6 Async-vs-Sync 16-GPU Matrix

This matrix is purpose-built to compare `training.async.enabled=false` (sync) vs `true` (async), while fixing `alpha=0.1`.

File naming is explicit:

- `..._async_false.yaml` => `training.async.enabled: false`
- `..._async_true.yaml` => `training.async.enabled: true`

## Layout

- `step/`: 8 configs (`chunk_step.enabled=false`)
- `chunk/`: 8 configs (`chunk_step.enabled=true`)

## Factors (2^3 per mode)

- `residual.observation.state_mode`: `fused` / `raw`
- `residual.epsilon_gating.enabled`: `false` / `true`
- `training.async.enabled`: `false` (sync) / `true` (async)

`residual.alpha` is fixed to `0.1` in all 16 configs.

## Shared controls

- `normalization.enabled=false`
- `training.warmup.episodes=100`
- `training.online_prefill.enabled=false`
- `training.async_eval.enabled=false`
- unique `env.remote.port` / `openpi.port` / `training.async_eval.env_port`
- unique `hydra.run.dir`

Warmup note:

- With `online_prefill.enabled=false`, warmup is runtime base-only collection.
- No pre-collected prefill PKL/manifest is loaded.

## Suggested two-machine mapping

- Machine A: `step/*` on GPUs `0..7`
- Machine B: `chunk/*` on GPUs `0..7`
