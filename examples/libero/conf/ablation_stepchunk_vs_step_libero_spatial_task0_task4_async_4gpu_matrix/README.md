# LIBERO Spatial Task0/Task4 Async Step-vs-Chunk 8-Config Matrix

This folder contains 8 configs for comparing `step` vs `chunk` on `libero_spatial`
for `task_id=0` and `task_id=4`, with both `fused` and `raw` state modes.

Fixed settings in all configs:

- `task.suite_name=libero_spatial`
- `training.async.enabled=true`
- `residual.alpha=0.1`
- `residual.epsilon_gating.enabled=false`
- `residual.observation.state_mode in {fused, raw}`
- `training.warmup.episodes=100`
- `training.online_prefill.enabled=false`

Layout:

- `step/`: task0/task4 x fused/raw (`chunk_step.enabled=false`)
- `chunk/`: task0/task4 x fused/raw (`chunk_step.enabled=true`)

Warmup is collected online at runtime in the current version.

- No pre-collected `online_prefill` manifests are required.
- `step` and `chunk` still use their own replay logic during runtime warmup.

Port allocation:

- task0 step fused: `36010/36011/36012` (`env/openpi/async_eval`)
- task0 chunk fused: `36030/36031/36032`
- task4 step fused: `36050/36051/36052`
- task4 chunk fused: `36070/36071/36072`
- task0 step raw: `36090/36091/36092`
- task0 chunk raw: `36110/36111/36112`
- task4 step raw: `36130/36131/36132`
- task4 chunk raw: `36150/36151/36152`
