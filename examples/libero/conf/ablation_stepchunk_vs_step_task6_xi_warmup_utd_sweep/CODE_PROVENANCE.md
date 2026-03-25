# Code Provenance: ablation_stepchunk_vs_step_task6_xi_warmup_utd_sweep

## Scope

- Config dir: `examples/libero/conf/ablation_stepchunk_vs_step_task6_xi_warmup_utd_sweep`
- Planned output dir: `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup_utd_sweep`
- Config set: 24-way ablation
  - base 8-way: (`step` vs `stepchunk`) x (`xi10`/`xi50`) x (`nowarmup`/`warmup100ep`)
  - sweep factor: `sac.utd_ratio in {1, 2, 4}`

## Generation Rule

- Source templates: `examples/libero/conf/ablation_stepchunk_vs_step_task6_xi_warmup/*.yaml`
- New file naming: `<original_name>_utd{1|2|4}.yaml`
- Controlled changes only:
  - `sac.utd_ratio` -> `1/2/4`
  - `hydra.run.dir` -> `_utd{1|2|4}` suffix and new output root
  - ports offset for concurrent runs on one machine:
    - utd1: `+0`
    - utd2: `+100`
    - utd4: `+200`
  - affected ports:
    - `openpi.port`
    - `env.remote.port`
    - `training.async_eval.env_port`

## Run Status

- This directory contains config templates only.
- No training run has been recorded under this directory yet.
