# Code Provenance: ablation_stepchunk_vs_step_task6_alpha_warmup

## Scope

- Config dir: `examples/libero/conf/ablation_stepchunk_vs_step_task6_alpha_warmup`
- Output dir: `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_alpha_warmup`
- Config set: 8-way ablation (`step` vs `stepchunk`) x (`xi10`/`xi50`) x (`nowarmup`/`warmup100ep`)

## Training Run Window

- Earliest run: `2026-03-24 21:06:19 +0800`
- Latest run: `2026-03-24 21:06:30 +0800`

## Code Commit (Inferred at Run Time)

- Commit: `2421c3902abb3c6101b9d11abbb1cf981b7e9190` (`2421c39`)
- Commit time: `2026-03-24 20:49:37 +0800`
- Subject: `exp(libero): enable async training for task8 step/chunk-step ablation configs`

## Important Caveat

- Task6 config commit `16f70cdf08e268ad72d19261f112eaa576356370` (`16f70cd`) happened at `2026-03-24 21:08:40 +0800`, which is **later** than this run window.
- Therefore this group likely used local working-tree config content before it was committed.

## Evidence

- `git reflog --date=iso` around run timestamps points to `HEAD=2421c39`.
- Run metadata exists in:
  - `.../train_residual_sac.log`
  - `.../.hydra/hydra.yaml`
  - `.../.hydra/overrides.yaml`

## Notes

- This output group does not directly log `git sha` inside run artifacts.
- Provenance is reconstructed via timestamp + reflog and is medium/high confidence.
