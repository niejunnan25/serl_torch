# Code Provenance: ablation_stepchunk_vs_step_task8_alpha_warmup

## Scope

- Config dir: `examples/libero/conf/ablation_stepchunk_vs_step_task8_alpha_warmup`
- Output dir: `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task8_alpha_warmup`
- Config set: 8-way ablation (`step` vs `stepchunk`) x (`xi10`/`xi50`) x (`nowarmup`/`warmup100ep`)

## Training Run Window

- Earliest run: `2026-03-24 20:36:05 +0800`
- Latest run: `2026-03-24 20:37:52 +0800`

## Code Commit (Inferred at Run Time)

- Commit: `b14e73ee268a2878f183cebed44950182a0f11a3` (`b14e73e`)
- Commit time: `2026-03-24 19:28:14 +0800`
- Subject: `feat(libero): standardize async-eval seed and consolidate ablation configs`

## Evidence

- `git reflog --date=iso` around run timestamps points to `HEAD=b14e73e`.
- Run metadata exists in:
  - `.../train_residual_sac.log`
  - `.../.hydra/hydra.yaml`
  - `.../.hydra/overrides.yaml`

## Notes

- This output group does not directly log `git sha` inside run artifacts.
- Provenance is reconstructed via timestamp + reflog and is high-confidence.
