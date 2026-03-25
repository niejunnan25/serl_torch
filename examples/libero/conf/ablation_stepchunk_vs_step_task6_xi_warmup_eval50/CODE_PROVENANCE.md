# Code Provenance: ablation_stepchunk_vs_step_task6_xi_warmup_eval50

## Scope

- Config dir: `examples/libero/conf/ablation_stepchunk_vs_step_task6_xi_warmup_eval50`
- Output dir: `examples/libero/outputs/libero/ablation_stepchunk_vs_step_task6_xi_warmup_eval50`
- Config set: eval50 variant (current directory contains 4 configs)

## Training Run Window

- Run time: `2026-03-25 00:44:37 +0800`

## Code Commit (Inferred at Run Time)

- Commit: `c35783bb85f692ed7e3cada63656c6a1a22c6278` (`c35783b`)
- Commit time: `2026-03-25 00:02:10 +0800`
- Subject: `Merge pull request #4 from niejunnan25/feat/libero-stage1-chunk-step`

## Important Caveat

- This config directory is currently untracked in git status on the machine (`?? examples/libero/conf/ablation_stepchunk_vs_step_task6_xi_warmup_eval50/`).
- So this experiment used local (uncommitted) config files plus code at `HEAD=c35783b`.

## Evidence

- `git reflog --date=iso` around run timestamp points to `HEAD=c35783b` (`pull origin main: Fast-forward`).
- Run metadata exists in:
  - `.../train_residual_sac.log`
  - `.../.hydra/hydra.yaml`
  - `.../.hydra/overrides.yaml`

## Notes

- This output group does not directly log `git sha` inside run artifacts.
- Provenance is reconstructed via timestamp + reflog and is high confidence.
