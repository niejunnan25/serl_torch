# SERL Launcher Training Phase 3A

## Goal

Split the old mixed `serl_launcher.residual.runtime.config_utils` module into
clearer homes:

- training seeding helpers
- agent construction helpers
- residual-train-specific config helpers

## New Modules

- `serl_launcher/serl_launcher/training/seeding.py`
  - `set_global_seeds(...)`
- `serl_launcher/serl_launcher/agents/continuous/drq_config.py`
  - `create_drq_agent_from_cfg(...)`
  - private optimizer and mixed-precision config helpers
- `serl_launcher/serl_launcher/residual/train/config.py`
  - residual observation-state mode helpers
  - residual action-mask/control-index helpers
  - residual action-transform helper
  - probing-step sampling helper

## Removed Module

- `serl_launcher/serl_launcher/residual/runtime/config_utils.py`

## Import Updates

Updated the actor, learner, eval, and online materialization paths to consume
the new homes:

- `training.seeding`
- `agents.continuous.drq_config`
- `residual.train.config`

## Review

I checked:

- `python -m py_compile` on the new modules and their main call sites
- `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`
- `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- `conda run -n serl_torch python examples/libero/scripts/materialize_residual_training_online.py --help`
- `git diff --check`
- `rg` to confirm there are no remaining Python imports of
  `serl_launcher.residual.runtime.config_utils`

## Outcome

The old config utility bucket is now split along the intended package
boundaries:

- training-wide seeding
- agent builders close to `agents/`
- residual-train-specific config close to `residual/train/`
