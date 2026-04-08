# SERL Launcher Training Phase 3C

## Goal

Finish the package cleanup by moving the remaining async and agentlace runtime
helpers out of `serl_launcher.residual.runtime` and into `serl_launcher.training`.

## New Modules

- `serl_launcher/serl_launcher/training/async_runtime/agentlace.py`
  - asynchronous learner worker helpers
  - replay prefetch helpers
  - agentlace learner service
- `serl_launcher/serl_launcher/training/async_runtime/bridge.py`
  - actor-side agentlace bridge helpers
  - bounded-lag coordination helpers

## Removed Modules

- `serl_launcher/serl_launcher/residual/runtime/async_learning.py`
- `serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py`

## Important Review Fix

I initially tried `serl_launcher/training/async/`, but that path produced an
actual Python syntax problem because `async` is a reserved keyword in import
statements. I corrected the package name to:

- `serl_launcher.training.async_runtime`

before committing this phase.

## Import Updates

Updated actor and learner orchestration to import async helpers from:

- `serl_launcher.training.async_runtime.agentlace`
- `serl_launcher.training.async_runtime.bridge`

## Review

I checked:

- `python -m py_compile` on the moved async modules and updated call sites
- `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`
- `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- `git diff --check`
- `rg` to confirm there are no remaining Python imports of
  `serl_launcher.residual.runtime.agentlace_bridge` or
  `serl_launcher.residual.runtime.async_learning`

## Outcome

After this step:

- `serl_launcher.training` now owns:
  - checkpointing
  - profiling
  - replay batch preparation
  - seeding
  - loop scheduling helpers
  - telemetry helpers
  - async agentlace infrastructure
- `serl_launcher.residual.runtime` is reduced to an empty compatibility shell
  with only `__init__.py`

This completes the structural package move planned for Phase 3.
