# LIBERO Thin Train Loop V4

## Modification Summary

- Added [actor_support.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py) to introduce a practical actor-side context/state layer:
  - `ActorRuntimeContext`
  - `ActorLoopState`
  - shared helpers for:
    - checkpoint saving
    - progress bar updates
    - base-policy input construction
    - chunk-step warmup record building
    - async learner / sync prefetch startup
- Added [actor_warmup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py) to hold the base-only warmup phase.
- Slimmed [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py) by:
  - replacing the inlined warmup phase with `run_base_only_warmup(ctx, state)`
  - replacing the duplicated async-start / sync-prefetch-start blocks with `ensure_training_runtime_started(ctx)`
  - creating a shared `ctx/state` bridge for the next extraction stages

## Review Notes

- `py_compile` passed for:
  - [actor_support.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py)
  - [actor_warmup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py)
  - [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
  - [train_residual_sac.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_sac.py)
- I checked that:
  - warmup now runs through `run_base_only_warmup(...)`
  - async learner / sync prefetch startup now runs through one shared helper instead of two duplicated blocks
- Current line counts after this phase:
  - [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py): `3388`
  - [actor_support.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py): `394`
  - [actor_warmup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py): `465`

## Commit Summary

- Theme: carve out the warmup slice and introduce the shared actor-side runtime context needed for the next thinning steps.
- Result:
  - warmup is no longer embedded directly inside the main actor runtime function
  - repeated async startup logic is centralized
  - the actor runtime now has an explicit staging point for future phase extraction
