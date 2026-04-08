# LIBERO Thin Train Loop Summary

## Goal

Make the LIBERO training entrypoints thin without weakening the current `exp11` training system:

- keep `examples/libero` responsible for environment-specific bindings
- move the heavy training runtime into `serl_launcher/residual/runtime`
- make actor and learner entrypoints look more like orchestration scripts

## What Changed

### V1

- Added [examples/libero/runtime/runtime_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py)
- Added [serl_launcher/serl_launcher/residual/runtime/actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
- Slimmed [examples/libero/scripts/train_residual_sac.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/scripts/train_residual_sac.py) into a thin entrypoint

Commit:
- `dc502f0` `Thin LIBERO train entrypoint with runtime bindings`

### V2

- Added [serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py)
- Moved actor-side bootstrap / agentlace learner creation / timer stats / bounded-lag bookkeeping into the bridge module

Commit:
- `7a334ba` `Extract actor-side agentlace bridge from runtime`

### V3

- Added [serl_launcher/serl_launcher/residual/runtime/learner_service.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/learner_service.py)
- Slimmed [examples/libero/scripts/run_learner.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/scripts/run_learner.py) into a thin entrypoint

Commit:
- `765e5ab` `Thin LIBERO learner entrypoint with learner service`

## End State

- [examples/libero/scripts/train_residual_sac.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/scripts/train_residual_sac.py): 51 lines
- [examples/libero/scripts/run_learner.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/scripts/run_learner.py): 48 lines
- [serl_launcher/serl_launcher/residual/runtime/actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py): 3781 lines
- [serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py): 414 lines
- [serl_launcher/serl_launcher/residual/runtime/learner_service.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/learner_service.py): 714 lines

The most important structural result is:

- actor entrypoint is thin
- learner entrypoint is thin
- LIBERO bindings are explicit
- agentlace communication is separated from rollout logic

## Review Result

- `py_compile` passed for all newly added / refactored entrypoint-runtime files
- worktree was clean after the final verification

## Remaining Optional Work

This branch finishes the V1-V3 plan, but there is still optional cleanup if desired:

- further split [actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py) into:
  - warmup collection
  - train phase loop
  - checkpoint / async eval orchestration
- extract a smaller shared runtime context object
- reduce duplication between chunk-step and step-mode rollout branches

These are optimization / readability improvements, not blockers for the current architecture.
