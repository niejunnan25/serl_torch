# LIBERO Thin Train Loop Second-Stage Summary

## Branch

- `refactor/libero-thin-train-loop`

## Goal

Second-stage slimming focused on the actor-side training path after the first
round of runtime extraction.

The target was:

- keep `examples/libero/scripts/train_residual_sac.py` as a thin entrypoint
- keep `examples/libero` focused on environment binding
- continue moving runtime orchestration details into `serl_launcher`
- make the actor path read as:
  - build runtime session
  - dispatch into loop execution

## Completed Phases

### Phase 1

Commit:

- `ec0c6ef` `Extract actor warmup phase and shared runtime context`

Main result:

- introduced shared actor runtime context/state helpers
- extracted base-only warmup into [actor_warmup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py)

### Phase 2

Commit:

- `0fdb418` `Extract actor runtime setup builder`

Main result:

- moved actor-side setup ownership into [actor_setup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py)
- reduced [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py) from about `3388` lines to about `2471` lines

### Phase 3

Commit:

- `364f324` `Extract actor phase loop from runtime entrypoint`

Main result:

- moved the large actor phase/episode execution path into
  [actor_loop.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py)
- reduced [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py) to a thin builder + dispatch wrapper

## Final Structure

### Thin entrypoints

- [train_residual_sac.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_sac.py)
- [run_learner.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_learner.py)

### LIBERO environment binding

- [runtime_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py)
- [obs_adapter.py](/home/hello/codebase/serl_torch/examples/libero/runtime/obs_adapter.py)
- [policy_adapter.py](/home/hello/codebase/serl_torch/examples/libero/runtime/policy_adapter.py)

### Residual runtime layers

- [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
  - thin actor runtime entrypoint
- [actor_setup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py)
  - actor-side runtime/session construction
- [actor_loop.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py)
  - actor phase/episode execution loop
- [actor_warmup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py)
  - base-only warmup path
- [actor_support.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py)
  - shared actor runtime state/support helpers
- [agentlace_bridge.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/agentlace_bridge.py)
  - actor/learner communication layer
- [learner_service.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/learner_service.py)
  - learner-side runtime service

## Validation

Across this second stage, each phase was reviewed with:

- `python -m py_compile`
- `git diff --check`
- import smoke tests in the `serl_torch` conda environment when practical

## Result

The actor-side train path is now structured in the intended order:

1. thin script entrypoint
2. LIBERO runtime binding
3. actor runtime session builder
4. actor execution loop
5. agentlace bridge / learner service

This does not make the system artificially small, but it does make it much
easier to reason about than the previous single large training script.

## Remaining Optional Work

Not required for this stage, but reasonable future follow-ups:

- split [actor_loop.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py) further into per-phase or per-step helpers
- continue reducing setup-era unused imports inside extracted modules
- mirror the same style more aggressively on the learner side if another round
  of slimming is wanted later

## Summary Commit Suggestion

Suggested final summary commit message:

`Add second-stage thin train loop refactor summary`
