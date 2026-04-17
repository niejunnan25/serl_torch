# LIBERO Thin Train Loop V5

## Scope

Second-stage slimming, Phase 2.

This version moves actor-side runtime setup ownership into
`serl_launcher.residual.runtime.actor_setup` and leaves
`actor_runtime.run_residual_actor_loop(...)` responsible for:

- unpacking a prepared runtime session
- keeping loop-local counters and progress state
- executing the rollout/training phases

## Main Changes

### New setup module

Added:

- [actor_setup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py)

This module now owns the large initialization path that previously lived at the
top of `actor_runtime.py`, including:

- residual action / replay / async configuration resolution
- OpenPI client and sample observation bootstrap
- replay / offline / online prefill / warmstart preparation
- async learner and prefetch setup
- checkpoint / profiling / async-eval runtime setup
- `ActorRuntimeContext` population
- initial `ActorLoopState` creation

### Thinner actor runtime entry

Updated:

- [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)

`run_residual_actor_loop(...)` now starts from:

1. `ctx, state = build_actor_runtime_session(...)`
2. local unpack of the prepared runtime context
3. loop-specific helper closures
4. warmup / phase execution

The actor runtime file size dropped from roughly `3388` lines to `2471` lines in
this phase.

## Review

### Static checks

Passed:

- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_runtime.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_setup.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_support.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_warmup.py`
- `git diff --check`

### Import smoke test

Passed in the `serl_torch` conda environment with `PYTHONPATH` pointing at the
local `serl_launcher` checkout:

- `serl_launcher.residual.runtime.actor_setup`
- `serl_launcher.residual.runtime.actor_support`
- `serl_launcher.residual.runtime.actor_warmup`
- `serl_launcher.residual.runtime.actor_runtime`

### Notes

- During extraction, `actor_runtime.py` temporarily kept both the old setup
  block and the new builder call. This was fixed before commit.
- The file still contains several setup-era imports and helper closures. Those
  are left for the next slimming phase rather than being mixed into this commit.

## Commit Summary

Suggested commit message:

`Extract actor runtime setup builder`
