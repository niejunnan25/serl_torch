# LIBERO Thin Train Loop V6

## Scope

Second-stage slimming, Phase 3.

This phase moves the actor-side phase and episode execution loop out of
`actor_runtime.py` into a dedicated loop module so that the runtime entrypoint
becomes a real builder + dispatch wrapper.

## Main Changes

### New loop module

Added:

- [actor_loop.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py)

This module now owns the large actor-side execution path, including:

- warmup handoff after `run_base_only_warmup(...)`
- phase / episode / step rollout loops
- base policy querying and residual action composition
- replay insertion and async learner pacing
- step / episode logging
- async eval trigger and summary writing
- final actor-side shutdown within the loop implementation

### Thin runtime entrypoint

Updated:

- [actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)

`run_residual_actor_loop(...)` is now reduced to:

1. build actor runtime session
2. dispatch into `run_actor_loop(ctx, state)`

File size changed from roughly `2471` lines to `28` lines in this phase.

## Review

### Static checks

Passed:

- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_runtime.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_loop.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_setup.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_support.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_warmup.py`
- `python -m py_compile examples/libero/scripts/train_residual_sac.py`
- `git diff --check`

### Import smoke test

Passed in the `serl_torch` conda environment with local `serl_launcher` on
`PYTHONPATH`:

- `serl_launcher.residual.runtime.actor_runtime`
- `serl_launcher.residual.runtime.actor_loop`

### Notes

- This is primarily a structural move. The phase loop implementation remains
  large, but it is now isolated from the runtime entrypoint and session builder.
- The next slimming opportunity, if desired later, is to split
  `actor_loop.py` into smaller per-phase or per-step helpers.

## Commit Summary

Suggested commit message:

`Extract actor phase loop from runtime entrypoint`
