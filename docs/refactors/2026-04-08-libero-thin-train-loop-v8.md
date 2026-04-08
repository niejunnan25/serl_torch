# LIBERO Thin Train Loop V8

## Scope

Continuation slimming, Phase 5.

This phase deduplicates actor-loop helper logic by reusing the shared helpers in
[actor_support.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py).

## Main Changes

Updated:

- [actor_loop.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py)

### Deduplicated loop helpers

Removed local re-implementations of:

- train gate resolution
- policy input construction
- chunk-step record construction
- replay size inspection
- external actor flush
- agentlace lag baseline sync
- progress-bar construction

These now reuse the shared helpers from
[actor_support.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py).

### Kept thin local wrappers only where loop-local state sync is needed

The loop still keeps small wrappers for:

- sending agentlace timer stats
- waiting for async learner budget
- train-progress updates
- checkpoint saving

Those wrappers now exist only to sync local loop counters into
`ActorLoopState` before delegating to shared helpers.

### Removed duplicate train progress creation

`initialize_actor_loop_state(...)` already creates the top-level training
progress bar when needed. This phase removes the redundant second creation from
`actor_loop.py`.

## Review

Passed:

- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_loop.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_runtime.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_support.py`
- import smoke test in `serl_torch` conda env for:
  - `serl_launcher.residual.runtime.actor_runtime`
  - `serl_launcher.residual.runtime.actor_loop`
  - `serl_launcher.residual.runtime.actor_support`
- `git diff --check`

Additional signal:

- [actor_loop.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py)
  dropped from about `2467` lines to about `2186` lines in this phase

## Commit Summary

Suggested commit message:

`Deduplicate actor loop helpers through actor support`
