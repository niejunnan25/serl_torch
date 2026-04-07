# LIBERO Thin Train Loop V7

## Scope

Continuation slimming, Phase 4.

This phase is a stabilization pass after the actor loop extraction.

## Main Change

Updated:

- [actor_loop.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py)

Removed a duplicated session-construction block that had been accidentally left
inside `run_actor_loop(...)` after the previous extraction.

Before this fix, the loop body still contained:

- an internal `build_actor_runtime_session(...)` call
- a second full unpack of `ctx/state`

That meant the extracted loop still tried to rebuild runtime state instead of
strictly consuming the `ctx/state` passed in by
[actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py).

After this fix:

- `actor_runtime.py` is the only place that builds the actor runtime session
- `actor_loop.py` only executes the passed-in runtime session

## Review

Passed:

- `grep` check confirming `build_actor_runtime_session(...)` is only referenced in
  [actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_runtime.py`
- `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_loop.py`
- `python -m py_compile examples/libero/scripts/train_residual_sac.py`
- import smoke test in `serl_torch` conda env for:
  - `serl_launcher.residual.runtime.actor_runtime`
  - `serl_launcher.residual.runtime.actor_loop`
- `git diff --check`

## Commit Summary

Suggested commit message:

`Fix actor loop session ownership after extraction`
