# LIBERO Thin Train Loop V1

## Modification Summary

- Added [examples/libero/runtime/runtime_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py) to concentrate LIBERO-specific runtime construction:
  - create env
  - resolve image keys
  - expose `build_policy_input`, `build_step_core`, and profiled `build_step_obs`
- Added [serl_launcher/serl_launcher/residual/runtime/actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py) as the new home for the actor-side residual runtime orchestration.
- Slimmed [examples/libero/scripts/train_residual_sac.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_sac.py) into a thin entrypoint that now only:
  - builds run context
  - sets global seeds
  - builds LIBERO runtime bindings
  - dispatches into `run_residual_actor_loop(...)`
- Updated [examples/libero/runtime/__init__.py](/home/hello/codebase/serl_torch/examples/libero/runtime/__init__.py) to export the new runtime binding API.

## Review Notes

- `py_compile` passed for:
  - [examples/libero/scripts/train_residual_sac.py](/home/hello/codebase/serl_torch/examples/libero/scripts/train_residual_sac.py)
  - [examples/libero/runtime/runtime_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py)
  - [examples/libero/runtime/__init__.py](/home/hello/codebase/serl_torch/examples/libero/runtime/__init__.py)
  - [serl_launcher/serl_launcher/residual/runtime/actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
- The actor runtime module no longer imports LIBERO-specific helpers directly.
- The entry script dropped from roughly 4000 lines to 51 lines.
- I could not run a full runtime import smoke test in the current shell because this environment is missing `gym/gymnasium`; this is an environment dependency gap, not a syntax failure.

## Commit Summary

- Theme: thin the LIBERO actor entrypoint without changing the actor-side training semantics.
- Result:
  - `train_residual_sac.py` became a thin orchestrator
  - LIBERO-specific binding code moved into `examples/libero/runtime`
  - actor-side runtime control moved into `serl_launcher/residual/runtime`
