# LIBERO Thin Train Loop V11

## Goal

Introduce a shared `ResidualRuntimeBindings` protocol so the residual runtime depends on an explicit environment-binding contract instead of ad hoc `Any`-typed LIBERO structures.

## Changes

- Added [serl_launcher/serl_launcher/residual/runtime/bindings.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/bindings.py) with a new `ResidualRuntimeBindings` protocol.
- The protocol defines the minimal runtime-facing environment contract:
  - attributes:
    - `env`
    - `image_keys`
    - `normalizer`
    - `obs_cache`
    - `task_key`
    - `data_config`
  - methods:
    - `build_policy_input(...)`
    - `build_step_core(...)`
    - `build_step_obs(...)`
    - `build_step_obs_profiled(...)`
- Updated [examples/libero/runtime/runtime_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py) so `LiberoRuntimeBindings` explicitly implements `ResidualRuntimeBindings`.
- Updated actor-side runtime typing to consume the protocol rather than `Any`:
  - [serl_launcher/serl_launcher/residual/runtime/actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
  - [serl_launcher/serl_launcher/residual/runtime/actor_setup.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py)
  - [serl_launcher/serl_launcher/residual/runtime/actor_support.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py)

## Why This Helps

- The runtime now has a stable, documented contract for “what an environment binding must provide.”
- Future environments such as `franka_real` can target a concrete protocol instead of reverse-engineering the LIBERO implementation.
- This keeps the architectural direction aligned with the original goal:
  - `examples/<env>` provides environment-specific bindings
  - `serl_launcher/residual/runtime` consumes a shared runtime contract

## Scope Notes

- This phase is intentionally small and mostly type- and boundary-oriented.
- It does **not** change launcher/tooling.
- It does **not** yet rework learner-side code to depend on the new protocol everywhere.
- It does **not** yet add a second environment implementation.

## Self Review

- Ran `python -m py_compile` on:
  - [bindings.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/bindings.py)
  - [actor_runtime.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py)
  - [actor_setup.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py)
  - [actor_support.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py)
  - [runtime_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py)
- Ran `git diff --check`.
- Verified imports with:
  - `from serl_launcher.residual.runtime.bindings import ResidualRuntimeBindings`
  - `from serl_torch.examples.libero.runtime.runtime_bindings import LiberoRuntimeBindings`
  - `from serl_torch.examples.libero.runtime import build_libero_runtime_bindings`

## Commit Summary

- Define a shared residual runtime bindings protocol
- Make LIBERO bindings explicitly satisfy that protocol
- Replace actor-side `bindings: Any` typing with the shared runtime contract
