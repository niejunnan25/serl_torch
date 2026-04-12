# LIBERO Thin Train Loop V12

## Goal

Make actor-side runtime execution depend on `ResidualRuntimeBindings` at access time, instead of copying binding-derived environment fields into `ActorRuntimeContext`.

## Changes

- Updated [serl_launcher/serl_launcher/residual/runtime/actor_support.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py) with protocol-backed helper accessors:
  - `runtime_image_keys(...)`
  - `runtime_obs_cache(...)`
  - `runtime_task_key(...)`
  - `runtime_data_config(...)`
  - `clear_obs_cache(...)`
  - `build_step_core(...)`
  - `build_step_obs_profiled(...)`
- These helpers now route actor hot paths through `ctx.bindings` instead of duplicated values stored in `ctx.values`.
- Updated [serl_launcher/serl_launcher/residual/runtime/actor_setup.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py) so `ActorRuntimeContext` no longer caches these binding-derived fields:
  - `image_keys`
  - `obs_cache`
  - `task_key`
  - `data_config`
  - `build_residual_step_obs_profiled`
  - `build_residual_step_core`
- Updated [serl_launcher/serl_launcher/residual/runtime/actor_warmup.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py) to use the new helper layer for:
  - observation-cache reset
  - profiled residual observation assembly
- Updated [serl_launcher/serl_launcher/residual/runtime/actor_loop.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py) so rollout execution also uses the helper layer instead of copied binding state.

## Why This Helps

- `ActorRuntimeContext` is now less environment-shaped and more runtime-shaped.
- The actor runtime reads environment semantics from the bindings protocol directly, which makes the dependency boundary clearer.
- This reduces the amount of LIBERO-specific structure implicitly smeared into the generic actor context.
- A future `franka_real` binding can satisfy the same protocol without also needing to mirror a pile of copied context fields.

## Scope Notes

- This phase does **not** change launcher/tooling.
- This phase does **not** yet make learner-side code consume the protocol everywhere.
- This phase does **not** yet introduce residual algorithm abstraction.
- The change is intentionally focused on actor-side runtime access patterns and context ownership.

## Self Review

- Ran `python -m py_compile` on:
  - [actor_support.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_support.py)
  - [actor_warmup.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py)
  - [actor_loop.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py)
  - [actor_setup.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py)
- Ran `git diff --check`.
- Verified there are no remaining actor-runtime references to the removed cached fields:
  - `ctx.image_keys`
  - `ctx.obs_cache`
  - `ctx.task_key`
  - `ctx.data_config`
  - `ctx.build_residual_step_obs_profiled`
  - `ctx.build_residual_step_core`
- Ran import smoke tests in the `serl_torch` conda env for:
  - `serl_launcher.residual.runtime.actor_support`
  - `serl_launcher.residual.runtime.actor_warmup`
  - `serl_launcher.residual.runtime.actor_loop`

## Commit Summary

- Stop caching binding-derived environment fields inside `ActorRuntimeContext`
- Route actor warmup and rollout helpers through `ResidualRuntimeBindings`
- Tighten actor-side ownership around the shared runtime bindings protocol
