# LIBERO Thin Train Loop V9

## Goal

Complete the first post-thin-loop phase: remove direct OpenPI concrete imports from the residual actor runtime so the runtime depends on a chunk-policy interface and factory instead of a single backend implementation.

## Changes

- Added policy-side runtime contracts in [serl_launcher/serl_launcher/policy/base.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/policy/base.py):
  - `PolicyInferResult`
  - `PolicyPrefetcher`
- Added [serl_launcher/serl_launcher/policy/factory.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/policy/factory.py) with:
  - `resolve_policy_backend_type(...)`
  - `build_policy_client(...)`
  - `build_policy_prefetcher(...)`
  - `build_policy_backend_info(...)`
- Updated [serl_launcher/serl_launcher/residual/runtime/actor_setup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py) to build the chunk policy client and prefetcher through the factory instead of importing OpenPI concrete classes.
- Updated [serl_launcher/serl_launcher/residual/runtime/actor_loop.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_loop.py) to use `ctx.policy_client` and `ctx.policy_prefetcher`.
- Updated [serl_launcher/serl_launcher/residual/runtime/actor_warmup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_warmup.py) to use the same generic policy handles.
- Fixed the small trailing-newline-only diff in [serl_launcher/serl_launcher/residual/runtime/actor_runtime.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_runtime.py).

## Why This Helps

- The actor runtime no longer directly imports `OpenPIPolicyClient` or `AsyncOpenPIPolicyPrefetcher`.
- Future chunk backends such as `StarVLA` can be added behind the policy factory without changing actor loop orchestration.
- This keeps the current stable assumption intact: chunk inference remains the runtime contract.

## Self Review

- Ran `python -m py_compile` on the modified policy and runtime files.
- Ran a small import/config smoke test for `serl_launcher.policy.factory`.
- Confirmed there are no remaining `OpenPIPolicyClient` or `AsyncOpenPIPolicyPrefetcher` imports under `serl_launcher/serl_launcher/residual/runtime`.
- Ran `git diff --check`.

## Commit Summary

- Introduce policy factory and prefetch protocol for residual actor runtime
- Replace direct OpenPI runtime dependencies with generic chunk policy handles
- Preserve current `exp11` behavior while making future backend swaps cheaper
