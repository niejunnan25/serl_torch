# LIBERO Thin Train Loop V10

## Goal

Extend the first chunk-policy abstraction past the actor runtime so evaluation and residual-training data materialization also stop depending on OpenPI concrete classes directly, while introducing `policy.type` and `policy.id` as lightweight backend semantics.

## Changes

- Extended [serl_launcher/serl_launcher/policy/factory.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/policy/factory.py) with:
  - `resolve_policy_backend_id(...)`
  - `build_policy_backend_info(...)` now returning both `type` and `id`
  - null-safe fallback behavior so `policy.id: null` resolves back to the backend type
- Updated [serl_launcher/serl_launcher/residual/runtime/actor_setup.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_setup.py) logging to surface both backend type and backend id.
- Updated [examples/libero/scripts/eval_residual_fast.py](/home/hello/codebase/serl_torch/examples/libero/scripts/eval_residual_fast.py) to:
  - build the chunk policy through the policy factory
  - stop importing `OpenPIPolicyClient` directly
  - write `base_policy_type` and `base_policy_id` into the eval summary
- Updated [examples/libero/scripts/materialize_residual_training_online.py](/home/hello/codebase/serl_torch/examples/libero/scripts/materialize_residual_training_online.py) to:
  - build the chunk policy through the policy factory
  - record `base_policy_type` and `base_policy_id` in episode metadata and manifest metadata
  - keep `openpi_host/openpi_port` in manifest metadata only as backend-specific compatibility fields when `policy.type=openpi`
- Updated [examples/libero/scripts/materialize_residual_training_offline.py](/home/hello/codebase/serl_torch/examples/libero/scripts/materialize_residual_training_offline.py) to:
  - add lightweight `--policy_type` and `--policy_id` CLI flags
  - construct a temporary policy config and build the chunk policy through the factory
  - record `base_policy_type` and `base_policy_id` in episode metadata and manifest metadata
  - keep `openpi_host/openpi_port` in manifest metadata only for the OpenPI backend
- Added default `policy` blocks to:
  - [examples/libero/conf/train_residual_sac.yaml](/home/hello/codebase/serl_torch/examples/libero/conf/train_residual_sac.yaml)
  - [examples/libero/conf/eval_residual_fast.yaml](/home/hello/codebase/serl_torch/examples/libero/conf/eval_residual_fast.yaml)
- Added `policy.id: pi05_libero` to the current `exp11` pi05 configs:
  - [libero_10_task_8_chunk_async_null_pi05.yaml](/home/hello/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_pi05.yaml)
  - [libero_10_task_8_chunk_async_null_unfreeze_pi05.yaml](/home/hello/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_unfreeze_pi05.yaml)
  - [libero_10_task_8_chunk_async_null_utdeff1p_pi05.yaml](/home/hello/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_utdeff1p_pi05.yaml)
  - [libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml](/home/hello/codebase/serl_torch/examples/libero/configs/exp11/chunk/libero_10_task_8_chunk_async_null_utdeff1p_unfreeze_pi05.yaml)
- Fixed a pre-existing circular import trap by making [examples/libero/runtime/__init__.py](/home/hello/codebase/serl_torch/examples/libero/runtime/__init__.py) lazily expose `runtime_bindings`, and by switching the modified scripts to import concrete runtime submodules directly instead of going through the package re-export.

## Why This Helps

- Evaluation and both residual-training materializers now follow the same backend dispatch contract as the actor runtime.
- `policy.type` handles backend family selection and `policy.id` provides a stable trace label without forcing launcher/tool changes yet.
- Residual datasets and eval summaries now explicitly record which base policy instance generated them.
- The runtime package no longer blocks CLI entrypoints with the `training_config <-> runtime_bindings` circular import.

## Self Review

- Ran `python -m py_compile` on all modified files.
- Ran `git diff --check`.
- Ran a small `build_policy_backend_info(...)` smoke test with:
  - `policy.id='pi05_libero'`
  - `policy.id=null`
- Ran `materialize_residual_training_offline.py --help` with `PYTHONPATH=/.../serl_launcher` to confirm the new CLI path and to verify the circular import was resolved.

## Known Gaps

- Launcher and shell tools were intentionally left unchanged in this phase.
- `materialize_residual_training_online.py --help` still depends on `hydra` being installed in the active environment; this phase did not change that dependency story.
- Only the currently used `exp11` pi05 configs were labeled with `policy.id`; configs without a `policy` block still fall back to `id == type`.

## Commit Summary

- Thread `policy.type` and `policy.id` through eval and residual data materialization
- Replace remaining OpenPI concrete imports in the targeted LIBERO Python entrypoints
- Record base-policy identity in eval summaries and residual-training manifests
