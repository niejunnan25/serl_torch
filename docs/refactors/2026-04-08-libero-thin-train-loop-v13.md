# LIBERO Thin Train Loop V13

## Goal

Align the standalone learner path with the shared bindings model, without forcing the learner process to construct a full runtime environment.

## Changes

- Updated [serl_launcher/serl_launcher/residual/runtime/bindings.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/bindings.py) to split bindings into two layers:
  - `ResidualDataBindings`
  - `ResidualRuntimeBindings`
- `ResidualRuntimeBindings` now extends `ResidualDataBindings`, so actor-side runtime still consumes the full environment-facing contract while learner-side code can depend only on the smaller data/task-facing subset.
- Updated [examples/libero/runtime/runtime_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py):
  - added `LiberoDataBindings`
  - added `build_libero_data_bindings(...)`
  - made `LiberoRuntimeBindings` inherit from `LiberoDataBindings`
  - reused the same data-binding fields when constructing the full runtime bindings
- Updated [serl_launcher/serl_launcher/residual/runtime/learner_service.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/learner_service.py) so the learner service now consumes `bindings: ResidualDataBindings` instead of separately receiving:
  - `data_config`
  - `resolve_cfg_image_keys(...)`
- Updated [examples/libero/scripts/run_learner.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/scripts/run_learner.py) to construct `build_libero_data_bindings(...)` and pass that into the learner service.
- Updated [examples/libero/runtime/__init__.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/__init__.py) to lazily expose:
  - `LiberoDataBindings`
  - `build_libero_data_bindings(...)`

## Why This Helps

- The learner path now depends on an explicit protocol instead of reconstructing LIBERO-specific metadata ad hoc.
- The learner does **not** need a full env object, so this keeps the protocol honest: learner needs task/data bindings, actor runtime needs full runtime bindings.
- This makes future environment expansion cleaner:
  - `franka_real` can provide a `DataBindings` implementation for learner/materialization flows
  - and a `RuntimeBindings` implementation for actor/eval flows

## Scope Notes

- This phase does **not** change launcher/tooling.
- This phase does **not** yet abstract the residual algorithm.
- This phase does **not** yet generalize learner bootstrap contents beyond the existing actor-written payload.
- This phase is specifically about learner-side protocol alignment and reducing LIBERO-specific metadata reconstruction.

## Self Review

- Ran `python -m py_compile` on:
  - [bindings.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/bindings.py)
  - [learner_service.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/learner_service.py)
  - [runtime_bindings.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py)
  - [runtime/__init__.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/runtime/__init__.py)
  - [run_learner.py](/vla/users/niejunnan/codebase/serl_torch/examples/libero/scripts/run_learner.py)
- Ran `git diff --check`.
- Ran learner entrypoint smoke test:
  - `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- Verified the learner service now reads:
  - `image_keys`
  - `task_key`
  - `data_config`
  from bindings instead of reconstructing them internally.

## Commit Summary

- Introduce a smaller learner/data-facing bindings protocol
- Reuse LIBERO binding metadata across actor and learner entrypoints
