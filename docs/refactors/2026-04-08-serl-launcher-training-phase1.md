# Phase 1: Introduce `training/` and `residual/algorithms/`

## Goal
- Start reducing the overload in
  [serl_launcher/serl_launcher/residual/runtime/](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime)
  by moving the cleanest files into better-scoped top-level packages.

## Changes
- Added:
  - [serl_launcher/serl_launcher/training/__init__.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/training/__init__.py)
  - [serl_launcher/serl_launcher/residual/algorithms/__init__.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/algorithms/__init__.py)
- Moved training infrastructure:
  - `residual/runtime/checkpoint.py` -> [training/checkpoint.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/training/checkpoint.py)
  - `residual/runtime/profiling.py` -> [training/profiling.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/training/profiling.py)
  - `residual/runtime/replay_batch.py` -> [training/replay_batch.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/training/replay_batch.py)
- Moved residual algorithm layer:
  - `residual/runtime/algorithm.py` -> [residual/algorithms/base.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/algorithms/base.py)
  - `residual/runtime/sac_algorithm.py` -> [residual/algorithms/sac.py](/vla/users/niejunnan/codebase/serl_torch/serl_launcher/serl_launcher/residual/algorithms/sac.py)
- Updated all Python imports to the new locations.

## Why these files first
- They are the least ambiguous files in the current structure:
  - `checkpoint / profiling / replay_batch` are training infrastructure
  - `algorithm / sac_algorithm` are residual algorithm definitions
- This phase changes package boundaries without changing actor/learner behavior.

## Review
- Re-ran:
  - `python -m py_compile serl_launcher/serl_launcher/training/checkpoint.py serl_launcher/serl_launcher/training/profiling.py serl_launcher/serl_launcher/training/replay_batch.py serl_launcher/serl_launcher/residual/algorithms/base.py serl_launcher/serl_launcher/residual/algorithms/sac.py serl_launcher/serl_launcher/residual/runtime/actor_setup.py serl_launcher/serl_launcher/residual/runtime/actor_loop.py serl_launcher/serl_launcher/residual/runtime/async_learning.py serl_launcher/serl_launcher/residual/runtime/learner_service.py examples/libero/runtime/__init__.py examples/libero/runtime/runtime_bindings.py`
  - `git diff --check`
  - `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`
  - `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- Focused review:
  - confirmed no remaining Python imports to the old `residual.runtime.{algorithm,sac_algorithm,checkpoint,profiling,replay_batch}` paths
  - updated file headers in the moved `training/` files so the new package semantics match the code

## Outcome
- `serl_launcher` now has a real top-level `training/` package.
- `residual/algorithms/` is established as a dedicated home for residual algorithm adapters.
- The next migration phase can move residual train orchestration without carrying these infrastructure files inside `residual/runtime/`.
