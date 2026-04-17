# LIBERO Thin Train Loop V13 Fix

## Goal

Fix the remaining boundary problem from V13: learner/data entrypoints should not pull in full env/runtime dependencies by importing `runtime_bindings.py` at module import time.

## Problem

After V13, `run_learner.py` depended on `build_libero_data_bindings(...)`, but that function still lived in [runtime_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py).

That meant the learner path still imported:
- env factory code
- runtime observation adapters
- policy adapter code

at module import time, even though learner/data flows only need task/data metadata.

## Changes

- Added [data_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/data_bindings.py) as a physically separate, lightweight module for:
  - `LiberoDataBindings`
  - `build_libero_data_bindings(...)`
- Updated [runtime_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py) to import the data-binding pieces from `data_bindings.py` instead of defining them inline.
- Updated [run_learner.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_learner.py) to import `build_libero_data_bindings(...)` from the lightweight module.
- Updated [runtime/__init__.py](/home/hello/codebase/serl_torch/examples/libero/runtime/__init__.py) so:
  - data-binding names are lazily loaded from `data_bindings.py`
  - runtime-binding names are lazily loaded from `runtime_bindings.py`

## Why This Helps

- Learner/data paths now truly depend only on learner/data bindings.
- Actor/env paths still use the full runtime bindings module.
- This makes the split between:
  - `ResidualDataBindings`
  - `ResidualRuntimeBindings`
  real in the import graph, not just in type signatures.

## Self Review

- Ran `python -m py_compile` on:
  - [data_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/data_bindings.py)
  - [runtime_bindings.py](/home/hello/codebase/serl_torch/examples/libero/runtime/runtime_bindings.py)
  - [runtime/__init__.py](/home/hello/codebase/serl_torch/examples/libero/runtime/__init__.py)
  - [run_learner.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_learner.py)
- Ran `git diff --check`.
- Ran:
  - `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- Verified that [run_learner.py](/home/hello/codebase/serl_torch/examples/libero/scripts/run_learner.py) no longer imports from `runtime_bindings.py`.

## Commit Summary

- Physically separate LIBERO data bindings from runtime bindings
- Keep learner/data entrypoints off the full env/runtime import path
- Make the V13 protocol split real in the module graph
