# V16: Split Actor Episode Execution by Mode

## What changed
- Split the large episode executor in
  [actor_episode.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_episode.py)
  into three mode-oriented modules:
  - [actor_episode_shared.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_episode_shared.py)
  - [actor_episode_chunk.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_episode_chunk.py)
  - [actor_episode_step.py](/home/hello/codebase/serl_torch/serl_launcher/serl_launcher/residual/runtime/actor_episode_step.py)
- Kept `run_policy_episode(...)` as the stable top-level entrypoint and moved only
  internal episode execution bodies.
- Centralized shared episode helpers for:
  - replay insertion
  - replay sampling / sync update execution
  - update metric logging
  - profiling snapshot flushing
  - final episode result assembly

## Why
- `actor_loop.py` had already become mostly orchestration in V15, but
  `actor_episode.py` still carried both chunk-mode and step-mode rollout logic in one
  very large function.
- Splitting by stable execution semantics makes the next actor-side cleanup easier:
  shared helpers live in one place, while chunk and step execution can evolve more
  independently.

## Review
- Re-ran:
  - `python -m py_compile serl_launcher/serl_launcher/residual/runtime/actor_episode.py serl_launcher/serl_launcher/residual/runtime/actor_episode_shared.py serl_launcher/serl_launcher/residual/runtime/actor_episode_chunk.py serl_launcher/serl_launcher/residual/runtime/actor_episode_step.py serl_launcher/serl_launcher/residual/runtime/actor_loop.py`
  - `git diff --check`
  - `conda run -n serl_torch python examples/libero/scripts/train_residual_sac.py --help`
  - `conda run -n serl_torch python examples/libero/scripts/run_learner.py --help`
- During self-review I fixed two regressions before commit:
  - restored `agent_sample_actions` profiling on the chunk path
  - preserved the old "prefetch queue returned no batch -> skip this update" behavior
    instead of raising an exception

## Outcome
- `actor_episode.py` is now the thin per-episode orchestrator.
- Chunk-mode and step-mode execution are isolated behind explicit helpers.
- Shared training-update / profiling code is no longer duplicated across both paths.
